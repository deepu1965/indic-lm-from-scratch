"""
dataset.py
==========
Streaming, packed PyTorch IterableDataset for causal language model pretraining.

Overview
--------
The Hindi training corpus (``final_train_500M.jsonl``) contains 514,608
documents totalling ~500 M tokens.  Each record has:

    {
        "title": str,
        "source": str,
        "tokens": [int, int, ...]   ← raw BPE token IDs in [0, 9999]
    }

No BOS or EOS tokens exist in the stored token lists.

This module provides:

1. ``StreamingPackedDataset`` — an IterableDataset that:
   - Streams the file line-by-line (O(max_document_length + context_length) RAM).
   - Wraps every document as  [BOS] + tokens + [EOS].
   - Concatenates documents into one continuous token stream.
   - Emits non-overlapping (input_ids, labels) windows of length 512.

2. ``create_dataloader`` — a thin factory that wraps the dataset in a
   ``torch.utils.data.DataLoader`` with sane Kaggle-compatible defaults.

Sequence construction
---------------------
Each training example consumes a window of 513 tokens from the buffer:

    window    = buffer[:513]
    input_ids = window[:512]   shape (512,)
    labels    = window[1:513]  shape (512,)

After yielding, the buffer is advanced by exactly 512 tokens:

    del buffer[:512]

This keeps token 512 in the buffer as the first input token of the NEXT
example — ensuring seamless next-token-prediction transitions across the
full stream with zero gaps.

Window diagram::

    buffer index:  0   1   2   3 … 511   512   513 …
    input_ids:     ●   ●   ●   ● …  ●
    labels:            ●   ●   ●  …  ●     ●
                                           ↑
                                  this stays → start of next input

Checkpoint / resume
-------------------
The dataset tracks resumable state via instance variables.  After each
``yield``, the variables ``_snapshot_offset`` and ``_snapshot_buffer`` are
updated with the byte-offset into the JSONL file and the remaining token
buffer.  The training loop can call ``dataset.state_dict()`` at any time to
capture this state for inclusion in a model checkpoint.

To resume from a checkpoint:

    dataset = StreamingPackedDataset(path, context_length=512)
    dataset.load_state_dict(checkpoint["dataset_state"])
    dataloader = create_dataloader(dataset, batch_size=32)
    for batch in dataloader:
        ...   # continues exactly from where training stopped

Usage
-----
    from hindi.train.dataset import StreamingPackedDataset, create_dataloader

    train_ds = StreamingPackedDataset(
        jsonl_path="hindi/corpus/tokenised_splits/final_train_500M.jsonl",
        context_length=512,
        bos_id=2,
        eos_id=3,
    )
    train_loader = create_dataloader(train_ds, batch_size=32)

    for batch in train_loader:
        input_ids = batch["input_ids"]   # (batch_size, 512)
        labels    = batch["labels"]      # (batch_size, 512)
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

import torch
from torch.utils.data import DataLoader, IterableDataset


# ─── Special token IDs (matching hindi/tokenizer/models/bpe_v10000.json) ──────
_DEFAULT_BOS_ID: int = 2   # [BOS]
_DEFAULT_EOS_ID: int = 3   # [EOS]


class StreamingPackedDataset(IterableDataset):
    """
    Memory-efficient streaming IterableDataset for causal LM pretraining.

    Streams a JSONL file where each record must contain a ``"tokens"`` field
    (``list[int]``).  Documents are wrapped with BOS/EOS and packed into a
    continuous token stream.  Fixed-length windows are emitted without padding.

    Parameters
    ----------
    jsonl_path : str
        Absolute or relative path (from repo root) to the JSONL training or
        validation file.
    context_length : int
        Number of tokens in each emitted ``input_ids`` and ``labels`` tensor.
        Default: 512.
    bos_id : int
        Token ID prepended to every document.  Default: 2 ([BOS]).
    eos_id : int
        Token ID appended to every document.  Default: 3 ([EOS]).

    Yields
    ------
    dict with keys:
        "input_ids" : torch.LongTensor, shape (context_length,)
        "labels"    : torch.LongTensor, shape (context_length,)

    Notes
    -----
    - ``num_workers=0`` is required for correct state tracking (see
      ``create_dataloader``).  Multi-worker support can be added later via
      ``worker_init_fn`` file-sharding.
    - No shuffling is applied in this implementation.  Document order follows
      the JSONL file order.
    - Partial final buffers (fewer than context_length + 1 tokens remaining
      after the last document) are silently dropped — standard practice.
    """

    def __init__(
        self,
        jsonl_path: str,
        context_length: int = 512,
        bos_id: int = _DEFAULT_BOS_ID,
        eos_id: int = _DEFAULT_EOS_ID,
    ) -> None:
        super().__init__()

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"JSONL file not found: {jsonl_path!r}\n"
                "Run from the repository root, e.g.:\n"
                "  python -m hindi.train.validate_dataset"
            )
        if context_length <= 0:
            raise ValueError(
                f"context_length must be a positive integer, got {context_length}"
            )

        self.jsonl_path = jsonl_path
        self.context_length = context_length
        self.bos_id = bos_id
        self.eos_id = eos_id

        # ── Resume anchors (only ever modified by load_state_dict) ───────────
        # When __iter__ starts, it seeks to _resume_offset and pre-populates
        # the buffer with _resume_buffer.  Both default to "start of file /
        # empty buffer", so without load_state_dict() iteration begins fresh.
        self._resume_offset: int = 0
        self._resume_buffer: List[int] = []

        # ── Running counters (updated inside __iter__) ────────────────────────
        # These accumulate across the lifetime of the object (not per epoch).
        self._docs_processed: int = 0
        self._chunks_yielded: int = 0
        self._skipped_records: int = 0

        # ── Resumable snapshot (updated before each yield) ────────────────────
        # _snapshot_offset : byte offset in the JSONL file AFTER reading the
        #                    last document that contributed to the current buffer.
        # _snapshot_buffer : the rolling token buffer state immediately AFTER
        #                    the most-recently yielded chunk was removed.
        #
        # Invariant: if a new StreamingPackedDataset is constructed with the
        # same path/config and load_state_dict(self.state_dict()) is called,
        # then iter(new_dataset) will yield the same sequence of examples as
        # continuing the current iterator would.
        self._snapshot_offset: int = 0
        self._snapshot_buffer: List[int] = []

    # ── Core iteration ────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Generator that streams and packs documents into training examples.

        Data flow per iteration
        -----------------------
        1. Open JSONL file, seek to resume offset (default: 0).
        2. Read one line → parse JSON → validate ``tokens`` field.
        3. Append [BOS] + tokens + [EOS] to rolling buffer.
        4. While buffer has ≥ context_length + 1 tokens:
               input_ids = buffer[:context_length]
               labels    = buffer[1 : context_length + 1]
               del buffer[:context_length]   ← advance by context_length only
               snapshot state
               yield {"input_ids": ..., "labels": ...}
        5. Repeat from step 2.

        Buffer size is bounded by:
            max_doc_length + context_length + 2   (BOS + EOS)
        which for this corpus is approximately  41776 + 512 + 2 = ~42K integers
        ≈ 0.3 MB.  This is constant regardless of total corpus size.
        """
        # Restore resume state.  On a fresh dataset (no load_state_dict call),
        # this starts an empty buffer at byte offset 0.
        buffer: List[int] = list(self._resume_buffer)

        # Each iteration reads CONTEXT_LENGTH + 1 tokens to emit one example.
        window: int = self.context_length + 1   # 513 for context_length=512

        with open(self.jsonl_path, "r", encoding="utf-8") as f:

            # Seek to resume position (0 = start of file by default).
            if self._resume_offset > 0:
                f.seek(self._resume_offset)

            # Track the byte offset AFTER each successfully-parsed document.
            # We snapshot this before yielding so resume works correctly.
            current_file_offset: int = f.tell()

            while True:
                # ── Read one line ─────────────────────────────────────────────
                line: str = f.readline()
                if not line:
                    # EOF — stop the generator.
                    break

                stripped: str = line.strip()
                if not stripped:
                    # Blank line — advance offset and skip.
                    current_file_offset = f.tell()
                    continue

                # ── Parse JSON ────────────────────────────────────────────────
                try:
                    record: dict = json.loads(stripped)
                except json.JSONDecodeError:
                    # Malformed record — count and skip.
                    self._skipped_records += 1
                    current_file_offset = f.tell()
                    continue

                # ── Validate tokens field ─────────────────────────────────────
                tokens = record.get("tokens")
                if not isinstance(tokens, list) or len(tokens) == 0:
                    # Missing, non-list, or empty token list — skip.
                    self._skipped_records += 1
                    current_file_offset = f.tell()
                    continue

                # ── Wrap document and extend buffer ───────────────────────────
                # Buffer layout after appending one document:
                #   [...existing..., BOS, t0, t1, ..., tN, EOS]
                buffer.append(self.bos_id)   # prepend BOS
                buffer.extend(tokens)        # document body
                buffer.append(self.eos_id)   # append EOS

                self._docs_processed += 1

                # Advance file cursor AFTER successfully reading this document.
                current_file_offset = f.tell()

                # ── Emit packed windows ───────────────────────────────────────
                # Emit as many full windows as possible from the current buffer.
                # Each window consumes CONTEXT_LENGTH tokens (not CONTEXT_LENGTH+1)
                # so that the last token of this window becomes the first input
                # token of the next window (seamless prediction across docs).
                while len(buffer) >= window:

                    # Slice input_ids and labels from the buffer.
                    #
                    # input_ids: buffer[0 .. context_length-1]  (indices 0..511)
                    # labels:    buffer[1 .. context_length]     (indices 1..512)
                    #
                    # Shape of each: (context_length,) = (512,)
                    input_ids: torch.Tensor = torch.tensor(
                        buffer[:self.context_length], dtype=torch.long
                    )
                    labels: torch.Tensor = torch.tensor(
                        buffer[1:window], dtype=torch.long
                    )

                    # Advance the buffer by exactly context_length positions.
                    # buffer[context_length] (= buffer[512]) remains as the
                    # first element of the buffer → becomes input_ids[0] of
                    # the next example, completing the prediction chain.
                    del buffer[:self.context_length]

                    self._chunks_yielded += 1

                    # ── Snapshot resumable state before yield ─────────────────
                    # After this yield, the consumer (training loop) may
                    # checkpoint.  The snapshot captures:
                    #   - current_file_offset: byte pos after last doc read
                    #   - buffer: remaining tokens not yet yielded
                    # On resume: seek to offset, restore buffer, resume reads.
                    self._snapshot_offset = current_file_offset
                    self._snapshot_buffer = list(buffer)

                    yield {
                        "input_ids": input_ids,   # (context_length,)
                        "labels":    labels,       # (context_length,)
                    }

        # Partial buffer at EOF is silently dropped (standard for packed LM).
        # The training loop can observe self._chunks_yielded for total count.

    # ── Checkpoint / resume API ───────────────────────────────────────────────

    def state_dict(self) -> dict:
        """
        Return a JSON-serialisable snapshot of the dataset's current state.

        This state is sufficient for exact resume: constructing a new
        ``StreamingPackedDataset`` with the same ``jsonl_path`` and
        ``context_length``, then calling ``load_state_dict(state)`` on it,
        will cause the next ``iter()`` call to continue from exactly the same
        position.

        Returns
        -------
        dict with keys:
            "file_offset"    : int       — byte position in the JSONL file
            "buffer"         : list[int] — remaining packed token buffer
            "docs_processed" : int       — total documents consumed
            "chunks_yielded" : int       — total (input_ids, labels) pairs yielded
            "skipped_records": int       — records skipped due to errors

        Notes
        -----
        Safe to call from the training loop between DataLoader batches when
        ``num_workers=0``.  The snapshot is updated before every ``yield``
        inside ``__iter__``, so the returned state always reflects the most
        recently yielded chunk.
        """
        return {
            "file_offset":     self._snapshot_offset,
            "buffer":          list(self._snapshot_buffer),
            "docs_processed":  self._docs_processed,
            "chunks_yielded":  self._chunks_yielded,
            "skipped_records": self._skipped_records,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore dataset state from a previously saved ``state_dict()``.

        Must be called **before** creating the DataLoader or calling
        ``iter(dataset)``.  Has no effect if called after iteration begins.

        Parameters
        ----------
        state : dict
            A dict previously returned by ``state_dict()``.

        Raises
        ------
        KeyError
            If any required key is missing from ``state``.
        """
        required: set = {
            "file_offset", "buffer", "docs_processed",
            "chunks_yielded", "skipped_records",
        }
        missing: set = required - set(state.keys())
        if missing:
            raise KeyError(
                f"load_state_dict() received an incomplete state dict. "
                f"Missing keys: {missing}"
            )

        # Set resume anchors used by __iter__ when it next starts.
        self._resume_offset = int(state["file_offset"])
        self._resume_buffer = list(state["buffer"])

        # Restore running counters.
        self._docs_processed  = int(state["docs_processed"])
        self._chunks_yielded  = int(state["chunks_yielded"])
        self._skipped_records = int(state["skipped_records"])

        # Sync snapshot so state_dict() immediately after load_state_dict()
        # returns the same values.
        self._snapshot_offset = self._resume_offset
        self._snapshot_buffer = list(self._resume_buffer)

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"StreamingPackedDataset(\n"
            f"  jsonl_path={self.jsonl_path!r},\n"
            f"  context_length={self.context_length},\n"
            f"  bos_id={self.bos_id},\n"
            f"  eos_id={self.eos_id},\n"
            f"  docs_processed={self._docs_processed},\n"
            f"  chunks_yielded={self._chunks_yielded},\n"
            f"  skipped_records={self._skipped_records},\n"
            f")"
        )


# ─── DataLoader factory ───────────────────────────────────────────────────────


def create_dataloader(
    dataset: StreamingPackedDataset,
    batch_size: int = 16,
    pin_memory: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a ``torch.utils.data.DataLoader`` for a ``StreamingPackedDataset``.

    Parameters
    ----------
    dataset : StreamingPackedDataset
        The dataset instance to wrap.
    batch_size : int
        Number of (input_ids, labels) pairs per batch.
        Default: 16.  Kaggle T4 (16 GB) can typically use 32–64 for this model.
    pin_memory : bool
        Pin batch tensors to page-locked memory for faster CPU → GPU transfer.
        Set to False if running on CPU only.
        Default: True.
    drop_last : bool
        Drop the final batch if it is smaller than ``batch_size``.
        Recommended: True — ensures all batches have identical shapes.
        Default: True.
    num_workers : int
        Number of subprocesses for data loading.  Must be 0 for this initial
        implementation: ``StreamingPackedDataset`` keeps the file handle and
        state in the main process.  Multi-worker sharding can be added later
        via ``worker_init_fn``.
        Default: 0.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of dicts::

            {
                "input_ids": torch.LongTensor, shape (batch_size, context_length),
                "labels":    torch.LongTensor, shape (batch_size, context_length),
            }

    Raises
    ------
    ValueError
        If ``num_workers > 0`` (not yet supported).
    """
    if num_workers != 0:
        raise ValueError(
            "num_workers > 0 is not yet supported for StreamingPackedDataset. "
            "Multi-worker sharding requires a worker_init_fn that splits the "
            "JSONL file across workers.  Use num_workers=0 for now."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # collate_fn is left as default: it stacks dicts of tensors into
        # batched dicts, which is exactly what we need.
    )
