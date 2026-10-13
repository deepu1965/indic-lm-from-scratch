"""
sft_dataset.py
==============
SFT dataset and collator for Phase 3.2 supervised fine-tuning.

Sequence format
---------------
    [BOS]  input_text_tokens  [EOS]  target_text_tokens  [EOS]
     ID=2                      ID=3                       ID=3

The training follows the same label-shifting convention as pretraining:

    raw_seq   = [BOS, i1, i2, ..., iK, EOS_sep, t1, t2, ..., tM, EOS_final]
    input_ids = raw_seq[:-1]   length: K + M + 2
    labels    = raw_seq[1:]    length: K + M + 2

Loss mask (aligned with LABEL positions):
    labels index   0 .. K-1    → input tokens              → mask = 0
    labels index   K           → EOS separator             → mask = 0
    labels index   K+1 .. K+M  → target tokens             → mask = 1
    labels index   K+M+1       → final [EOS]               → mask = 1
    any padding positions       → mask = 0  (label = PAD_ID)

Derivation:
    labels[i] = raw_seq[i+1]
    The EOS separator is at raw_seq[K+1], so labels[K] = EOS_sep.
    Everything after labels[K] → target tokens + final EOS → mask = 1.

Usage
-----
    from phase3.scripts.training.sft_dataset import SFTDataset, SFTCollator, masked_cross_entropy

    ds = SFTDataset("phase3/outputs/reasoning_hi.jsonl",
                    tokenizer_path="hindi/tokenizer/models/bpe_v10000.json",
                    split="train",
                    max_seq_len=512)
    collator = SFTCollator(pad_id=1)
    loader = DataLoader(ds, batch_size=4, collate_fn=collator)

    for batch in loader:
        logits = model(batch["input_ids"])
        loss = masked_cross_entropy(logits, batch["labels"], batch["loss_mask"])
"""

from __future__ import annotations

import json
from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizers import Tokenizer


# ── Special token IDs (matching bpe_v10000.json for both Hindi and Nepali) ────
_PAD_ID: int = 1
_BOS_ID: int = 2
_EOS_ID: int = 3


class SFTDataset(Dataset):
    """
    Map-style dataset for SFT over the Phase 3.2 reasoning JSONL files.

    Each record produces one (input_ids, labels, loss_mask) triple.

    The raw token sequence is built as:
        [BOS]  +  encode(input_text)  +  [EOS]  +  encode(target_text)  +  [EOS]

    input_ids and labels are both of length len(raw_seq) - 1, following the
    same next-token-prediction convention used during pretraining.

    Sequences longer than max_seq_len tokens (after slicing into input_ids /
    labels) are silently discarded.  This avoids truncating mid-target, which
    would corrupt the loss mask.

    Parameters
    ----------
    jsonl_path : str
        Path to reasoning_hi.jsonl or reasoning_ne.jsonl.
    tokenizer_path : str
        Path to the frozen BPE tokenizer JSON file.
    split : str or None
        If given (e.g. "train", "validation", "test"), only records with
        the matching "split" field are loaded.  If None, all records are used.
    max_seq_len : int
        Maximum allowed length of input_ids / labels.  Sequences that would
        exceed this after tokenisation are dropped.  Default: 512 (matches
        the pretraining context_length).
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer_path: str,
        split: Optional[str] = "train",
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self._records: List[Dict] = []
        self._dropped: int = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                # ── Filter by split ───────────────────────────────────────────
                if split is not None and record.get("split") != split:
                    continue

                item = self._process_record(record)
                if item is None:
                    self._dropped += 1
                else:
                    self._records.append(item)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, text: str) -> List[int]:
        """Encode text with the frozen BPE tokenizer (no special tokens added)."""
        return self.tokenizer.encode(text).ids

    def _process_record(
        self, record: dict
    ) -> Optional[Dict]:
        """
        Convert one JSONL record into (input_ids, labels, loss_mask) tensors.

        Returns None if the resulting sequence exceeds max_seq_len.
        """
        input_tok: List[int] = self._encode(record["input_text"])
        target_tok: List[int] = self._encode(record["target_text"])

        # ── Build raw sequence ────────────────────────────────────────────────
        #
        #   raw_seq = [BOS] + input_tok + [EOS] + target_tok + [EOS]
        #
        # Indices in raw_seq:
        #   0                          → [BOS]
        #   1 .. K                     → input tokens         (K = len(input_tok))
        #   K + 1                      → [EOS] separator
        #   K + 2 .. K + 1 + M        → target tokens        (M = len(target_tok))
        #   K + 2 + M                  → [EOS] final
        #
        # Total length: 1 + K + 1 + M + 1 = K + M + 3
        K: int = len(input_tok)
        M: int = len(target_tok)
        raw_seq: List[int] = (
            [_BOS_ID]
            + input_tok
            + [_EOS_ID]
            + target_tok
            + [_EOS_ID]
        )

        # ── Length guard ──────────────────────────────────────────────────────
        # input_ids and labels will each have length len(raw_seq) - 1.
        seq_len: int = len(raw_seq) - 1
        if seq_len > self.max_seq_len:
            return None

        # ── Apply the next-token-prediction shift ────────────────────────────
        #
        #   input_ids = raw_seq[:-1]   (feed token at position t)
        #   labels    = raw_seq[1:]    (predict token at position t+1)
        #
        # This exactly matches the pretraining convention:
        #   input_ids = window[:context_length]
        #   labels    = window[1:context_length+1]
        input_ids = torch.tensor(raw_seq[:-1], dtype=torch.long)
        labels    = torch.tensor(raw_seq[1:],  dtype=torch.long)

        # ── Build loss mask (aligned with LABEL positions) ───────────────────
        #
        # labels[i] = raw_seq[i + 1]
        #
        # We want mask = 1 only where the label is a TARGET token or EOS_final.
        # The separator [EOS] is at raw_seq[K+1], so it appears in labels at
        # index K  (because labels[K] = raw_seq[K+1] = EOS_sep).
        #
        # Mask derivation:
        #   labels index  0 ..  K-1  → raw_seq[1..K]   = input tokens  → 0
        #   labels index  K           → raw_seq[K+1]    = EOS_sep       → 0
        #   labels index  K+1 .. K+M  → raw_seq[K+2..]  = target tokens → 1
        #   labels index  K+M+1       → raw_seq[K+M+2]  = final EOS     → 1
        #
        loss_mask = torch.zeros(seq_len, dtype=torch.long)
        # Everything after the separator (index K) gets mask = 1.
        loss_mask[K + 1:] = 1

        # Sanity: mask should cover exactly M target tokens + 1 final EOS.
        assert loss_mask.sum().item() == M + 1, (
            f"Mask sum {loss_mask.sum().item()} != M+1={M+1}"
        )

        return {
            "input_ids":  input_ids,
            "labels":     labels,
            "loss_mask":  loss_mask,
            # Preserve metadata for inspection / evaluation
            "input_text":        record["input_text"],
            "target_text":       record["target_text"],
            "split":             record.get("split", ""),
            "task_family":       record.get("task_family", ""),
            "num_input_tokens":  K,
            "num_target_tokens": M,
        }

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        return self._records[idx]

    def __repr__(self) -> str:
        return (
            f"SFTDataset(\n"
            f"  records={len(self._records)},\n"
            f"  dropped_too_long={self._dropped},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f")"
        )


class SFTCollator:
    """
    Collate a list of SFT records into padded batch tensors.

    Pads input_ids, labels, and loss_mask to the length of the longest
    sequence in the batch.  Padding positions receive:
        input_ids  → pad_id (1 = [PAD])
        labels     → pad_id (masked via ignore_index in masked_cross_entropy)
        loss_mask  → 0     (padding never contributes to loss)

    Parameters
    ----------
    pad_id : int
        Token ID used for padding.  Default: 1 ([PAD]).

    Returns
    -------
    dict with keys:
        "input_ids"  : LongTensor (B, T_max)
        "labels"     : LongTensor (B, T_max)
        "loss_mask"  : LongTensor (B, T_max)
    """

    def __init__(self, pad_id: int = _PAD_ID) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len: int = max(item["input_ids"].size(0) for item in batch)

        padded_input_ids: List[torch.Tensor] = []
        padded_labels:    List[torch.Tensor] = []
        padded_masks:     List[torch.Tensor] = []

        for item in batch:
            L: int = item["input_ids"].size(0)
            pad: int = max_len - L

            padded_input_ids.append(
                F.pad(item["input_ids"], (0, pad), value=self.pad_id)
            )
            padded_labels.append(
                F.pad(item["labels"], (0, pad), value=self.pad_id)
            )
            # loss_mask is always 0 at padding positions
            padded_masks.append(
                F.pad(item["loss_mask"], (0, pad), value=0)
            )

        return {
            "input_ids": torch.stack(padded_input_ids),   # (B, T_max)
            "labels":    torch.stack(padded_labels),       # (B, T_max)
            "loss_mask": torch.stack(padded_masks),        # (B, T_max)
        }


# ── Loss function ─────────────────────────────────────────────────────────────


def masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss computed ONLY over positions where loss_mask == 1.

    Drop-in replacement for ``F.cross_entropy`` in the SFT training loop.
    Computes per-token losses with ``reduction='none'``, applies the binary
    mask, and returns the mean over active positions only.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, T, vocab_size)
        Raw model output logits.
    labels : torch.Tensor, shape (B, T)
        Token IDs to predict (already shifted relative to input_ids).
    loss_mask : torch.Tensor, shape (B, T)
        Binary mask: 1 = include in loss, 0 = exclude.

    Returns
    -------
    torch.Tensor
        Scalar mean loss over masked-in positions.

    Raises
    ------
    ValueError
        If no positions are active (all mask entries are 0).
    """
    B, T, V = logits.shape

    # Per-token cross-entropy without reduction.
    # ignore_index=_PAD_ID means [PAD] labels never accumulate gradients even
    # if loss_mask were accidentally 1 at a padding position.
    per_token_loss = F.cross_entropy(
        logits.view(B * T, V),
        labels.view(B * T),
        reduction="none",
        ignore_index=_PAD_ID,
    ).view(B, T)   # (B, T)

    mask = loss_mask.float()
    masked_loss = (per_token_loss * mask).sum()
    num_active = mask.sum()

    if num_active == 0:
        raise ValueError(
            "masked_cross_entropy: all loss_mask entries are 0. "
            "No target tokens found in this batch — check sequence construction."
        )

    return masked_loss / num_active
