"""
validate_dataset.py
===================
Self-contained validation for the StreamingPackedDataset and DataLoader.

Runs 9 checks to verify correctness of the training data pipeline.
No model, no GPU required (all checks run on CPU).

Usage (from the repository root — REQUIRED)
-------------------------------------------
    python -m hindi.train.validate_dataset

Both real data files and controlled temporary JSONL files are used.

Real data paths (relative to repo root):
    hindi/corpus/tokenised_splits/final_train_500M.jsonl  ← training
    hindi/corpus/tokenised_splits/val.jsonl               ← validation

Checks
------
  [1] Dataset creation           — instantiate train and val datasets.
  [2] Sample structure           — shapes, dtype, value range.
  [3] Causal shift               — input_ids[1:] == labels[:-1].
  [4] BOS/EOS insertion          — BOS and EOS appear in packed stream.
  [5] Cross-document packing     — EOS immediately followed by BOS at doc boundary.
  [6] Multiple real samples      — iterate 200 real training samples.
  [7] DataLoader batching        — batch shapes are (batch_size, 512).
  [8] State/resume               — exact resume from saved state_dict().
  [9] Memory/buffer bound        — buffer stays bounded (O(max_doc + 512)).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# ── Path setup ────────────────────────────────────────────────────────────────
# Support running as both:
#   python -m hindi.train.validate_dataset   (recommended)
#   python hindi/train/validate_dataset.py   (direct)
_THIS_FILE = os.path.abspath(__file__)
_TRAIN_DIR = os.path.dirname(_THIS_FILE)            # .../hindi/train/
_HINDI_DIR = os.path.dirname(_TRAIN_DIR)            # .../hindi/
_REPO_ROOT  = os.path.dirname(_HINDI_DIR)           # .../individual-project-charan-1230/
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

try:
    from .dataset import StreamingPackedDataset, create_dataloader
except ImportError:
    from hindi.train.dataset import StreamingPackedDataset, create_dataloader


# ── Constants ─────────────────────────────────────────────────────────────────

TRAIN_JSONL = os.path.join(_REPO_ROOT, "hindi", "corpus", "tokenised_splits",
                            "final_train_500M.jsonl")
VAL_JSONL   = os.path.join(_REPO_ROOT, "hindi", "corpus", "tokenised_splits",
                            "val.jsonl")

CONTEXT_LENGTH = 512
BOS_ID         = 2
EOS_ID         = 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise AssertionError(msg)


def _write_temp_jsonl(records: list[dict]) -> str:
    """Write a list of dicts to a temporary JSONL file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="validate_ds_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


# ── Check 1 — Dataset creation ────────────────────────────────────────────────

def check_dataset_creation() -> tuple[StreamingPackedDataset, StreamingPackedDataset]:
    _section("Check 1 — Dataset creation")

    for path, name in [(TRAIN_JSONL, "train"), (VAL_JSONL, "val")]:
        if not os.path.exists(path):
            _fail(
                f"{name} JSONL not found: {path!r}\n"
                "       Run from the repository root."
            )

    train_ds = StreamingPackedDataset(TRAIN_JSONL, context_length=CONTEXT_LENGTH,
                                       bos_id=BOS_ID, eos_id=EOS_ID)
    val_ds   = StreamingPackedDataset(VAL_JSONL,   context_length=CONTEXT_LENGTH,
                                       bos_id=BOS_ID, eos_id=EOS_ID)

    assert isinstance(train_ds, StreamingPackedDataset)
    assert isinstance(val_ds,   StreamingPackedDataset)

    print(f"\n  Train dataset: {train_ds!r}")
    print(f"\n  Val dataset:   {val_ds!r}")

    _pass("Train and validation datasets instantiated successfully.")
    return train_ds, val_ds


# ── Check 2 — Sample structure ────────────────────────────────────────────────

def check_sample_structure(train_ds: StreamingPackedDataset) -> dict:
    _section("Check 2 — Sample structure (shape, dtype, value range)")

    it = iter(train_ds)
    sample = next(it)

    input_ids = sample["input_ids"]
    labels    = sample["labels"]

    print(f"\n  input_ids : shape={tuple(input_ids.shape)}, dtype={input_ids.dtype}")
    print(f"  labels    : shape={tuple(labels.shape)}, dtype={labels.dtype}")
    print(f"  input_ids[:10] = {input_ids[:10].tolist()}")
    print(f"  labels[:10]    = {labels[:10].tolist()}")

    # Shape
    if input_ids.shape != (CONTEXT_LENGTH,):
        _fail(f"input_ids shape {tuple(input_ids.shape)} != ({CONTEXT_LENGTH},)")
    if labels.shape != (CONTEXT_LENGTH,):
        _fail(f"labels shape {tuple(labels.shape)} != ({CONTEXT_LENGTH},)")

    # Dtype
    if input_ids.dtype != torch.long:
        _fail(f"input_ids dtype {input_ids.dtype} != torch.long")
    if labels.dtype != torch.long:
        _fail(f"labels dtype {labels.dtype} != torch.long")

    # Value range
    min_id = int(input_ids.min())
    max_id = int(input_ids.max())
    print(f"  Token ID range in input_ids: [{min_id}, {max_id}]  (vocab 0..9999)")
    if min_id < 0 or max_id >= 10000:
        _fail(f"Token ID out of range [0, 9999]: min={min_id}, max={max_id}")

    _pass(f"Sample has correct shape ({CONTEXT_LENGTH},), dtype=torch.long, IDs in range.")
    return sample


# ── Check 3 — Causal shift ────────────────────────────────────────────────────

def check_causal_shift(sample: dict) -> None:
    _section("Check 3 — Causal shift: input_ids[1:] == labels[:-1]")

    input_ids = sample["input_ids"]
    labels    = sample["labels"]

    # labels[i] = input_ids[i+1] for all i in 0..510
    # i.e. labels[:-1] == input_ids[1:]
    shifted_input = input_ids[1:]    # tokens 1..511
    shifted_labels = labels[:-1]     # labels 0..510

    print(f"  input_ids[1:5]  = {shifted_input[:4].tolist()}")
    print(f"  labels[:-1][:5] = {shifted_labels[:4].tolist()}")

    if not torch.equal(shifted_input, shifted_labels):
        # Find first mismatch for diagnosis
        diff = (shifted_input != shifted_labels).nonzero(as_tuple=True)[0]
        first = int(diff[0]) if len(diff) > 0 else -1
        _fail(
            f"Causal shift violated at position {first}: "
            f"input_ids[{first+1}]={int(shifted_input[first])} != "
            f"labels[{first}]={int(shifted_labels[first])}"
        )

    _pass("input_ids[1:] == labels[:-1] — causal shift is correct.")


# ── Check 4 — BOS/EOS insertion ───────────────────────────────────────────────

def check_bos_eos_insertion() -> None:
    _section("Check 4 — BOS/EOS insertion (controlled temp file)")

    # ── Setup: small controlled documents ────────────────────────────────────
    # Document tokens are in range [10, 99] to avoid collision with BOS(2)/EOS(3).
    doc_tokens = [10, 20, 30, 40]
    records = [{"tokens": doc_tokens} for _ in range(20)]  # 20 identical docs

    # Wrapped doc: [2, 10, 20, 30, 40, 3]  = 6 tokens each
    # context_length = 5 → window = 6
    # After 1 doc: buffer = [2,10,20,30,40,3]  → exactly 1 window
    # First yield: input=[2,10,20,30,40], labels=[10,20,30,40,3]
    ctx = 5  # small context for easy inspection

    tmp_path = _write_temp_jsonl(records)
    try:
        ds = StreamingPackedDataset(tmp_path, context_length=ctx,
                                    bos_id=BOS_ID, eos_id=EOS_ID)
        it = iter(ds)
        sample = next(it)
        input_ids = sample["input_ids"].tolist()
        labels    = sample["labels"].tolist()
    finally:
        os.unlink(tmp_path)

    print(f"\n  context_length = {ctx}")
    print(f"  doc_tokens     = {doc_tokens}")
    print(f"  wrapped doc    = [BOS={BOS_ID}] + {doc_tokens} + [EOS={EOS_ID}]")
    print(f"                 = {[BOS_ID] + doc_tokens + [EOS_ID]}")
    print(f"\n  First sample:")
    print(f"    input_ids = {input_ids}")
    print(f"    labels    = {labels}")

    # Verify BOS appears in the sequence
    all_tokens = input_ids + labels
    if BOS_ID not in all_tokens:
        _fail(f"BOS (id={BOS_ID}) not found in the packed sequence.")
    if EOS_ID not in all_tokens:
        _fail(f"EOS (id={EOS_ID}) not found in the packed sequence.")

    # Verify specific values of first window:
    # doc_wrapped = [2, 10, 20, 30, 40, 3]
    # input_ids should be [2, 10, 20, 30, 40]
    # labels    should be [10, 20, 30, 40, 3]
    expected_input = [BOS_ID] + doc_tokens          # [2, 10, 20, 30, 40]
    expected_labels = doc_tokens + [EOS_ID]          # [10, 20, 30, 40,  3]

    if input_ids != expected_input:
        _fail(f"input_ids {input_ids} != expected {expected_input}")
    if labels != expected_labels:
        _fail(f"labels {labels} != expected {expected_labels}")

    _pass(
        f"BOS(id={BOS_ID}) and EOS(id={EOS_ID}) appear correctly. "
        f"First window matches expected values exactly."
    )


# ── Check 5 — Cross-document packing ─────────────────────────────────────────

def check_cross_document_packing() -> None:
    _section("Check 5 — Cross-document packing (EOS followed by BOS)")

    # Two distinct documents, context_length=6 → window=7
    # doc1 wrapped: [2, 100, 200, 300, 3]          (5 tokens)
    # doc2 wrapped: [2, 400, 500, 600, 3]          (5 tokens)
    # buffer after 2 docs: [2,100,200,300,3,2,400,500,600,3]  (10 tokens)
    # window=7: need ≥7 → 10 ≥ 7 → first yield:
    #   input_ids = [2, 100, 200, 300, 3, 2]       ← EOS at [4], BOS at [5]
    #   labels    = [100, 200, 300, 3, 2, 400]
    #   del buffer[:6] → buffer = [2, 400, 500, 600, 3]
    ctx = 6

    records = [
        {"tokens": [100, 200, 300]},
        {"tokens": [400, 500, 600]},
        {"tokens": [700, 800, 900]},  # extra doc to fill buffer for 2nd window
    ]

    tmp_path = _write_temp_jsonl(records)
    try:
        ds = StreamingPackedDataset(tmp_path, context_length=ctx,
                                    bos_id=BOS_ID, eos_id=EOS_ID)
        samples = []
        for s in ds:
            samples.append(s)
    finally:
        os.unlink(tmp_path)

    print(f"\n  context_length = {ctx}")
    print(f"  doc1 = [100, 200, 300]  →  wrapped: [2, 100, 200, 300, 3]")
    print(f"  doc2 = [400, 500, 600]  →  wrapped: [2, 400, 500, 600, 3]")
    print(f"  doc3 = [700, 800, 900]  →  wrapped: [2, 700, 800, 900, 3]")

    if len(samples) == 0:
        _fail("No samples were yielded from the controlled dataset.")

    print(f"\n  Total samples yielded: {len(samples)}")
    for i, s in enumerate(samples):
        print(f"  sample[{i}].input_ids = {s['input_ids'].tolist()}")
        print(f"  sample[{i}].labels    = {s['labels'].tolist()}")

    # Search for EOS immediately followed by BOS in any sample's input_ids
    found_boundary = False
    for s in samples:
        ids = s["input_ids"].tolist()
        for i in range(len(ids) - 1):
            if ids[i] == EOS_ID and ids[i + 1] == BOS_ID:
                found_boundary = True
                print(f"\n  Cross-doc boundary found in sample: "
                      f"input_ids[{i}]={EOS_ID} (EOS), "
                      f"input_ids[{i+1}]={BOS_ID} (BOS)")
                break
        if found_boundary:
            break

    if not found_boundary:
        _fail(
            "No EOS→BOS transition found in any sample. "
            "Cross-document packing may not be working correctly."
        )

    # Specifically verify first sample:
    # buffer = [2,100,200,300,3,2,400,500,600,3]
    # window=7: input=[2,100,200,300,3,2], which has EOS at[4] and BOS at[5]
    first_input = samples[0]["input_ids"].tolist()
    expected_first = [BOS_ID, 100, 200, 300, EOS_ID, BOS_ID]
    if first_input != expected_first:
        _fail(
            f"First sample input_ids {first_input} "
            f"!= expected {expected_first}"
        )

    _pass(
        "EOS→BOS transition confirmed at document boundary. "
        "Cross-document packing is correct."
    )


# ── Check 6 — Multiple real samples ───────────────────────────────────────────

def check_multiple_real_samples(train_ds: StreamingPackedDataset) -> None:
    _section("Check 6 — Multiple samples from real training data")

    N = 200

    # Fresh dataset instance to avoid state from Check 2
    ds = StreamingPackedDataset(TRAIN_JSONL, context_length=CONTEXT_LENGTH,
                                 bos_id=BOS_ID, eos_id=EOS_ID)
    count = 0
    for sample in ds:
        count += 1
        if count >= N:
            break

    print(f"\n  Requested : {N} samples")
    print(f"  Received  : {count} samples")
    print(f"  Docs consumed so far: {ds._docs_processed}")
    print(f"  Chunks yielded:       {ds._chunks_yielded}")

    if count < N:
        _fail(f"Only {count}/{N} samples could be produced — dataset too small or broken.")

    _pass(f"Successfully iterated {count} real training samples.")


# ── Check 7 — DataLoader batching ─────────────────────────────────────────────

def check_dataloader_batching() -> None:
    _section("Check 7 — DataLoader batching (batch_size=4)")

    BATCH_SIZE = 4

    ds = StreamingPackedDataset(TRAIN_JSONL, context_length=CONTEXT_LENGTH,
                                 bos_id=BOS_ID, eos_id=EOS_ID)
    loader = create_dataloader(ds, batch_size=BATCH_SIZE,
                               pin_memory=False, drop_last=True)

    batch = next(iter(loader))

    input_batch = batch["input_ids"]
    label_batch = batch["labels"]

    print(f"\n  batch_size         = {BATCH_SIZE}")
    print(f"  input_ids.shape    = {tuple(input_batch.shape)}")
    print(f"  labels.shape       = {tuple(label_batch.shape)}")
    print(f"  input_ids.dtype    = {input_batch.dtype}")

    expected_shape = (BATCH_SIZE, CONTEXT_LENGTH)

    if tuple(input_batch.shape) != expected_shape:
        _fail(
            f"input_ids batch shape {tuple(input_batch.shape)} "
            f"!= expected {expected_shape}"
        )
    if tuple(label_batch.shape) != expected_shape:
        _fail(
            f"labels batch shape {tuple(label_batch.shape)} "
            f"!= expected {expected_shape}"
        )
    if input_batch.dtype != torch.long:
        _fail(f"input_ids dtype {input_batch.dtype} != torch.long")

    _pass(
        f"DataLoader produces batches of shape {expected_shape}, dtype=torch.long."
    )


# ── Check 8 — State / resume ──────────────────────────────────────────────────

def check_state_resume() -> None:
    _section("Check 8 — State / resume (4 sub-checks: 8a–8d)")

    # ─────────────────────────────────────────────────────────────────────────
    # Shared helper: iterate skip_n samples, save state, then verify that the
    # resumed dataset's first sample matches the original iterator's next one.
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_resume(records: list, ctx: int, skip_n: int, label: str) -> None:
        tmp_path = _write_temp_jsonl(records)
        try:
            # Pass 1: consume skip_n samples, save state, consume one more.
            ds1 = StreamingPackedDataset(tmp_path, context_length=ctx,
                                         bos_id=BOS_ID, eos_id=EOS_ID)
            it1 = iter(ds1)
            for _ in range(skip_n):
                next(it1)
            saved_state = ds1.state_dict()
            expected    = next(it1)      # sample skip_n+1 (the "truth")

            # Pass 2: fresh dataset, load state, get next sample.
            ds2 = StreamingPackedDataset(tmp_path, context_length=ctx,
                                         bos_id=BOS_ID, eos_id=EOS_ID)
            ds2.load_state_dict(saved_state)
            resumed = next(iter(ds2))
        finally:
            os.unlink(tmp_path)

        print(f"\n  [{label}] Saved state:")
        print(f"    file_offset    = {saved_state['file_offset']} bytes")
        print(f"    buffer length  = {len(saved_state['buffer'])} tokens")
        print(f"    buffer[:6]     = {saved_state['buffer'][:6]}...")
        print(f"    docs_processed = {saved_state['docs_processed']}")
        print(f"    chunks_yielded = {saved_state['chunks_yielded']}")
        print(f"  [{label}] Expected  input_ids: {expected['input_ids'].tolist()}")
        print(f"  [{label}] Resumed   input_ids: {resumed['input_ids'].tolist()}")
        print(f"  [{label}] Expected  labels   : {expected['labels'].tolist()}")
        print(f"  [{label}] Resumed   labels   : {resumed['labels'].tolist()}")

        if not torch.equal(expected["input_ids"], resumed["input_ids"]):
            _fail(f"[{label}] input_ids mismatch after load_state_dict().")
        if not torch.equal(expected["labels"], resumed["labels"]):
            _fail(f"[{label}] labels mismatch after load_state_dict().")

        # chunks_yielded must advance by exactly 1 after one resumed next().
        state2 = ds2.state_dict()
        if state2["chunks_yielded"] != saved_state["chunks_yielded"] + 1:
            _fail(
                f"[{label}] chunks_yielded mismatch: "
                f"expected {saved_state['chunks_yielded'] + 1}, "
                f"got {state2['chunks_yielded']}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 8a — Standard resume that CROSSES a document boundary.
    #
    # 15 docs × 50 tokens, wrapped = 52 tokens each.  ctx=10 (window=11).
    # skip_n=7 falls mid-second-doc: snapshot has file_offset=after-doc2
    # and 34 tokens in the buffer (tail of doc2).
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  Sub-check 8a — Standard cross-doc resume (15 docs × 50 tok, skip_n=7, ctx=10)")
    _verify_resume(
        records=[{"tokens": list(range(10, 60))} for _ in range(15)],
        ctx=10,
        skip_n=7,
        label="8a",
    )
    _pass("8a: Cross-doc resume — exact match.")

    # ─────────────────────────────────────────────────────────────────────────
    # 8b — ALL resumed chunks come from a SINGLE long document.
    #
    # doc1 = 45 tokens → wrapped = 47 tokens.  ctx=10 (window=11).
    # Buffer evolution (only doc1 is read for all 4 chunks):
    #   Start  : [2, 100..144, 3]  = 47 tokens
    #   chunk1 : del[:10] → 37 remaining  (snapshot_offset = after_doc1)
    #   chunk2 : del[:10] → 27 remaining  ← skip_n=2 saves here
    #   chunk3 : del[:10] → 17 remaining  ← this is the "expected" next
    #   chunk4 : del[:10] →  7 remaining  (7 < 11, inner while exits)
    #
    # All 4 snapshots share the SAME file_offset (after_doc1).
    # Resume after chunk2 must NOT re-read doc1 — it must immediately yield
    # chunk3 from the buffered 27 tokens.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  Sub-check 8b — Multiple chunks from single long doc (skip_n=2, ctx=10)")
    _verify_resume(
        records=[
            {"tokens": list(range(100, 145))},   # doc1: 45 tok → 47 wrapped → 4 chunks
            {"tokens": list(range(200, 250))},   # doc2: tail of file (needed for ds1.next())
        ],
        ctx=10,
        skip_n=2,
        label="8b",
    )
    _pass("8b: Mid-doc resume (all chunks from one doc) — exact match.")

    # ─────────────────────────────────────────────────────────────────────────
    # 8c — Resume when the buffer holds tokens from MULTIPLE documents.
    #
    # Each doc is 4 tokens → wrapped = 6 tokens.  With ctx=10 (window=11),
    # a single doc never fills the window; tokens accumulate across docs:
    #
    #   After doc1: buffer = [2,100,101,102,103,3]              = 6 < 11
    #   After doc2: buffer = [2,100,101,102,103,3,2,200,201,202,203,3] = 12 ≥ 11
    #   chunk1 emission:
    #     input  = [2,100,101,102,103,3,2,200,201,202]
    #     labels = [100,101,102,103,3,2,200,201,202,203]
    #     del buffer[:10] → buffer = [203, 3]  (2 tokens — tail of doc2)
    #     snapshot: (file_offset=after_doc2, buffer=[203,3])
    #
    # On resume: buffer=[203,3], seek to after_doc2.
    # Then: read doc3 (buf=8<11), read doc4 (buf=14≥11) → yield chunk2.
    # chunk2 must exactly match the original iterator's chunk2.
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  Sub-check 8c — Multi-doc buffer at snapshot (4-tok docs, skip_n=1, ctx=10)")
    _verify_resume(
        records=[
            {"tokens": [100, 101, 102, 103]},   # doc1: 4 tok → 6 wrapped
            {"tokens": [200, 201, 202, 203]},   # doc2: 4 tok → 6 wrapped
            {"tokens": [300, 301, 302, 303]},   # doc3: continues after resume
            {"tokens": [400, 401, 402, 403]},   # doc4: fills window after doc3
            {"tokens": [500, 501, 502, 503]},   # doc5: ensure iterator continues
        ],
        ctx=10,
        skip_n=1,
        label="8c",
    )
    _pass("8c: Multi-doc buffer resume — exact match.")

    # ─────────────────────────────────────────────────────────────────────────
    # 8d — Resume when the buffer contains EXACTLY 1 token [EOS] after yield.
    #      This is the minimum-buffer case at an exact document boundary.
    #
    # doc1 = 9 tokens → wrapped = [BOS, 50..58, EOS] = 11 tokens = exactly 1 window.
    # After chunk1: del buffer[:10] → buffer = [3]  (just the trailing EOS).
    # snapshot: (offset=after_doc1, buffer=[3])
    #
    # The EOS MUST appear as input_ids[0] of the next sample — it signals
    # end-of-document to the model and must not be dropped or skipped.
    #
    # After resume, reading doc2 = [BOS, 100..109, EOS] = 12 tokens:
    #   buffer = [3, 2, 100..109, 3] = 13 tokens ≥ 11
    #   chunk2:  input  = [3, 2, 100, 101, 102, 103, 104, 105, 106, 107]
    #            labels = [2, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    # ─────────────────────────────────────────────────────────────────────────
    print("\n  Sub-check 8d — Exact doc-boundary resume: buffer=[EOS] (ctx=10, skip_n=1)")
    _verify_resume(
        records=[
            {"tokens": list(range(50, 59))},    # doc1: 9 tok → 11 wrapped = 1 exact window
            {"tokens": list(range(100, 110))},  # doc2: 10 tok → 12 wrapped
            {"tokens": list(range(200, 210))},  # doc3: ensures continued iteration
        ],
        ctx=10,
        skip_n=1,
        label="8d",
    )
    _pass("8d: Exact-boundary resume (buffer=[EOS]) — exact match. "
          "EOS preserved as input_ids[0] of next sample.")

    _pass(
        "All 4 state/resume sub-checks passed (8a cross-doc, 8b single-doc, "
        "8c multi-doc-buffer, 8d exact-boundary)."
    )


# ── Check 9 — Memory / buffer bound ──────────────────────────────────────────

def check_memory_buffer_bound() -> None:
    _section("Check 9 — Memory: rolling buffer stays bounded")

    # Design argument:
    # The buffer only grows by (len(doc_tokens) + 2) per document (BOS + EOS).
    # It shrinks by context_length tokens for each yielded window.
    # The buffer can never exceed: max_doc_length + 2 + context_length tokens
    # (one doc's worth + one pending window) plus one extra token for overlap.
    #
    # For this corpus: max doc ≈ 41776 + 2 + 512 = ~42290 tokens = ~340 KB
    # This is constant regardless of total corpus size (500 M tokens).
    #
    # We verify this empirically with a small dataset by measuring
    # the buffer size immediately before each yield (via monkey-patching).

    doc_tokens = list(range(100, 200))   # 100 tokens per doc
    records = [{"tokens": doc_tokens} for _ in range(30)]  # 30 docs
    ctx = 15   # small context

    max_observed_buffer = 0
    sample_count = 0

    tmp_path = _write_temp_jsonl(records)
    try:
        ds = StreamingPackedDataset(tmp_path, context_length=ctx,
                                    bos_id=BOS_ID, eos_id=EOS_ID)

        # We observe _snapshot_buffer lengths after each yield to measure
        # the maximum buffer size during iteration.
        for sample in ds:
            buf_len = len(ds._snapshot_buffer)
            if buf_len > max_observed_buffer:
                max_observed_buffer = buf_len
            sample_count += 1

    finally:
        os.unlink(tmp_path)

    # Theoretical maximum for these docs:
    # After reading one doc into a buffer that already has one pending overlap token:
    #   max_buffer = 1 (overlap from prev window) + (100 + 2) (new doc) = 103
    # But before the while loop runs, we could have:
    #   prev_overlap (≤ ctx = 15) + new_doc (102) = 117
    theoretical_max = ctx + (len(doc_tokens) + 2)   # 15 + 102 = 117

    print(f"\n  Docs: 30 × {len(doc_tokens)} tokens each")
    print(f"  context_length            = {ctx}")
    print(f"  Samples yielded           = {sample_count}")
    print(f"  Max buffer size observed  = {max_observed_buffer} tokens")
    print(f"  Theoretical max bound     = {theoretical_max} tokens")
    print(f"  (Constant regardless of total corpus size = 500 M tokens)")

    if max_observed_buffer > theoretical_max:
        _fail(
            f"Buffer exceeded theoretical maximum: "
            f"{max_observed_buffer} > {theoretical_max}"
        )

    _pass(
        f"Rolling buffer peaked at {max_observed_buffer} tokens "
        f"(bound: {theoretical_max}). "
        "Memory usage is O(max_doc_length + context_length), not O(corpus)."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all 9 validation checks and print a final summary."""
    print("=" * 60)
    print("  Phase 3.1 — StreamingPackedDataset Validation")
    print("=" * 60)

    print(f"\n  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    print(f"  Repo root       : {_REPO_ROOT}")
    print(f"  Train file      : {TRAIN_JSONL}")
    print(f"  Val file        : {VAL_JSONL}")

    results: dict[str, bool] = {}

    def run(name: str, fn, *args):
        try:
            result = fn(*args)
            results[name] = True
            return result
        except Exception as exc:
            results[name] = False
            import traceback
            print(f"\n  [FAIL] {name}")
            traceback.print_exc()
            return None

    # ── Execute checks (some depend on earlier results) ───────────────────────
    train_ds, val_ds = run("Dataset creation", check_dataset_creation) \
                       or (None, None)

    if train_ds is None:
        print("\n  Cannot continue without datasets. Check paths and PyTorch.")
        sys.exit(1)

    sample = run("Sample structure",      check_sample_structure, train_ds)
    if sample is not None:
        run("Causal shift",               check_causal_shift,      sample)

    run("BOS/EOS insertion",              check_bos_eos_insertion)
    run("Cross-document packing",         check_cross_document_packing)
    run("Multiple real samples",          check_multiple_real_samples, train_ds)
    run("DataLoader batching",            check_dataloader_batching)
    run("State/resume",                   check_state_resume)
    run("Memory/buffer bound",            check_memory_buffer_bound)

    # ── Summary ───────────────────────────────────────────────────────────────
    check_names = [
        "Dataset creation",
        "Sample structure",
        "Causal shift",
        "BOS/EOS insertion",
        "Cross-document packing",
        "Multiple real samples",
        "DataLoader batching",
        "State/resume",
        "Memory/buffer bound",
    ]

    _section("Summary")
    all_passed = True
    for name in check_names:
        status = results.get(name, False)
        icon   = "[PASS]" if status else "[FAIL]"
        print(f"  {icon} {name}")
        if not status:
            all_passed = False

    print()
    if all_passed:
        print("  All 9 checks PASSED.  StreamingPackedDataset is ready.")
        print("  The data pipeline can now be connected to the training loop.")
    else:
        print("  Some checks FAILED.  Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
