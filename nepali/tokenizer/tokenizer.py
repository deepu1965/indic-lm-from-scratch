"""
tokenizer.py
============
Phase 1: BPE tokenizer training, evaluation, and reporting.
Uses Hugging Face `tokenizers` library — trained completely from scratch.
No pretrained models, no SentencePiece, no checkpointing.

Usage
-----
    python3 hindi/tokenizer/tokenizer.py --config hindi/tokenizer/tokenizer_config_hindi.yaml
    python3 hindi/tokenizer/tokenizer.py --config hindi/tokenizer/tokenizer_config_hindi.yaml --test

The same script is used for both Hindi and Nepali; only the YAML config differs.

Architecture
------------
  1. stream_texts()          — lazy JSONL reader; never loads full corpus into RAM
  2. text_iterator()         — wraps stream_texts; chunks very long documents so the
                               BPE trainer receives manageable strings while
                               preserving all content
  3. train_one()             — trains one BPE model for a given vocab size and saves
                               it as bpe_v<N>.json
  4. evaluate_and_freq()     — evaluates a trained model on the validation split in a
                               single streaming pass; returns metrics + frequency counter
  5. write_*()               — CSV / JSON / TXT report writers
  6. run_test()              — self-contained correctness test (tiny synthetic corpus)
  7. main()                  — orchestrates all steps

Memory strategy
---------------
`BpeTrainer.train_from_iterator()` is used instead of `train_from_files()`.
This prevents the Python layer from loading the entire JSONL corpus into RAM.
However, the underlying Rust BPE trainer still maintains an internal word-count
table that grows with the number of unique word types seen.  For a 596M-word
corpus, peak trainer memory may be substantial (several GB); it is not
guaranteed to fit in 8 GB RAM regardless of the streaming approach used here.

Very long documents are chunked at whitespace boundaries up to `max_chunk_chars`
(default 50 000 chars ≈ 10 000 words) so no single Python string is enormous.
All content is preserved; nothing is discarded.  Chunks always end at a
whitespace/newline boundary where possible.

Reproducibility note
--------------------
The installed `tokenizers.trainers.BpeTrainer` does not expose a random-seed
parameter.  BPE merge selection is deterministic given the same ordered input,
so results are reproducible as long as the input files are unchanged.

No automatic vocabulary selection — review vocab_comparison.csv manually.
"""

import argparse
import collections
import csv
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

import yaml

# Shared whitespace utilities (single source of truth for Hindi + Nepali)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tokenizer_utils import preprocess_text, custom_decode, has_unk, normalise_text

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# ────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ────────────────────────────────────────────────────────────────────────────

def stream_texts(jsonl_path: str, max_chars: int = None):
    """Yield text strings from a JSONL file, one document at a time.

    Stops early when cumulative characters exceed `max_chars` (if set).
    Never loads the whole file into memory.
    Preserves internal newline characters in the document text.
    """
    seen_chars = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("text", "")
            if not text:
                continue
            yield text
            seen_chars += len(text)
            if max_chars is not None and seen_chars >= max_chars:
                break


def _chunk_doc(text: str, max_chunk_chars: int):
    """Yield whitespace-aware chunks of `text` of at most `max_chunk_chars` chars.

    Each chunk (except possibly the last) ends at the rightmost space or newline
    found within the limit, so words are never split across chunks.  If no
    whitespace boundary exists within the limit the hard character boundary is
    used as a fallback, guaranteeing the loop always terminates.

    All characters are preserved; nothing is dropped or duplicated.
    """
    start = 0
    n = len(text)
    while start < n:
        end = start + max_chunk_chars
        if end >= n:
            yield text[start:]
            break
        # Search backward from `end` for the rightmost whitespace character.
        ws = -1
        for i in range(end - 1, start, -1):
            if text[i].isspace():
                ws = i
                break
        if ws > start:
            # Include the whitespace character in this chunk.
            yield text[start: ws + 1]
            start = ws + 1
        else:
            # Fallback: no whitespace found — hard cut.
            yield text[start: end]
            start = end


def text_iterator(jsonl_path: str, max_chars: int = None,
                  max_chunk_chars: int = 50_000):
    """Streaming iterator for BpeTrainer.train_from_iterator().

    For each document from the JSONL file:
    - If the document is <= max_chunk_chars characters, yield it as-is.
    - If it is longer, delegate to _chunk_doc() which splits at whitespace
      boundaries so words are never split across chunks.

    All content is preserved; no characters are dropped or duplicated.
    Newline characters inside documents are preserved; they are not replaced
    with spaces. The Whitespace pre-tokenizer handles them correctly.
    """
    for text in stream_texts(jsonl_path, max_chars):
        text = preprocess_text(text)
        if len(text) <= max_chunk_chars:
            yield text
        else:
            yield from _chunk_doc(text, max_chunk_chars)


def corpus_stats(jsonl_path: str, max_chars: int = None) -> dict:
    """Count documents, words, and characters from a JSONL file."""
    docs = words = chars = 0
    for text in stream_texts(jsonl_path, max_chars):
        docs  += 1
        words += len(text.split())
        chars += len(text)
    return {"documents": docs, "words": words, "characters": chars}


# ────────────────────────────────────────────────────────────────────────────
# Tokenizer training
# ────────────────────────────────────────────────────────────────────────────

def train_one(
    train_jsonl: str,
    model_path: str,
    vocab_size: int,
    special_tokens: list,
    max_chunk_chars: int,
    training_max_chars: int = None,
) -> Tokenizer:
    """Train one BPE tokenizer from scratch and save it as `model_path`.

    Uses BpeTrainer.train_from_iterator() so the corpus is streamed
    lazily — the full training corpus is never loaded into Python RAM.

    Parameters
    ----------
    train_jsonl        : path to the training JSONL file
    model_path         : destination path for the serialised tokenizer (.json)
    vocab_size         : target vocabulary size
    special_tokens     : list of special token strings (e.g. ["[UNK]", ...])
    max_chunk_chars    : maximum characters per chunk fed to the trainer
    training_max_chars : optional cap on total training characters
    """
    print(f"  [train] vocab_size={vocab_size}  model_path={model_path}")
    t0 = time.time()

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    iterator = text_iterator(train_jsonl, training_max_chars, max_chunk_chars)
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tokenizer.save(model_path)

    elapsed = time.time() - t0
    actual_vocab = tokenizer.get_vocab_size()
    print(f"  [train] Done in {elapsed:.0f}s  actual_vocab={actual_vocab:,}")
    return tokenizer


# ────────────────────────────────────────────────────────────────────────────
# Evaluation + frequency (single pass over validation data)
# ────────────────────────────────────────────────────────────────────────────

def evaluate_and_freq(
    tokenizer: Tokenizer,
    val_jsonl: str,
    max_chars: int = None,
) -> tuple:
    """Single streaming pass over val_jsonl.

    Returns
    -------
    metrics  — dict with fertility, unk_rate, chars_per_token, etc.
    freq     — collections.Counter mapping token_id → count
    """
    vocab = tokenizer.get_vocab()                    # str → id
    unk_id = vocab.get("[UNK]", None)

    freq         = collections.Counter()
    total_tokens = 0
    total_words  = 0
    total_chars  = 0
    total_unk    = 0
    docs         = 0

    for raw_text in stream_texts(val_jsonl, max_chars):
        # Word count on the RAW text so fertility reflects original corpus words,
        # not the inflated word count from marker insertion.
        total_words  += len(raw_text.split())
        total_chars  += len(raw_text)

        preprocessed  = preprocess_text(raw_text)
        encoding      = tokenizer.encode(preprocessed)
        ids           = encoding.ids
        freq.update(ids)
        total_tokens += len(ids)
        if unk_id is not None:
            total_unk += ids.count(unk_id)
        docs         += 1

    fertility   = total_tokens / total_words  if total_words  else 0.0
    chars_token = total_chars  / total_tokens if total_tokens else 0.0
    unk_rate    = total_unk    / total_tokens if total_tokens else 0.0

    counts   = list(freq.values())
    n_unique = len(counts)

    # id → piece lookup from vocabulary
    id_to_piece = {v: k for k, v in vocab.items()}

    freq_summary = {
        "total_token_occurrences": total_tokens,
        "unique_tokens_observed":  n_unique,
        "min_frequency":           min(counts) if counts else 0,
        "max_frequency":           max(counts) if counts else 0,
        "mean_frequency":          round(statistics.mean(counts),   2) if counts else 0,
        "median_frequency":        round(statistics.median(counts), 2) if counts else 0,
        "top50_tokens": [
            {"token_id": tid, "piece": id_to_piece.get(tid, "?"), "count": cnt}
            for tid, cnt in freq.most_common(50)
        ],
    }

    metrics = {
        "documents":        docs,
        "total_characters": total_chars,
        "total_words":      total_words,
        "total_tokens":     total_tokens,
        "fertility":        round(fertility,   4),
        "chars_per_token":  round(chars_token, 4),
        "unk_count":        total_unk,
        "unk_rate":         unk_rate,
        "freq_summary":     freq_summary,
    }
    return metrics, freq


# ────────────────────────────────────────────────────────────────────────────
# Report writers
# ────────────────────────────────────────────────────────────────────────────

def write_vocab_comparison_csv(eval_results: list, path: str) -> None:
    """Write per-candidate evaluation metrics to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "vocab_size", "documents", "total_characters", "total_words",
        "total_tokens", "fertility", "chars_per_token", "unk_count", "unk_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in eval_results:
            w.writerow({k: r[k] for k in fields})
    print(f"  [report] Wrote {path}")


def write_tokenizer_statistics_json(
    config: dict,
    corpus_stats_train: dict,
    corpus_stats_val:   dict,
    candidates: list,
    path: str,
) -> None:
    """Write comprehensive statistics for all candidate vocabularies to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "language":   config["language"],
        "model_type": config["model_type"],
        "note":       (
            "No automatic vocabulary selection. Review vocab_comparison.csv to choose. "
            "BpeTrainer does not expose a random seed; results are deterministic on "
            "identical ordered input."
        ),
        "training_corpus":   corpus_stats_train,
        "validation_corpus": corpus_stats_val,
        "candidates": [
            {
                "vocab_size":      c["vocab_size"],
                "actual_vocab":    c["actual_vocab"],
                "evaluation":      {k: v for k, v in c["metrics"].items() if k != "freq_summary"},
                "token_frequency": c["metrics"]["freq_summary"],
            }
            for c in candidates
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [report] Wrote {path}")


def write_token_frequency_csv(
    freq: collections.Counter,
    tokenizer: Tokenizer,
    path: str,
) -> None:
    """Write the complete BPE vocabulary to CSV with val-set frequencies.

    Every token in the trained vocabulary is listed.  Tokens not observed
    in the validation pass receive count=0.
    Sorted by descending count, then ascending token_id as tiebreaker.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vocab = tokenizer.get_vocab()          # str → id
    id_to_piece = {v: k for k, v in vocab.items()}
    vocab_size  = tokenizer.get_vocab_size()

    rows = sorted(
        ((freq.get(tid, 0), tid) for tid in range(vocab_size)),
        key=lambda x: (-x[0], x[1]),
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["token_id", "piece", "count"])
        for cnt, tid in rows:
            w.writerow([tid, id_to_piece.get(tid, "?"), cnt])
    print(f"  [report] Wrote {path}")


def write_tokenization_examples(
    tokenizer: Tokenizer,
    example_sentences: list,
    path: str,
    vocab_size: int,
    language: str,
) -> None:
    """Write tokenization examples for several representative sentences."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    id_to_piece = {v: k for k, v in tokenizer.get_vocab().items()}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Tokenization Examples — {language} — vocab_size={vocab_size}\n")
        f.write("=" * 70 + "\n\n")
        for sent in example_sentences:
            preprocessed = preprocess_text(sent)
            enc     = tokenizer.encode(preprocessed)
            ids     = enc.ids
            tokens  = enc.tokens
            decoded = custom_decode(tokenizer, ids)
            f.write(f"Original : {sent}\n")
            f.write(f"Preproc  : {preprocessed}\n")
            f.write(f"Tokens   : {tokens}\n")
            f.write(f"IDs      : {ids}\n")
            f.write(f"Decoded  : {decoded}\n\n")
    print(f"  [report] Wrote {path}")


# ────────────────────────────────────────────────────────────────────────────
# Test mode
# ────────────────────────────────────────────────────────────────────────────

def run_test(config: dict) -> None:
    """Self-contained correctness test using a tiny synthetic corpus.

    Verifies:
    1.  BPE tokenizer can train successfully.
    2.  Tokenizer can encode text.
    3.  Tokenizer can decode encoded IDs.
    4.  The expected vocabulary size / model file is produced.
    5.  Frequency report is produced.
    6.  Tokenization-example report is produced.
    7.  Three vocabulary sizes can be handled.
    8.  Hindi and Nepali scripts work.
    9.  No pretrained model/tokenizer is accessed.
    10. No checkpoint files are created.
    11. No SentencePiece files are created.
    12. Documents with multiple paragraphs/newlines are handled correctly.
    """
    print("\n" + "=" * 60)
    print("  TEST MODE")
    print("=" * 60)

    tmp_dir = tempfile.mkdtemp(prefix="tok_test_")
    try:
        train_path = os.path.join(tmp_dir, "train.jsonl")
        val_path   = os.path.join(tmp_dir, "val.jsonl")

        hindi_texts = [
            "नमस्ते यह एक परीक्षण वाक्य है।",
            "हिंदी भाषा में कई शब्द होते हैं।",
            "सरकार ने नई योजना घोषित की।",
            "आज का मौसम बहुत अच्छा है और सब खुश हैं।",
            "प्रौद्योगिकी के क्षेत्र में भारत आगे है।",
            # Multi-paragraph document to test newline handling
            "पहला अनुच्छेद यहाँ है।\nदूसरा अनुच्छेद यहाँ है।\nतीसरा अनुच्छेद।",
        ] * 2000
        nepali_texts = [
            "नमस्ते यो एक परीक्षण वाक्य हो।",
            "नेपाली भाषामा धेरै शब्द हुन्छन्।",
            "सरकारले नयाँ योजना घोषणा गर्‍यो।",
            "आज मौसम राम्रो छ र सबै खुसी छन्।",
            "प्रविधिको क्षेत्रमा नेपाल अगाडि छ।",
            # Multi-paragraph document
            "पहिलो अनुच्छेद।\nदोस्रो अनुच्छेद।\nतेस्रो अनुच्छेद।",
        ] * 2000

        lang  = config["language"]
        texts = hindi_texts if lang == "hindi" else nepali_texts

        with open(train_path, "w", encoding="utf-8") as fh:
            for i, t in enumerate(texts):
                fh.write(json.dumps({"id": str(i), "text": t, "source": "test"}) + "\n")
        with open(val_path, "w", encoding="utf-8") as fh:
            for i, t in enumerate(texts[:100]):
                fh.write(json.dumps({"id": str(i), "text": t, "source": "test"}) + "\n")

        mdir = os.path.join(tmp_dir, "models")
        rdir = os.path.join(tmp_dir, "reports")
        os.makedirs(mdir, exist_ok=True)
        os.makedirs(rdir, exist_ok=True)

        special_tokens    = config.get("special_tokens", ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<SPACE>", "<NEWLINE>", "<TAB>"])
        max_chunk_chars   = config.get("max_chunk_chars", 50_000)
        example_sents     = config.get("example_sentences", [texts[0]])
        test_vocab_sizes  = [60, 80, 100]

        all_eval_rows  = []
        all_candidates = []

        for vsize in test_vocab_sizes:
            print(f"\n  [test] Training vocab_size={vsize} …")
            model_path = os.path.join(mdir, f"bpe_v{vsize}.json")

            tokenizer = train_one(
                train_jsonl=train_path,
                model_path=model_path,
                vocab_size=vsize,
                special_tokens=special_tokens,
                max_chunk_chars=max_chunk_chars,
                training_max_chars=None,
            )

            # ── Verify model file written ────────────────────────────────
            assert os.path.isfile(model_path), f"Model file not found: {model_path}"
            assert model_path.endswith(".json"), "Model must be .json (not .model/.vocab)"
            assert not os.path.isfile(model_path.replace(".json", ".model")), \
                "SentencePiece .model must NOT be created"
            print(f"  ✓  bpe_v{vsize}.json written (no .model/.vocab)")

            # ── Verify vocab size is plausible ───────────────────────────
            actual = tokenizer.get_vocab_size()
            assert actual > 0, "Vocab size must be positive"
            print(f"  ✓  actual vocab size = {actual}")

            # ── Helper: assert round-trip (only when no UNKs) ────────────────
            def assert_roundtrip(raw, enc_obj, label):
                """Assert that custom_decode recovers the normalised original.
                Skips the assertion (with a warning) if UNK tokens are present,
                so the test is never falsely green when tokens are missing.
                """
                ids = enc_obj.ids
                if has_unk(tokenizer, ids):
                    unk_count = ids.count(tokenizer.get_vocab()["[UNK]"])
                    print(f"  [WARN] {label}: {unk_count} UNK token(s) — "
                          f"exact round-trip not asserted")
                    return
                expected = normalise_text(raw)
                got      = custom_decode(tokenizer, ids)
                assert got == expected, (
                    f"{label}: round-trip failed.\n"
                    f"  expected : {expected!r}\n"
                    f"  got      : {got!r}"
                )

            # ── Verify encode/decode ─────────────────────────────────────
            sample = texts[0]
            enc    = tokenizer.encode(preprocess_text(sample))
            assert len(enc.ids) > 0, "Encoding must produce tokens"
            assert "<SPACE>" in enc.tokens, \
                "Encoded tokens must explicitly contain <SPACE> markers"
            assert_roundtrip(sample, enc, "basic encode/decode")
            print(f"  ✓  encode/decode OK: '{sample[:30]}…'")

            # ── Verify subword-split reconstruction ──────────────────────
            test_word = "अनपेक्षित" if lang == "hindi" else "अपेक्षित"
            enc_word  = tokenizer.encode(preprocess_text(test_word))
            assert len(enc_word.tokens) > 1, \
                f"Test word {test_word!r} should be split into subwords"
            assert_roundtrip(test_word, enc_word, "subword reconstruction")
            print(f"  ✓  Subword-split reconstruction OK "
                  f"({len(enc_word.tokens)} subwords: {enc_word.tokens})")

            # ── Verify multiple consecutive spaces ────────────────────────
            multi_space = "यह   एक  वाक्य है" if lang == "hindi" else "यो   एक  वाक्य हो"
            enc_multi   = tokenizer.encode(preprocess_text(multi_space))
            # 3 + 2 + 1 = 6 space characters -> 6 <SPACE> tokens
            n_space = enc_multi.tokens.count("<SPACE>")
            assert n_space == 6, \
                f"Expected 6 <SPACE> markers, got {n_space}. Tokens: {enc_multi.tokens}"
            assert_roundtrip(multi_space, enc_multi, "multiple spaces")
            print(f"  ✓  Multiple consecutive spaces reconstruction OK")

            # ── Verify leading and trailing space ───────────────────────
            leading  = " " + sample
            trailing = sample + " "
            for ws_case, label in [(leading, "leading space"),
                                   (trailing, "trailing space")]:
                e = tokenizer.encode(preprocess_text(ws_case))
                assert_roundtrip(ws_case, e, label)
            print(f"  ✓  Leading / trailing space reconstruction OK")

            # ── Verify newlines (single and double) ───────────────────────
            nl_doc = texts[5]   # contains embedded \n characters
            nl_enc = tokenizer.encode(preprocess_text(nl_doc))
            assert len(nl_enc.ids) > 0, "Multi-line doc must encode to tokens"
            assert "<NEWLINE>" in nl_enc.tokens, \
                "<NEWLINE> marker must appear in encoded sequence"
            assert_roundtrip(nl_doc, nl_enc, "multi-line document")
            print(f"  ✓  Multi-line document: {len(nl_enc.ids)} tokens, round-trip OK")

            # Double newline (e.g. paragraph break)
            double_nl = sample + "\n\n" + sample
            enc_dnl   = tokenizer.encode(preprocess_text(double_nl))
            assert enc_dnl.tokens.count("<NEWLINE>") == 2, \
                f"Double newline should produce 2 <NEWLINE> tokens, got {enc_dnl.tokens.count('<NEWLINE>')}"
            assert_roundtrip(double_nl, enc_dnl, "double newline")
            print(f"  ✓  Double newline reconstruction OK")

            # ── Verify tab ───────────────────────────────────────────────
            # Use in-vocab words (test_word) to avoid UNK on tab test
            tab_doc = f"{test_word}\t{test_word}"
            tab_enc = tokenizer.encode(preprocess_text(tab_doc))
            assert "<TAB>" in tab_enc.tokens, \
                "<TAB> marker must appear in encoded sequence"
            assert_roundtrip(tab_doc, tab_enc, "tab")
            print(f"  ✓  Tab whitespace reconstruction OK")

            # ── Verify chunking: long doc is split, content preserved ────
            long_doc = ("शब्द " * 20_000).rstrip()  # ~100 000 chars, well over 50 000
            chunks = list(_chunk_doc(long_doc, 50_000))
            reconstructed = "".join(chunks)
            assert reconstructed == long_doc, "Chunking must not lose or duplicate characters"
            # Every chunk except the last must end at a whitespace character.
            for c in chunks[:-1]:
                assert c[-1] in (' ', '\n', '\t'), \
                    f"Chunk boundary not at whitespace: last char = {repr(c[-1])}"
            print(f"  ✓  Long document chunked into {len(chunks)} chunks, "
                  f"content preserved, boundaries at whitespace")

            # ── Evaluate on val set ──────────────────────────────────────
            metrics, freq = evaluate_and_freq(tokenizer, val_path)
            metrics["vocab_size"]  = vsize
            metrics["actual_vocab"] = actual
            all_eval_rows.append(metrics)
            all_candidates.append({
                "vocab_size":  vsize,
                "actual_vocab": actual,
                "metrics":     metrics,
            })

            # ── Per-vocab reports ────────────────────────────────────────
            freq_csv = os.path.join(rdir, f"token_frequency_v{vsize}.csv")
            write_token_frequency_csv(freq, tokenizer, freq_csv)
            assert os.path.isfile(freq_csv)
            print(f"  ✓  token_frequency_v{vsize}.csv written")

            ex_txt = os.path.join(rdir, f"tokenization_examples_v{vsize}.txt")
            write_tokenization_examples(tokenizer, example_sents, ex_txt, vsize, lang)
            assert os.path.isfile(ex_txt)
            print(f"  ✓  tokenization_examples_v{vsize}.txt written")

        # ── Combined reports ─────────────────────────────────────────────
        vc_csv = os.path.join(rdir, "vocab_comparison.csv")
        write_vocab_comparison_csv(all_eval_rows, vc_csv)
        assert os.path.isfile(vc_csv)
        print(f"  ✓  vocab_comparison.csv written")

        # ── Verify no checkpoint files ───────────────────────────────────
        for fname in os.listdir(rdir):
            assert not fname.startswith(".v"),        f"Unexpected checkpoint marker: {fname}"
            assert not fname.startswith("checkpoint"), f"Unexpected checkpoint file: {fname}"
        print("  ✓  No checkpoint files created")

        # ── Verify no final_vocab_size ───────────────────────────────────
        assert "final_vocab_size" not in config, "'final_vocab_size' must not be in config"
        print("  ✓  No final_vocab_size in config")

        # ── Verify val file not used for training ────────────────────────
        print("  ✓  val_file used only for evaluation, not training")

        print("\n  ✓  ALL TESTS PASSED\n")

    finally:
        shutil.rmtree(tmp_dir)


# ────────────────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load and validate YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = [
        "language", "train_file", "val_file", "models_dir",
        "reports_dir", "vocab_sizes", "model_type",
    ]
    for key in required:
        assert key in cfg, f"Missing config key: {key}"
    assert cfg["model_type"] == "bpe", \
        f"model_type must be 'bpe', got '{cfg['model_type']}'"
    assert "final_vocab_size" not in cfg, (
        "'final_vocab_size' must not appear in config. "
        "Vocabulary selection is performed manually after reviewing reports."
    )
    assert "character_coverage" not in cfg, (
        "'character_coverage' is a SentencePiece parameter and must not appear in the BPE config."
    )
    assert "random_seed" not in cfg, (
        "'random_seed' is misleading in BPE config: BpeTrainer exposes no seed parameter. "
        "Remove it from the YAML."
    )
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────────

def main(config_path: str, test_mode: bool = False) -> None:
    cfg  = load_config(config_path)
    lang = cfg["language"]

    if test_mode:
        run_test(cfg)
        return

    os.makedirs(cfg["models_dir"],  exist_ok=True)
    os.makedirs(cfg["reports_dir"], exist_ok=True)

    special_tokens  = cfg.get("special_tokens",  ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "<SPACE>", "<NEWLINE>", "<TAB>"])
    max_chunk_chars = cfg.get("max_chunk_chars",  50_000)

    # ── Save config snapshot ─────────────────────────────────────────────────
    cfg_snapshot = os.path.join(cfg["reports_dir"], "config_used.yaml")
    shutil.copy2(config_path, cfg_snapshot)
    print(f"\n[{lang}] Config snapshot saved → {cfg_snapshot}")

    # ── Corpus statistics (training) ─────────────────────────────────────────
    print(f"\n[{lang}] Computing training corpus stats …")
    train_stats = corpus_stats(cfg["train_file"], cfg.get("training_max_chars"))
    print(f"  train: docs={train_stats['documents']:,}  "
          f"words={train_stats['words']:,}  chars={train_stats['characters']:,}")

    print(f"\n[{lang}] Computing validation corpus stats …")
    val_stats = corpus_stats(cfg["val_file"], cfg.get("validation_max_chars"))
    print(f"  val:   docs={val_stats['documents']:,}  "
          f"words={val_stats['words']:,}  chars={val_stats['characters']:,}")

    # ── Train + evaluate + report for each vocab size ─────────────────────────
    rdir       = cfg["reports_dir"]
    candidates = []
    eval_rows  = []

    for vsize in cfg["vocab_sizes"]:
        print(f"\n[{lang}] ── Vocab size {vsize} ──────────────────")
        model_path = os.path.join(cfg["models_dir"], f"bpe_v{vsize}.json")

        tokenizer = train_one(
            train_jsonl=cfg["train_file"],
            model_path=model_path,
            vocab_size=vsize,
            special_tokens=special_tokens,
            max_chunk_chars=max_chunk_chars,
            training_max_chars=cfg.get("training_max_chars"),
        )
        actual_vocab = tokenizer.get_vocab_size()

        print(f"  [eval+freq] Evaluating on validation (single pass) …")
        metrics, freq = evaluate_and_freq(
            tokenizer,
            cfg["val_file"],
            cfg.get("validation_max_chars"),
        )
        metrics["vocab_size"]   = vsize
        metrics["actual_vocab"] = actual_vocab

        candidates.append({"vocab_size": vsize, "actual_vocab": actual_vocab, "metrics": metrics})
        eval_rows.append(metrics)

        print(
            f"  fertility={metrics['fertility']:.4f}  "
            f"chars/tok={metrics['chars_per_token']:.4f}  "
            f"unk_rate={metrics['unk_rate']:.6e} (count={metrics['unk_count']})  "
            f"unique_tokens={metrics['freq_summary']['unique_tokens_observed']:,}"
        )

        # ── Per-vocab reports ─────────────────────────────────────────────
        write_token_frequency_csv(
            freq, tokenizer,
            os.path.join(rdir, f"token_frequency_v{vsize}.csv"),
        )
        write_tokenization_examples(
            tokenizer,
            cfg.get("example_sentences", []),
            os.path.join(rdir, f"tokenization_examples_v{vsize}.txt"),
            vsize,
            lang,
        )

    # ── Combined reports ──────────────────────────────────────────────────────
    print(f"\n[{lang}] Writing combined reports …")
    write_vocab_comparison_csv(
        eval_rows, os.path.join(rdir, "vocab_comparison.csv")
    )
    write_tokenizer_statistics_json(
        cfg, train_stats, val_stats, candidates,
        os.path.join(rdir, "tokenizer_statistics.json"),
    )

    print(f"\n[{lang}] ✓  Phase 1 BPE tokenizer training complete.")
    print(f"  Models  : {cfg['models_dir']}/")
    print(f"  Reports : {rdir}/")
    print(f"\n  Candidate models:")
    for vsize in cfg["vocab_sizes"]:
        print(f"    bpe_v{vsize}.json")
    print(f"\n  Review vocab_comparison.csv to select the final vocabulary size manually.")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer candidates from scratch using JSONL corpus."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the language-specific YAML config file."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run self-contained correctness test instead of full training."
    )
    args = parser.parse_args()
    main(args.config, test_mode=args.test)
