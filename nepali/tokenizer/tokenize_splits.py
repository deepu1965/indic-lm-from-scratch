"""
tokenize_splits.py
==================
Phase 2: Apply the already-trained FINAL 10K BPE tokenizer to every document
in the train / validation / test splits.

Usage
-----
    # Run from the project root directory:
    python3 hindi/tokenize_splits.py
    python3 nepali/tokenize_splits.py

    # Or copy to the language folder and run from project root:
    python3 tokenize_splits.py          # (after editing LANGUAGE constants below)

What this script does
---------------------
1. Loads the frozen 10K BPE tokenizer -- does NOT retrain it.
2. Streams every JSONL record from each split (train / val / test) one at a time.
3. For each record:
   - Preserves ALL existing JSON fields (id, source, title, doc_id, ...)
   - Removes the "text" field
   - Adds  "tokens": [int, int, ...]   (token IDs from the 10K BPE model)
4. Writes output to corpus/tokenised_splits/ as valid UTF-8 JSONL.
5. Collects per-split, per-source statistics and vocabulary-usage statistics.
6. Saves report files to tokenizer/reports/ without overwriting Phase 1 reports.

Memory strategy
---------------
Documents are processed one at a time.  No full file is ever loaded into RAM.
Vocabulary-frequency counting uses a fixed-length array of size vocab_size
(approx 10 000 integers) so memory is O(vocab_size), not O(corpus_tokens).

Source classification
---------------------
Classification is driven by the actual "source" field in each JSON record:
  "wikimedia"         -> Manual   (Wikipedia / Wikisource — manually curated)
  "manual"            -> Manual   (Nepali manual scrape)
  "sangraha verified" -> Public/Downloaded
  anything else       -> Other  (logged as a warning)

Note: Both Hindi and Nepali treat wikimedia as Manual.
Hindi manual data comes from Wikimedia (project='wikipedia' or 'wikisource').
Nepali also has source='manual' records in addition to wikimedia.
Only 'sangraha verified' is Public/Downloaded in both languages.

Do NOT modify this script's tokenization logic.
All downstream code must treat it as the canonical Phase 2 tokenisation step.
"""

# =============================================================================
#  CONFIGURATION
#  ─────────────────────────────────────────────────────────────────────────────
#  These are the ONLY constants you need to change when switching
#  between Hindi and Nepali.  Everything else is language-agnostic.
#
#  For NEPALI -- change the five lines below to:
#    LANGUAGE          = "nepali"
#    INPUT_SPLITS_DIR  = "nepali/corpus/splits"
#    OUTPUT_SPLITS_DIR = "nepali/corpus/tokenised_splits"
#    TOKENIZER_PATH    = "nepali/tokenizer/models/bpe_v10000.json"
#    REPORTS_DIR       = "nepali/tokenizer/reports"
# =============================================================================

LANGUAGE          = "nepali"
INPUT_SPLITS_DIR  = "nepali/corpus/splits"
OUTPUT_SPLITS_DIR = "nepali/corpus/tokenised_splits"
TOKENIZER_PATH    = "nepali/tokenizer/models/bpe_v10000.json"
REPORTS_DIR       = "nepali/tokenizer/reports"

# Runtime options (usually no need to change) ---------------------------------
# Number of sample documents to print for decode sanity-check per split.
SAMPLE_DECODE_COUNT = 3

# Progress reporting interval (lines read from input file).
PROGRESS_EVERY = 100_000

# =============================================================================
# Standard-library imports only up here; tokenizers is imported after checks
# =============================================================================

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

# Add parent directory to path to import tokenizer_utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tokenizer_utils import preprocess_text, custom_decode

# =============================================================================
# Source classification
# =============================================================================

# Canonical category names used throughout the script.
CAT_MANUAL = "manual"
CAT_PUBLIC = "public"
CAT_OTHER  = "other"


def classify_source(record: dict) -> str:
    """Return the source category for a record based on its 'source' field.

    Classification is driven purely by the actual value of the 'source' key
    in the JSON object -- never by file names or positional assumptions.

    Known source values
    -------------------
    'wikimedia'         -> CAT_MANUAL  (Wikipedia / Wikisource -- manually curated;
                                        present in both Hindi and Nepali)
    'manual'            -> CAT_MANUAL  (Nepali manual scrape)
    'sangraha verified' -> CAT_PUBLIC  (Sangraha public/downloaded data)
    anything else       -> CAT_OTHER   (unexpected -- will be warned about)

    Hindi note: Hindi manual data comes from Wikimedia (source='wikimedia',
    project='wikipedia' or 'wikisource').  Both sub-projects are Manual.
    """
    src = str(record.get("source", "")).strip().lower()
    if src.startswith("wikimedia"):
        return CAT_MANUAL
    if src == "manual":
        return CAT_MANUAL
    if src.startswith("sangraha"):
        return CAT_PUBLIC
    return CAT_OTHER


# =============================================================================
# Statistics containers
# =============================================================================

class SplitStats:
    """Accumulate token/document counts per source category for one split."""

    CATEGORIES = (CAT_MANUAL, CAT_PUBLIC, CAT_OTHER)

    def __init__(self, split_name: str):
        self.split_name       = split_name
        self.docs_ok          = 0   # successfully processed
        self.docs_skipped     = 0   # malformed / empty text
        self.docs_empty_text  = 0   # text field missing or blank
        self.docs_malformed   = 0   # JSON parse failures
        self.unknown_sources: dict = {}  # source_value -> count (for CAT_OTHER)

        # Per-category counters
        self.cat_docs:   dict = {c: 0 for c in self.CATEGORIES}
        self.cat_tokens: dict = {c: 0 for c in self.CATEGORIES}

    # ── Mutators ──────────────────────────────────────────────────────────────

    def record_ok(self, category: str, n_tokens: int, source_raw: str):
        self.docs_ok += 1
        self.cat_docs[category]   += 1
        self.cat_tokens[category] += n_tokens
        if category == CAT_OTHER:
            self.unknown_sources[source_raw] = (
                self.unknown_sources.get(source_raw, 0) + 1
            )

    def record_empty_text(self):
        self.docs_skipped    += 1
        self.docs_empty_text += 1

    def record_malformed(self):
        self.docs_skipped   += 1
        self.docs_malformed += 1

    # ── Derived totals ─────────────────────────────────────────────────────────

    @property
    def total_docs(self) -> int:
        return self.docs_ok + self.docs_skipped

    @property
    def total_tokens(self) -> int:
        return sum(self.cat_tokens.values())

    @property
    def manual_tokens(self) -> int:
        return self.cat_tokens[CAT_MANUAL]

    @property
    def public_tokens(self) -> int:
        return self.cat_tokens[CAT_PUBLIC]

    def avg_tokens_per_doc(self) -> float:
        return self.total_tokens / self.docs_ok if self.docs_ok else 0.0

    # ── Pretty print ──────────────────────────────────────────────────────────

    def print_summary(self):
        name = self.split_name.upper()
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")
        print(f"  Documents processed : {self.docs_ok:>14,}")
        print(f"  Documents skipped   : {self.docs_skipped:>14,}")
        if self.docs_empty_text:
            print(f"    |__ empty/missing text : {self.docs_empty_text:>10,}")
        if self.docs_malformed:
            print(f"    |__ JSON malformed     : {self.docs_malformed:>10,}")
        print()
        print(f"  {'Category':<22}  {'Documents':>12}  {'Tokens':>14}  {'Avg tok/doc':>12}")
        print(f"  {'─'*64}")
        for cat in self.CATEGORIES:
            nd  = self.cat_docs[cat]
            nt  = self.cat_tokens[cat]
            avg = nt / nd if nd else 0.0
            label = cat.capitalize() + "/Downloaded" if cat == CAT_PUBLIC else cat.capitalize()
            print(f"  {label:<22}  {nd:>12,}  {nt:>14,}  {avg:>12.1f}")
        print(f"  {'─'*64}")
        nd  = self.docs_ok
        nt  = self.total_tokens
        avg = self.avg_tokens_per_doc()
        print(f"  {'Total':<22}  {nd:>12,}  {nt:>14,}  {avg:>12.1f}")
        if self.unknown_sources:
            print(f"\n  WARNING: Unknown source values classified as 'other':")
            for sv, cnt in sorted(self.unknown_sources.items(),
                                  key=lambda x: -x[1]):
                print(f"       {sv!r}: {cnt:,}")


class VocabStats:
    """Track token-ID frequencies using a fixed-size array for memory efficiency."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.freq = [0] * vocab_size   # freq[token_id] = total occurrences

    def update(self, token_ids: list):
        for tid in token_ids:
            self.freq[tid] += 1

    # ── Derived statistics ────────────────────────────────────────────────────

    @property
    def total_occurrences(self) -> int:
        return sum(self.freq)

    @property
    def unique_observed(self) -> int:
        return sum(1 for f in self.freq if f > 0)

    @property
    def zero_frequency_count(self) -> int:
        return sum(1 for f in self.freq if f == 0)

    def coverage_pct(self) -> float:
        return 100.0 * self.unique_observed / self.vocab_size if self.vocab_size else 0.0

    def exactly_once(self) -> int:
        return sum(1 for f in self.freq if f == 1)

    def fewer_than(self, threshold: int) -> int:
        return sum(1 for f in self.freq if 0 < f < threshold)

    def merge(self, other: "VocabStats") -> "VocabStats":
        """Return a new VocabStats combining self and other (must have same vocab_size)."""
        assert self.vocab_size == other.vocab_size
        merged = VocabStats(self.vocab_size)
        for i in range(self.vocab_size):
            merged.freq[i] = self.freq[i] + other.freq[i]
        return merged

    def print_summary(self, label: str = ""):
        prefix = f"  [{label}]" if label else " "
        print(f"\n{prefix} Vocabulary coverage:")
        print(f"    Vocab size                : {self.vocab_size:>10,}")
        print(f"    Total token occurrences   : {self.total_occurrences:>10,}")
        print(f"    Unique token IDs observed : {self.unique_observed:>10,}")
        print(f"    Coverage                  : {self.coverage_pct():>10.2f}%")
        print(f"    Never-observed tokens     : {self.zero_frequency_count:>10,}")
        print(f"    Tokens occurring  = 1     : {self.exactly_once():>10,}")
        print(f"    Tokens occurring  < 10    : {self.fewer_than(10):>10,}")
        print(f"    Tokens occurring  < 100   : {self.fewer_than(100):>10,}")


# =============================================================================
# Core processing
# =============================================================================

def process_split(
    split_name: str,
    input_path: str,
    output_path: str,
    tokenizer,
    vocab_size: int,
    sample_decode_count: int,
) -> tuple:
    """Stream-tokenize one split.  Returns (SplitStats, VocabStats).

    For each input record:
    - All original JSON fields are preserved except "text".
    - "tokens": [int, ...] is added.
    - The output JSONL is written one line at a time -- no full-file buffering.

    Malformed records are skipped and counted; processing continues.

    Parameters
    ----------
    split_name           : 'train', 'val', or 'test'
    input_path           : path to input JSONL
    output_path          : destination JSONL path (created/overwritten)
    tokenizer            : loaded Tokenizer object
    vocab_size           : tokenizer.get_vocab_size()
    sample_decode_count  : how many sample docs to print for sanity checking

    Returns
    -------
    stats       : SplitStats for this split
    vocab_stats : VocabStats for this split
    """
    stats       = SplitStats(split_name)
    vocab_stats = VocabStats(vocab_size)

    # Samples for decode sanity-check (collected from first N successful docs)
    decode_samples = []

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    t0 = time.time()
    print(f"\n[{split_name}] Reading  : {input_path}")
    print(f"[{split_name}] Writing  : {output_path}")

    with open(input_path,  "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line_no, raw_line in enumerate(f_in, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # ── Parse JSON ────────────────────────────────────────────────────
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                stats.record_malformed()
                if stats.docs_malformed <= 10:
                    print(
                        f"  [WARN] Malformed JSON at line {line_no}: {exc}",
                        file=sys.stderr,
                    )
                continue

            # ── Extract text ──────────────────────────────────────────────────
            text = record.get("text", "")
            if not isinstance(text, str) or not text.strip():
                stats.record_empty_text()
                continue

            # ── Tokenize ──────────────────────────────────────────────────────
            preprocessed_text = preprocess_text(text)
            encoding  = tokenizer.encode(preprocessed_text)
            token_ids = encoding.ids   # list[int]

            if not token_ids:
                # Tokenization returned nothing (guard against edge cases)
                stats.record_empty_text()
                continue

            # ── Validate IDs are in range (first 1000 docs + spot-check) ─────
            if line_no <= 1000 or line_no % 50_000 == 0:
                bad_ids = [tid for tid in token_ids
                           if not (0 <= tid < vocab_size)]
                if bad_ids:
                    print(
                        f"  [WARN] Out-of-range token IDs at line {line_no}: "
                        f"{bad_ids[:5]} (vocab_size={vocab_size})",
                        file=sys.stderr,
                    )

            # ── Build output record ───────────────────────────────────────────
            # Preserve every field except "text"; add "tokens".
            out_record = {k: v for k, v in record.items() if k != "text"}
            out_record["tokens"] = token_ids

            # ── Write output line (valid UTF-8 JSONL, native Python ints) ─────
            f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            # ── Update statistics ─────────────────────────────────────────────
            category = classify_source(record)
            src_raw  = str(record.get("source", "")).strip()
            stats.record_ok(category, len(token_ids), src_raw)
            vocab_stats.update(token_ids)

            # ── Collect decode samples ────────────────────────────────────────
            if len(decode_samples) < sample_decode_count:
                decode_samples.append({
                    "line_no":      line_no,
                    "source":       src_raw,
                    "text_snippet": text[:120],
                    "token_ids":    token_ids[:20],
                    "decoded":      custom_decode(tokenizer, token_ids[:20]),
                })

            # ── Progress ──────────────────────────────────────────────────────
            if line_no % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate    = stats.docs_ok / elapsed if elapsed > 0 else 0
                print(
                    f"  [{split_name}] {line_no:>10,} lines  "
                    f"{stats.docs_ok:>10,} ok  "
                    f"{stats.total_tokens:>14,} tokens  "
                    f"{rate:,.0f} docs/s",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"  [{split_name}] Done in {elapsed:.1f}s  "
        f"{stats.docs_ok:,} docs  {stats.total_tokens:,} tokens"
    )

    # ── Sanity-check: read back the output file ────────────────────────────────
    _verify_output(output_path, split_name, stats.docs_ok, vocab_size)

    # ── Print decode samples ───────────────────────────────────────────────────
    if decode_samples:
        print(f"\n  [{split_name}] Sample decode verification "
              f"({len(decode_samples)} docs):")
        for s in decode_samples:
            print(f"    Line {s['line_no']} | source={s['source']!r}")
            print(f"      text snippet  : {s['text_snippet']!r}")
            print(f"      first 20 IDs  : {s['token_ids']}")
            print(f"      decoded       : {s['decoded']!r}")

    return stats, vocab_stats


def _verify_output(output_path: str, split_name: str, expected_docs: int,
                   vocab_size: int) -> None:
    """Read back the output JSONL and perform sanity checks."""
    print(f"\n  [{split_name}] Verifying output ...")
    ok_count  = 0
    bad_count = 0
    with open(output_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_count += 1
                if bad_count <= 3:
                    print(f"  [WARN] Read-back parse error at line {i}",
                          file=sys.stderr)
                continue

            # Must have tokens, must not have text
            if "tokens" not in rec:
                bad_count += 1
                if bad_count <= 3:
                    print(f"  [WARN] Missing 'tokens' at line {i}",
                          file=sys.stderr)
                continue
            if "text" in rec:
                bad_count += 1
                if bad_count <= 3:
                    print(f"  [WARN] Unexpected 'text' still present at line {i}",
                          file=sys.stderr)
                continue

            tokens = rec["tokens"]
            if not isinstance(tokens, list) or not tokens:
                bad_count += 1
                continue
            if not all(isinstance(t, int) and 0 <= t < vocab_size
                       for t in tokens):
                bad_count += 1
                if bad_count <= 3:
                    bad = [t for t in tokens
                           if not isinstance(t, int) or
                           not (0 <= t < vocab_size)]
                    print(f"  [WARN] Invalid token IDs at line {i}: {bad[:5]}",
                          file=sys.stderr)
                continue

            ok_count += 1

    if ok_count != expected_docs:
        print(
            f"  [ERROR] COUNT MISMATCH: wrote {expected_docs:,} docs "
            f"but read back {ok_count:,}",
            file=sys.stderr,
        )
    else:
        print(
            f"  OK  Read-back: {ok_count:,} valid records "
            f"(all have 'tokens', none have stray 'text')"
        )
    if bad_count:
        print(f"  WARNING: {bad_count:,} problematic records on read-back",
              file=sys.stderr)


# =============================================================================
# Report writers
# =============================================================================

def write_statistics_json(
    language:       str,
    tokenizer_path: str,
    vocab_size:     int,
    splits_stats:   list,
    splits_vocab:   list,
    input_paths:    dict,
    output_paths:   dict,
    path:           str,
) -> None:
    """Write comprehensive statistics to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _split_dict(stats: SplitStats, vst: VocabStats) -> dict:
        return {
            "split":                 stats.split_name,
            "input_file":            input_paths.get(stats.split_name, ""),
            "output_file":           output_paths.get(stats.split_name, ""),
            "documents_ok":          stats.docs_ok,
            "documents_skipped":     stats.docs_skipped,
            "documents_empty_text":  stats.docs_empty_text,
            "documents_malformed":   stats.docs_malformed,
            "total_tokens":          stats.total_tokens,
            "avg_tokens_per_doc":    round(stats.avg_tokens_per_doc(), 2),
            "by_source_category": {
                cat: {
                    "documents":         stats.cat_docs[cat],
                    "tokens":            stats.cat_tokens[cat],
                    "avg_tokens_per_doc": round(
                        stats.cat_tokens[cat] / stats.cat_docs[cat], 2
                    ) if stats.cat_docs[cat] else 0,
                }
                for cat in SplitStats.CATEGORIES
            },
            "unknown_source_values": stats.unknown_sources,
            "vocab_usage": {
                "vocab_size":                  vocab_size,
                "total_token_occurrences":     vst.total_occurrences,
                "unique_token_ids_observed":   vst.unique_observed,
                "coverage_pct":                round(vst.coverage_pct(), 4),
                "zero_frequency_tokens":       vst.zero_frequency_count,
                "tokens_occurring_exactly_1":  vst.exactly_once(),
                "tokens_occurring_lt_10":      vst.fewer_than(10),
                "tokens_occurring_lt_100":     vst.fewer_than(100),
            },
        }

    data = {
        "phase":          "Phase 2 -- Corpus Tokenisation",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "language":       language,
        "tokenizer_path": tokenizer_path,
        "vocab_size":     vocab_size,
        "splits": [
            _split_dict(st, vst)
            for st, vst in zip(splits_stats, splits_vocab)
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n  [report] Wrote {path}")


def write_statistics_csv(
    language:       str,
    tokenizer_path: str,
    vocab_size:     int,
    splits_stats:   list,
    splits_vocab:   list,
    path:           str,
) -> None:
    """Write one CSV row per (split, source_category) plus a TOTAL row."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fields = [
        "language", "tokenizer_path", "vocab_size",
        "split", "source_category",
        "documents", "tokens", "avg_tokens_per_doc",
        "unique_token_ids", "coverage_pct", "zero_frequency_tokens",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stats, vst in zip(splits_stats, splits_vocab):
            for cat in SplitStats.CATEGORIES:
                nd = stats.cat_docs[cat]
                nt = stats.cat_tokens[cat]
                writer.writerow({
                    "language":            language,
                    "tokenizer_path":      tokenizer_path,
                    "vocab_size":          vocab_size,
                    "split":               stats.split_name,
                    "source_category":     cat,
                    "documents":           nd,
                    "tokens":              nt,
                    "avg_tokens_per_doc":  round(nt / nd, 2) if nd else 0,
                    # Vocab stats are per-split, not per-category
                    "unique_token_ids":      vst.unique_observed,
                    "coverage_pct":          round(vst.coverage_pct(), 4),
                    "zero_frequency_tokens": vst.zero_frequency_count,
                })
            # Total row for this split
            writer.writerow({
                "language":            language,
                "tokenizer_path":      tokenizer_path,
                "vocab_size":          vocab_size,
                "split":               stats.split_name,
                "source_category":     "TOTAL",
                "documents":           stats.docs_ok,
                "tokens":              stats.total_tokens,
                "avg_tokens_per_doc":  round(stats.avg_tokens_per_doc(), 2),
                "unique_token_ids":      vst.unique_observed,
                "coverage_pct":          round(vst.coverage_pct(), 4),
                "zero_frequency_tokens": vst.zero_frequency_count,
            })
    print(f"  [report] Wrote {path}")


def write_vocab_usage_csv(
    vocab_stats: VocabStats,
    tokenizer,
    path:        str,
    label:       str = "",
) -> None:
    """Write token_id / piece / frequency for every vocabulary entry.

    Tokens with zero frequency are included and appear at the bottom.
    Sorted by descending frequency, then ascending token_id as tiebreaker.

    This CSV lets you inspect exactly which of the 10K vocabulary tokens were
    actually observed in the tokenised corpus.

    Columns: token_id, piece, frequency
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vocab       = tokenizer.get_vocab()          # piece -> id
    id_to_piece = {v: k for k, v in vocab.items()}
    vs = vocab_stats.vocab_size

    rows = sorted(
        ((vocab_stats.freq[tid], tid) for tid in range(vs)),
        key=lambda x: (-x[0], x[1]),
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "piece", "frequency"])
        for freq, tid in rows:
            writer.writerow([tid, id_to_piece.get(tid, "?"), freq])

    tag = f" ({label})" if label else ""
    print(f"  [report] Wrote {path}{tag}")


# =============================================================================
# Grand summary printer
# =============================================================================

def print_grand_summary(splits_stats: list, splits_vocab: list) -> None:
    """Print the source-wise token count table."""
    print("\n" + "=" * 60)
    print("  SOURCE-WISE TOKEN COUNTS")
    print("=" * 60)
    cat_labels = {
        CAT_MANUAL: "Manual",
        CAT_PUBLIC: "Public/Downloaded",
        CAT_OTHER:  "Other",
    }
    for stats in splits_stats:
        name = stats.split_name.upper()
        print(f"\n  {name}")
        for cat, label in cat_labels.items():
            nd = stats.cat_docs[cat]
            nt = stats.cat_tokens[cat]
            print(f"    {label:<22}: {nt:>14,} tokens  ({nd:,} docs)")
        print(f"    {'Total':<22}: {stats.total_tokens:>14,} tokens  "
              f"({stats.docs_ok:,} docs)")

    print("\n" + "=" * 60)
    print("  VOCABULARY COVERAGE PER SPLIT")
    print("=" * 60)
    for stats, vst in zip(splits_stats, splits_vocab):
        vst.print_summary(label=stats.split_name.upper())

    # Combined all-splits vocab
    if len(splits_vocab) > 1:
        combined = splits_vocab[0]
        for vst in splits_vocab[1:]:
            combined = combined.merge(vst)
        combined.print_summary(label="ALL SPLITS COMBINED")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print(f"  Phase 2 -- Corpus Tokenisation")
    print(f"  Language  : {LANGUAGE}")
    print(f"  Tokenizer : {TOKENIZER_PATH}")
    print(f"  Input     : {INPUT_SPLITS_DIR}/")
    print(f"  Output    : {OUTPUT_SPLITS_DIR}/")
    print(f"  Reports   : {REPORTS_DIR}/")
    print("=" * 60)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    if not os.path.isfile(TOKENIZER_PATH):
        sys.exit(
            f"\n[ERROR] Tokenizer not found: {TOKENIZER_PATH}\n"
            "  Make sure you are running from the project root directory and\n"
            "  that the Phase 1 tokenizer has been trained.\n"
        )

    missing = []
    for split in ("train", "val", "test"):
        p = os.path.join(INPUT_SPLITS_DIR, f"{split}.jsonl")
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        sys.exit(f"\n[ERROR] Missing input splits:\n" +
                 "\n".join(f"  {p}" for p in missing) + "\n")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit(
            "\n[ERROR] The 'tokenizers' package is not installed.\n"
            "  Install it with:  pip install tokenizers\n"
        )

    print(f"\nLoading tokenizer ...  {TOKENIZER_PATH}")
    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    vocab      = tokenizer.get_vocab()
    print(f"  Vocabulary size : {vocab_size:,}")

    # Confirm special tokens
    for st in ("[UNK]", "[PAD]", "[BOS]", "[EOS]"):
        tid = vocab.get(st)
        if tid is None:
            print(f"  [WARN] Special token {st!r} not found in vocabulary",
                  file=sys.stderr)
        else:
            print(f"  {st} -> ID {tid}")

    # Create output directories
    os.makedirs(OUTPUT_SPLITS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Process each split ────────────────────────────────────────────────────
    SPLITS = [
        ("train", "train.jsonl"),
        ("val",   "val.jsonl"),
        ("test",  "test.jsonl"),
    ]

    input_paths  = {}
    output_paths = {}
    splits_stats = []
    splits_vocab = []

    for split_name, filename in SPLITS:
        inp = os.path.join(INPUT_SPLITS_DIR,  filename)
        out = os.path.join(OUTPUT_SPLITS_DIR, filename)
        input_paths[split_name]  = inp
        output_paths[split_name] = out

        stats, vst = process_split(
            split_name          = split_name,
            input_path          = inp,
            output_path         = out,
            tokenizer           = tokenizer,
            vocab_size          = vocab_size,
            sample_decode_count = SAMPLE_DECODE_COUNT,
        )
        stats.print_summary()
        splits_stats.append(stats)
        splits_vocab.append(vst)

    # ── Grand summary ─────────────────────────────────────────────────────────
    print_grand_summary(splits_stats, splits_vocab)

    # ── Reports ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Writing reports -> {REPORTS_DIR}/")
    print(f"{'─'*60}")

    write_statistics_json(
        language       = LANGUAGE,
        tokenizer_path = TOKENIZER_PATH,
        vocab_size     = vocab_size,
        splits_stats   = splits_stats,
        splits_vocab   = splits_vocab,
        input_paths    = input_paths,
        output_paths   = output_paths,
        path           = os.path.join(REPORTS_DIR,
                                      "phase2_tokenised_split_statistics.json"),
    )

    write_statistics_csv(
        language       = LANGUAGE,
        tokenizer_path = TOKENIZER_PATH,
        vocab_size     = vocab_size,
        splits_stats   = splits_stats,
        splits_vocab   = splits_vocab,
        path           = os.path.join(REPORTS_DIR,
                                      "phase2_tokenised_split_statistics.csv"),
    )

    # Vocabulary usage CSVs -- one per split, one combined across all
    train_stats, train_vst = splits_stats[0], splits_vocab[0]
    val_stats,   val_vst   = splits_stats[1], splits_vocab[1]
    test_stats,  test_vst  = splits_stats[2], splits_vocab[2]

    write_vocab_usage_csv(
        train_vst, tokenizer,
        path  = os.path.join(REPORTS_DIR, "phase2_vocab_usage_train.csv"),
        label = "train split",
    )
    write_vocab_usage_csv(
        val_vst, tokenizer,
        path  = os.path.join(REPORTS_DIR, "phase2_vocab_usage_val.csv"),
        label = "val split",
    )
    write_vocab_usage_csv(
        test_vst, tokenizer,
        path  = os.path.join(REPORTS_DIR, "phase2_vocab_usage_test.csv"),
        label = "test split",
    )

    # Combined: merge all three splits
    combined_vst = train_vst.merge(val_vst).merge(test_vst)
    write_vocab_usage_csv(
        combined_vst, tokenizer,
        path  = os.path.join(REPORTS_DIR, "phase2_vocab_usage_all.csv"),
        label = "all splits combined",
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2 tokenisation complete -- {LANGUAGE.upper()}")
    print(f"{'='*60}")
    print(f"\n  Tokenised splits : {OUTPUT_SPLITS_DIR}/")
    for split_name, filename in SPLITS:
        p = os.path.join(OUTPUT_SPLITS_DIR, filename)
        size_mb = os.path.getsize(p) / 1024**2 if os.path.isfile(p) else 0.0
        print(f"    {filename:<12}  {size_mb:>8.1f} MB")

    print(f"\n  Reports          : {REPORTS_DIR}/")
    for fname in [
        "phase2_tokenised_split_statistics.json",
        "phase2_tokenised_split_statistics.csv",
        "phase2_vocab_usage_train.csv",
        "phase2_vocab_usage_val.csv",
        "phase2_vocab_usage_test.csv",
        "phase2_vocab_usage_all.csv",
    ]:
        p = os.path.join(REPORTS_DIR, fname)
        size_kb = os.path.getsize(p) / 1024 if os.path.isfile(p) else 0.0
        print(f"    {fname:<52}  {size_kb:>7.1f} KB")

    print()


# =============================================================================
if __name__ == "__main__":
    main()
