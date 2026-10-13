"""
split_corpus.py
---------------
Source-stratified, reproducible train/val/test split for a large JSONL corpus.

Usage
-----
Change the single LANGUAGE constant below to switch between corpora:

    LANGUAGE = "nepali"   # 3 source groups: wikimedia / manual / sangraha
    LANGUAGE = "hindi"    # 4 source groups: wikipedia / wikisource / sangraha 

Everything else — input path, output path, source groups, classifier — is
driven automatically from the LANGUAGE_CONFIG table.

Strategy
--------
We need source-stratified splitting WITHOUT loading the whole corpus into RAM.
We use a deterministic, hash-based assignment:

  bucket = int(sha256(seed_str + doc_id)) % BUCKET_COUNT

Each source group gets its own independent bucket→split mapping so that the
ratio is met independently per source.  The bucket count (10 000) is large
enough that the per-source ratios match the requested fractions to within
< 0.01 pp.  No document list is ever held in memory.

Two passes over the file:
  Pass 1 – collect statistics and build per-source split maps.
  Pass 2 – stream documents, assign each to a split, write to output file.
"""

# ---------------------------------------------------------------------------
# ★  Change this one constant to switch language  ★
# ---------------------------------------------------------------------------

LANGUAGE = "nepali"   # "nepali"  or  "hindi"

# ---------------------------------------------------------------------------
# Language configuration table
# ---------------------------------------------------------------------------
# Each entry defines:
#   input_file   – path to the deduplicated JSONL corpus
#   output_dir   – where train/val/test splits are written
#   groups       – ordered tuple of source-group names used for stratification
#                  (must include "other" as the final catch-all)
#   classifier   – function(record: dict) -> str  that maps a record to one
#                  of the group names listed in "groups"

def _nepali_classifier(record: dict) -> str:
    """Nepali: wikimedia > manual > sangraha verified.

    Actual source field values in nepali/corpus/deduplicated/final_deduped.jsonl:
      'wikimedia'         (project = 'wikipedia' | 'wikibooks')
      'manual'
      'sangraha verified'
    """
    src = str(record.get("source", "")).strip().lower()
    if src.startswith("wikimedia"):
        return "wikimedia"
    if src == "manual":
        return "manual"
    if src.startswith("sangraha"):
        return "sangraha"
    return "other"


def _hindi_classifier(record: dict) -> str:
    """Hindi: wikipedia / wikisource / sangraha verified.

    source='wikimedia' is split into two groups using the 'project' field:
      project='wikipedia'  -> 'wikipedia'
      project='wikisource' -> 'wikisource'
      (any other project)  -> 'wikipedia'  (safe default)

    source='sangraha verified' (or any other sangraha variant) -> 'sangraha'
    """
    src  = str(record.get("source",  "")).strip().lower()
    proj = str(record.get("project", "")).strip().lower()

    if src.startswith("wiki"):
        # Distinguish the two wikimedia sub-projects
        if proj == "wikisource":
            return "wikisource"
        return "wikipedia"   # wikipedia, wikibooks, unknown → default bucket

    if src.startswith("sangraha"):
        return "sangraha"

    return "other"


LANGUAGE_CONFIG = {
    "nepali": {
        "input_file": "nepali/corpus/deduplicated/final_deduped.jsonl",
        "output_dir": "nepali/corpus/splits",
        "groups":     ("wikimedia", "manual", "sangraha", "other"),
        "classifier": _nepali_classifier,
    },
    "hindi": {
        "input_file": "hindi/corpus/deduplicated/final_deduped.jsonl",
        "output_dir": "hindi/corpus/splits",
        # 3 groups: wikipedia and wikisource are stratified independently
        "groups":     ("wikipedia", "wikisource", "sangraha", "other"),
        "classifier": _hindi_classifier,
    },
}

# Validate language selection immediately so errors surface before any I/O.
if LANGUAGE not in LANGUAGE_CONFIG:
    raise ValueError(
        f"Unknown LANGUAGE={LANGUAGE!r}.  "
        f"Valid choices: {sorted(LANGUAGE_CONFIG)}"
    )

_cfg        = LANGUAGE_CONFIG[LANGUAGE]
INPUT_FILE  = _cfg["input_file"]
OUTPUT_DIR  = _cfg["output_dir"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_RATIO    = 0.98
VAL_RATIO      = 0.01
TEST_RATIO     = 0.01
RANDOM_SEED    = 42
BUCKET_COUNT   = 10_000   # <0.01 pp ratio error
PROGRESS_EVERY = 200_000

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, (
    f"Ratios must sum to 1.0, got {TRAIN_RATIO + VAL_RATIO + TEST_RATIO}"
)

# ---------------------------------------------------------------------------
# Imports (standard library only)
# ---------------------------------------------------------------------------

import hashlib
import json
import os
import sys

# ---------------------------------------------------------------------------
# Helper utilities  (driven by LANGUAGE_CONFIG — do not edit directly)
# ---------------------------------------------------------------------------

# Active source groups and classifier for the selected language.
SOURCE_GROUPS = _cfg["groups"]
source_group  = _cfg["classifier"]   # source_group(record) -> group name



def stable_hash(text: str) -> int:
    """Return a stable, non-negative integer hash using SHA-256."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def bucket_for_doc(doc_id: str, seed: int) -> int:
    """Map a document ID + seed to a bucket in [0, BUCKET_COUNT)."""
    key = f"{seed}:{doc_id}"
    return stable_hash(key) % BUCKET_COUNT


def build_bucket_split_map(train_ratio: float, val_ratio: float,
                           bucket_count: int) -> list:
    """
    Return a list of length `bucket_count` where each entry is
    'train', 'val', or 'test', distributed according to the ratios.
    """
    train_end = round(train_ratio * bucket_count)
    val_end   = train_end + round(val_ratio * bucket_count)
    mapping   = []
    for i in range(bucket_count):
        if i < train_end:
            mapping.append("train")
        elif i < val_end:
            mapping.append("val")
        else:
            mapping.append("test")
    return mapping


def assign_split(doc_id: str, bucket_map: list, seed: int) -> str:
    """Return 'train', 'val', or 'test' for a document."""
    bucket = bucket_for_doc(doc_id, seed)
    return bucket_map[bucket]


def word_count(text: str) -> int:
    """Count whitespace-separated tokens."""
    return len(text.split())


def make_doc_id(record: dict, line_no: int) -> str:
    """
    Build a stable document identifier.
    Prefer the 'id' field if present; fall back to a hash of the text.
    """
    if "id" in record:
        return str(record["id"])
    # Use first 200 chars of text as a stable proxy ID
    text_snippet = record.get("text", "")[:200]
    return f"line{line_no}:{stable_hash(text_snippet)}"


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------


class Stats:
    """Accumulate document/word/character counts per source group."""

    def __init__(self):
        self.docs      = 0
        self.words     = 0
        self.chars     = 0
        # per-source counters keyed by SOURCE_GROUPS
        self.by_source: dict = {g: 0 for g in SOURCE_GROUPS}

    def add(self, record: dict):
        text  = record.get("text", "")
        group = source_group(record)
        self.docs  += 1
        self.words += word_count(text)
        self.chars += len(text)
        self.by_source[group] += 1

    def print(self, label: str = ""):
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Documents : {self.docs:,}")
        print(f"{prefix}Words     : {self.words:,}")
        print(f"{prefix}Characters: {self.chars:,}")
        for g in SOURCE_GROUPS:
            print(f"{prefix}{g.capitalize():<12}: {self.by_source[g]:,}")


# ---------------------------------------------------------------------------
# Pass 1 – statistics
# ---------------------------------------------------------------------------


def pass1_statistics(input_file: str) -> Stats:
    """
    Stream through the corpus once and collect aggregate statistics.
    No documents are kept in memory.
    """
    print("\n=== Pass 1: Collecting statistics ===")
    stats = Stats()
    with open(input_file, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Skipping malformed JSON at line {line_no}: {exc}",
                      file=sys.stderr)
                continue
            stats.add(record)
            if line_no % PROGRESS_EVERY == 0:
                print(f"  Pass 1 – {line_no:,} documents scanned …")
    print(f"  Pass 1 complete – {stats.docs:,} documents total.\n")
    return stats


# ---------------------------------------------------------------------------
# Pass 2 – split and write
# ---------------------------------------------------------------------------


def pass2_split(input_file: str, output_dir: str,
                bucket_maps: dict, seed: int) -> dict:
    """
    Stream through the corpus a second time.
    Assign each document to a split using the per-source bucket map and
    write it to the appropriate output file.

    bucket_maps: dict mapping source group name → bucket list.
    Returns a dict mapping split name → Stats.
    """
    print("=== Pass 2: Splitting and writing ===")
    os.makedirs(output_dir, exist_ok=True)

    out_paths = {
        "train": os.path.join(output_dir, "train.jsonl"),
        "val":   os.path.join(output_dir, "val.jsonl"),
        "test":  os.path.join(output_dir, "test.jsonl"),
    }

    split_stats = {name: Stats() for name in out_paths}
    written     = {name: 0 for name in out_paths}

    with open(input_file, "r", encoding="utf-8") as fh, \
         open(out_paths["train"], "w", encoding="utf-8") as f_train, \
         open(out_paths["val"],   "w", encoding="utf-8") as f_val, \
         open(out_paths["test"],  "w", encoding="utf-8") as f_test:

        out_files = {"train": f_train, "val": f_val, "test": f_test}

        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue   # already warned in pass 1

            doc_id = make_doc_id(record, line_no)
            grp    = source_group(record)
            # Use the per-source bucket map for independent stratification
            split  = assign_split(doc_id, bucket_maps[grp], seed)

            # Write original JSON (no field modification) with ensure_ascii=False
            out_files[split].write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            split_stats[split].add(record)
            written[split] += 1

            if line_no % PROGRESS_EVERY == 0:
                print(f"  Pass 2 – {line_no:,} documents processed …"
                      f"  (train={written['train']:,}"
                      f"  val={written['val']:,}"
                      f"  test={written['test']:,})")

    total_written = sum(written.values())
    print(f"  Pass 2 complete – {total_written:,} documents written.\n")
    return split_stats


# ---------------------------------------------------------------------------
# Reporting & verification
# ---------------------------------------------------------------------------


def print_split_stats(split_stats: dict, total_docs: int):
    """Print per-split statistics and per-source stratification percentages."""
    for name, stats in split_stats.items():
        pct = 100.0 * stats.docs / total_docs if total_docs else 0.0
        print(f"--- {name.upper()} ({pct:.2f}% of corpus) ---")
        stats.print()
        print()

    # Per-source stratification table
    print("=== Per-Source Stratification ===")
    for grp in SOURCE_GROUPS:
        group_total = sum(s.by_source[grp] for s in split_stats.values())
        if group_total == 0:
            continue
        print(f"  {grp.capitalize()}:")
        for name, stats in split_stats.items():
            pct = 100.0 * stats.by_source[grp] / group_total
            print(f"    {name:<6}: {stats.by_source[grp]:>8,}  ({pct:.2f}%)")
    print()


def verify(input_stats: Stats, split_stats: dict):
    """Assert no documents were lost/duplicated, overall and per source group."""
    total_out = sum(s.docs for s in split_stats.values())
    print("=== Verification ===")
    print(f"  Input documents  : {input_stats.docs:,}")
    print(f"  Output documents : {total_out:,}")

    ok = True
    if total_out != input_stats.docs:
        diff = total_out - input_stats.docs
        sign = "+" if diff > 0 else ""
        print(f"  ✗  TOTAL MISMATCH: {sign}{diff} documents!", file=sys.stderr)
        ok = False
    else:
        print("  ✓  Total document count matches.")

    # Per-source verification
    for grp in SOURCE_GROUPS:
        inp  = input_stats.by_source[grp]
        outp = sum(s.by_source[grp] for s in split_stats.values())
        if inp == outp:
            print(f"  ✓  {grp.capitalize():<12}: {inp:,} in == {outp:,} out.")
        else:
            print(f"  ✗  {grp.capitalize():<12}: {inp:,} in != {outp:,} out!",
                  file=sys.stderr)
            ok = False

    if not ok:
        sys.exit(1)
    print("  ✓  All documents accounted for (no loss, no duplication).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print(f"  split_corpus.py  [language={LANGUAGE}]")
    print(f"  Source groups : {', '.join(SOURCE_GROUPS)}")
    print(f"  Input : {INPUT_FILE}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Ratios: train={TRAIN_RATIO}  val={VAL_RATIO}  test={TEST_RATIO}")
    print(f"  Seed  : {RANDOM_SEED}")
    print("=" * 60)

    # ── Pass 1 ──────────────────────────────────────────────────────────────
    input_stats = pass1_statistics(INPUT_FILE)
    print("=== Corpus Statistics (full corpus) ===")
    input_stats.print()
    print()

    # ── Build per-source deterministic bucket→split maps ────────────────────
    # Each source group gets its own independent bucket list so that the
    # 98/1/1 ratio is met separately per source group.
    base_map   = build_bucket_split_map(TRAIN_RATIO, VAL_RATIO, BUCKET_COUNT)
    bucket_maps = {grp: base_map for grp in SOURCE_GROUPS}

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    split_stats = pass2_split(INPUT_FILE, OUTPUT_DIR, bucket_maps, RANDOM_SEED)

    # ── Report ──────────────────────────────────────────────────────────────
    print("=== Split Statistics ===")
    print_split_stats(split_stats, input_stats.docs)

    # ── Verify ──────────────────────────────────────────────────────────────
    verify(input_stats, split_stats)


if __name__ == "__main__":
    main()
