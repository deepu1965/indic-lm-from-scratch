"""
final_tokens_selector.py
========================
Phase 2: Experimental ~500M-token training corpus selection.

Reads from the ALREADY TOKENISED train split (output of tokenize_splits.py).
Does NOT re-tokenize text.  Does NOT write output files.
PRINTS statistics only.

After reviewing the printed vocabulary coverage statistics, we will add a
WRITE_OUTPUT step to save the final selected corpus.

Algorithm — two passes, memory-efficient
-----------------------------------------
Pass 1  (index pass)
    Stream through the tokenised JSONL.
    For each record store ONLY (line_no, n_tokens) — no token arrays.
    Separate into Manual sub-groups and Public pool.
    Memory: ~80-100 MB for ~1.7 M records.

Selection  (in-memory, seeded)
    Manual:
        Subsample proportionally across manual sub-groups
        (wikimedia/wikipedia, wikimedia/wikibooks, wikimedia/wikisource,
         manual/scrape, etc.) to reach MANUAL_TARGET (~100 M tokens).
        Sub-groups smaller than their proportional quota are taken in full
        and the deficit is redistributed to the remaining sub-groups.
    Public:
        Shuffle public index with seeded rng, take greedily until
        public_target = TOTAL_TARGET - actual_manual_tokens is reached.

Pass 2  (stats pass)
    Stream through file again.
    For selected line numbers only: json.loads the record, accumulate
    vocabulary frequency array and per-source-group statistics.
    Non-selected lines: skip immediately after checking line number.
    Memory: O(vocab_size) for freq array (10K ints ≈ 40 KB).

Source classification (identical to tokenize_splits.py)
--------------------------------------------------------
    source = "wikimedia"          -> Manual
    source = "manual"             -> Manual
    source = "sangraha verified"  -> Public/Downloaded
    anything else                 -> Other (not selected)
"""

# ==============================================================================
#  CONFIGURATION — Change ONLY these constants to switch between languages.
# ==============================================================================

LANGUAGE       = "nepali"
TRAIN_JSONL    = "nepali/corpus/tokenised_splits/train.jsonl"
TOKENIZER_PATH = "nepali/tokenizer/models/bpe_v10000.json"

# Total token budget for the selected ~500 M-token training corpus.
TOTAL_TARGET = 500_000_000          # 500 M tokens

# Manual token target (~100 M, 20% of total).
# The script proportionally samples across manual sub-groups
# (wikimedia/wikipedia, wikimedia/wikibooks, wikimedia/wikisource,
#  manual/scrape, etc.) to reach this target.
# Sub-groups smaller than their proportional quota are taken in full;
# the deficit is redistributed to the remaining sub-groups.
MANUAL_TARGET = 100_000_000         # ~100 M manual tokens

# Fixed random seed for reproducibility.
RANDOM_SEED = 42

# Progress reporting interval (lines read).
PROGRESS_EVERY = 200_000

# Output directory for the final selected corpus.
# The script writes:  <OUTPUT_DIR>/train.jsonl
OUTPUT_DIR = "nepali/corpus/final_train"

# ==============================================================================
#  For HINDI — change the six values above to:
#    LANGUAGE       = "hindi"
#    TRAIN_JSONL    = "hindi/corpus/tokenised_splits/train.jsonl"
#    TOKENIZER_PATH = "hindi/tokenizer/models/bpe_v10000.json"
#    MANUAL_TARGET  = 100_000_000             # ~100M manual tokens
#    RANDOM_SEED    = 42                      # keep the same seed
#    OUTPUT_DIR     = "hindi/corpus/final_train"
#
#  Nepali corpus statistics (for reference):
#    Total tokens   : 1,270,578,286
#    Manual tokens  :   139,410,635
#    Public tokens  : 1,131,167,651
#  Target: ~100M manual (20%) + ~400M public (80%) = ~500M total
# ==============================================================================

import collections
import json
import os
import random
import sys
import time


# ==============================================================================
# Source classification
# ==============================================================================

CAT_MANUAL = "manual"
CAT_PUBLIC = "public"
CAT_OTHER  = "other"


def classify_source(source_str: str) -> str:
    """Project-level source classification — identical to tokenize_splits.py.

    'wikimedia'         -> CAT_MANUAL  (Wikipedia / Wikisource / Wikibooks)
    'manual'            -> CAT_MANUAL  (Nepali manual scrape)
    'sangraha verified' -> CAT_PUBLIC  (Sangraha public download)
    anything else       -> CAT_OTHER
    """
    s = str(source_str).strip().lower()
    if s.startswith("wikimedia"):
        return CAT_MANUAL
    if s == "manual":
        return CAT_MANUAL
    if s.startswith("sangraha"):
        return CAT_PUBLIC
    return CAT_OTHER


def subgroup_label(source: str, project: str) -> str:
    """Human-readable sub-group key for stratified reporting."""
    src  = str(source).strip().lower()
    proj = str(project).strip().lower() if project else ""
    if src.startswith("wikimedia") and proj:
        return f"wikimedia/{proj}"
    if src.startswith("wikimedia"):
        return "wikimedia"
    if src == "manual":
        return "manual/scrape"
    if src.startswith("sangraha"):
        return "sangraha"
    return f"other/{src}"


# ==============================================================================
# Pass 1: build lightweight index
# ==============================================================================

def pass1_build_index(path: str) -> tuple:
    """
    Single streaming pass: collect (line_no, n_tokens) per source group.

    Returns
    -------
    manual_groups : dict  subgroup_label -> list of (line_no, n_tokens)
    public_items  : list  of (line_no, n_tokens)
    other_count   : int   records that map to CAT_OTHER (not selected)
    malformed     : int   JSON-parse failures
    total_lines   : int   non-blank lines seen
    """
    manual_groups = collections.defaultdict(list)
    public_items  = []
    other_count   = 0
    malformed     = 0
    total_lines   = 0

    t0 = time.time()
    print(f"\n[Pass 1] Building index from {path} ...")

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total_lines += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue

            src  = rec.get("source", "")
            proj = rec.get("project", "")
            ntok = len(rec.get("tokens", []))
            cat  = classify_source(src)

            if cat == CAT_MANUAL:
                key = subgroup_label(src, proj)
                manual_groups[key].append((line_no, ntok))
            elif cat == CAT_PUBLIC:
                public_items.append((line_no, ntok))
            else:
                other_count += 1

            if line_no % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate    = line_no / elapsed if elapsed else 0
                print(f"  [Pass 1] {line_no:>10,} lines  "
                      f"{rate:>8,.0f} lines/s  elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  [Pass 1] Done. {total_lines:,} records in {elapsed:.1f}s  "
          f"(malformed={malformed})")
    return dict(manual_groups), public_items, other_count, malformed, total_lines


# ==============================================================================
# Selection
# ==============================================================================

def _greedy_sample(items: list, token_budget: int, rng: random.Random) -> tuple:
    """
    Shuffle items with rng, take greedily until token_budget is met.

    Returns (selected_line_set, actual_tokens, actual_docs).
    If token_budget >= total_available: take all.
    """
    total_available = sum(n for _, n in items)
    if token_budget >= total_available:
        return {ln for ln, _ in items}, total_available, len(items)

    shuffled = list(items)
    rng.shuffle(shuffled)

    selected = set()
    acc_tok  = 0
    acc_doc  = 0
    for ln, ntok in shuffled:
        if acc_tok >= token_budget:
            break
        selected.add(ln)
        acc_tok += ntok
        acc_doc += 1
    return selected, acc_tok, acc_doc


def select_manual(
    manual_groups: dict,
    manual_target,           # int or None
    rng: random.Random,
) -> tuple:
    """
    Select Manual documents.

    MANUAL_TARGET = None → take ALL manual (Nepali mode).
    MANUAL_TARGET = int  → proportionally sample each sub-group (Hindi mode).
                           If a sub-group is exhausted, its deficit is
                           redistributed to the remaining sub-groups.

    Returns
    -------
    selected_lines   : set of line numbers
    per_group_stats  : dict  sub-group label -> info dict
    total_toks_sel   : int
    total_docs_sel   : int
    """
    # Pre-compute totals per sub-group
    group_docs = {k: len(v)               for k, v in manual_groups.items()}
    group_toks = {k: sum(n for _, n in v) for k, v in manual_groups.items()}
    grand_toks = sum(group_toks.values())
    grand_docs = sum(group_docs.values())

    # ── Mode: take all ────────────────────────────────────────────────────────
    if manual_target is None or manual_target >= grand_toks:
        selected = {ln for items in manual_groups.values() for ln, _ in items}
        per_group = {
            k: {
                "docs_available":  group_docs[k],
                "toks_available":  group_toks[k],
                "toks_target":     group_toks[k],
                "docs_selected":   group_docs[k],
                "toks_selected":   group_toks[k],
                "mode": "take_all",
            }
            for k in manual_groups
        }
        return selected, per_group, grand_toks, grand_docs

    # ── Mode: proportional sampling ───────────────────────────────────────────
    # Proportional targets (integer, rounded down; deficit assigned to largest)
    group_targets = {k: int(manual_target * group_toks[k] / grand_toks)
                     for k in manual_groups}
    rounding_diff = manual_target - sum(group_targets.values())
    if rounding_diff and manual_groups:
        largest = max(group_targets, key=group_targets.get)
        group_targets[largest] += rounding_diff

    selected  = set()
    per_group = {}
    total_toks_sel = 0
    total_docs_sel = 0
    pending_redist = 0   # tokens deficit from saturated groups

    for key, items in manual_groups.items():
        tgt  = group_targets[key]
        avail = group_toks[key]

        if tgt >= avail:
            # Saturated: take all, record deficit for redistribution
            sel_set = {ln for ln, _ in items}
            sel_tok = avail
            sel_doc = len(items)
            pending_redist += (tgt - avail)
            mode = "take_all"
        else:
            sel_set, sel_tok, sel_doc = _greedy_sample(items, tgt, rng)
            mode = "sampled"

        selected       |= sel_set
        total_toks_sel += sel_tok
        total_docs_sel += sel_doc
        per_group[key] = {
            "docs_available": group_docs[key],
            "toks_available": avail,
            "toks_target":    tgt,
            "docs_selected":  sel_doc,
            "toks_selected":  sel_tok,
            "mode":           mode,
        }

    # ── Redistribute deficit from saturated groups ────────────────────────────
    if pending_redist > 0:
        for key, items in manual_groups.items():
            if per_group[key]["mode"] == "take_all" or pending_redist <= 0:
                continue
            # Pool of not-yet-selected docs from this sub-group
            pool = [(ln, ntok) for ln, ntok in items if ln not in selected]
            if not pool:
                continue
            rng.shuffle(pool)
            for ln, ntok in pool:
                if pending_redist <= 0:
                    break
                selected.add(ln)
                per_group[key]["docs_selected"] += 1
                per_group[key]["toks_selected"] += ntok
                total_toks_sel += ntok
                total_docs_sel += 1
                pending_redist -= ntok

    return selected, per_group, total_toks_sel, total_docs_sel


def select_public(
    public_items: list,
    public_target: int,
    rng: random.Random,
) -> tuple:
    """
    Randomly sample Public (Sangraha) documents to reach public_target tokens.

    Returns (selected_lines, actual_tokens, actual_docs).
    """
    sel_set, sel_tok, sel_doc = _greedy_sample(public_items, public_target, rng)
    return sel_set, sel_tok, sel_doc


# ==============================================================================
# Vocabulary frequency tracker
# ==============================================================================

class VocabFreq:
    """Fixed-size token frequency array. O(vocab_size) memory (~40 KB for 10K)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.freq = [0] * vocab_size

    def update(self, token_ids: list):
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                self.freq[tid] += 1

    @property
    def total_occurrences(self) -> int:
        return sum(self.freq)

    @property
    def unique_observed(self) -> int:
        return sum(1 for f in self.freq if f > 0)

    @property
    def zero_count(self) -> int:
        return self.vocab_size - self.unique_observed

    def coverage_pct(self) -> float:
        return 100.0 * self.unique_observed / self.vocab_size if self.vocab_size else 0.0

    def exactly_once(self) -> int:
        return sum(1 for f in self.freq if f == 1)

    def fewer_than(self, threshold: int) -> int:
        return sum(1 for f in self.freq if 0 < f < threshold)


# ==============================================================================
# Pass 2: accumulate vocabulary stats over selected records
# ==============================================================================

def pass2_vocab_stats(
    path:           str,
    selected_lines: set,
    vocab_size:     int,
) -> tuple:
    """
    Second streaming pass: read selected records only, accumulate stats.

    Returns
    -------
    vf        : VocabFreq
    sub_stats : dict  subgroup_label -> {"docs": int, "tokens": int}
    """
    vf        = VocabFreq(vocab_size)
    sub_stats = collections.defaultdict(lambda: {"docs": 0, "tokens": 0})
    seen      = 0

    t0 = time.time()
    print(f"\n[Pass 2] Computing vocab stats "
          f"({len(selected_lines):,} selected docs) ...")

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if line_no not in selected_lines:
                continue                     # ← skip: no json.loads needed
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            tokens = rec.get("tokens", [])
            src    = rec.get("source", "")
            proj   = rec.get("project", "")
            key    = subgroup_label(src, proj)

            vf.update(tokens)
            sub_stats[key]["docs"]   += 1
            sub_stats[key]["tokens"] += len(tokens)
            seen += 1

            if seen % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  [Pass 2] {seen:>10,} selected docs  "
                      f"tokens={vf.total_occurrences:>13,}  "
                      f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  [Pass 2] Done. {seen:,} docs in {elapsed:.1f}s  "
          f"total tokens accumulated={vf.total_occurrences:,}")
    return vf, dict(sub_stats)


# ==============================================================================
# Report
# ==============================================================================

def print_report(
    language:          str,
    train_jsonl:       str,
    tokenizer_path:    str,
    vocab_size:        int,
    total_target:      int,
    manual_target,
    random_seed:       int,
    # Index summary
    avail_manual_docs: int,
    avail_manual_toks: int,
    avail_public_docs: int,
    avail_public_toks: int,
    other_count:       int,
    # Selection results
    manual_per_group:  dict,
    manual_toks_sel:   int,
    manual_docs_sel:   int,
    public_toks_sel:   int,
    public_docs_sel:   int,
    public_target:     int,
    # Pass 2
    vf:        VocabFreq,
    sub_stats: dict,
):
    total_toks_sel = manual_toks_sel + public_toks_sel
    total_docs_sel = manual_docs_sel + public_docs_sel
    pct_m = 100.0 * manual_toks_sel / total_toks_sel if total_toks_sel else 0
    pct_p = 100.0 * public_toks_sel / total_toks_sel if total_toks_sel else 0

    W = 72
    print()
    print("=" * W)
    print(f"  PHASE 2 — ~500M TOKEN SELECTION REPORT")
    print(f"  Language    : {language.upper()}")
    print(f"  Input       : {train_jsonl}")
    print(f"  Tokenizer   : {tokenizer_path}")
    print(f"  Vocab size  : {vocab_size:,}")
    print(f"  Random seed : {random_seed}")
    print("=" * W)

    # ── Available data ────────────────────────────────────────────────────────
    print(f"\n  AVAILABLE TRAINING DATA")
    print(f"  {'─'*60}")
    print(f"  {'Category':<22}  {'Docs':>10}  {'Tokens':>14}")
    print(f"  {'─'*50}")
    print(f"  {'Manual':<22}  {avail_manual_docs:>10,}  {avail_manual_toks:>14,}")
    print(f"  {'Public/Downloaded':<22}  {avail_public_docs:>10,}  {avail_public_toks:>14,}")
    if other_count:
        print(f"  {'Other (not selected)':<22}  {other_count:>10,}  {'N/A':>14}")

    # ── Targets ────────────────────────────────────────────────────────────────
    print(f"\n  TOKEN TARGETS")
    print(f"  {'─'*60}")
    print(f"  Total target   : {total_target:>14,}")
    if manual_target is None:
        print(f"  Manual target  : {'ALL available':>14}  ({avail_manual_toks:,} tokens)")
    else:
        print(f"  Manual target  : {manual_target:>14,}")
    print(f"  Public target  : {public_target:>14,}  (= total − actual manual selected)")

    # ── Selection result ───────────────────────────────────────────────────────
    print(f"\n  SELECTION RESULT")
    print(f"  {'─'*60}")
    print(f"  {'Category':<22}  {'Docs':>10}  {'Tokens':>14}  {'% tokens':>9}")
    print(f"  {'─'*60}")
    print(f"  {'Manual':<22}  {manual_docs_sel:>10,}  {manual_toks_sel:>14,}  {pct_m:>8.2f}%")
    print(f"  {'Public/Downloaded':<22}  {public_docs_sel:>10,}  {public_toks_sel:>14,}  {pct_p:>8.2f}%")
    print(f"  {'─'*60}")
    print(f"  {'TOTAL':<22}  {total_docs_sel:>10,}  {total_toks_sel:>14,}  {'100.00%':>9}")

    # ── Manual sub-group breakdown ─────────────────────────────────────────────
    print(f"\n  MANUAL SUB-SOURCE BREAKDOWN")
    print(f"  {'─'*84}")
    hdr = (f"  {'Sub-group':<26}  {'Avail docs':>10}  {'Avail toks':>12}  "
           f"{'Sel docs':>9}  {'Sel toks':>12}  {'Mode':<10}")
    print(hdr)
    print(f"  {'─'*84}")
    for key in sorted(manual_per_group.keys()):
        g = manual_per_group[key]
        print(f"  {key:<26}  {g['docs_available']:>10,}  {g['toks_available']:>12,}  "
              f"{g['docs_selected']:>9,}  {g['toks_selected']:>12,}  {g['mode']:<10}")

    # ── Public sub-group breakdown (from Pass 2) ───────────────────────────────
    print(f"\n  PUBLIC SUB-SOURCE BREAKDOWN  (from Pass 2 actual counts)")
    print(f"  {'─'*60}")
    pub_sub = {k: v for k, v in sub_stats.items()
               if "sangraha" in k or classify_source(k.split("/")[0]) == CAT_PUBLIC}
    # Fallback: show all sub_stats not in manual
    if not pub_sub:
        pub_sub = {k: v for k, v in sub_stats.items()
                   if k not in manual_per_group}
    for key, v in sorted(pub_sub.items()):
        avg = v["tokens"] / v["docs"] if v["docs"] else 0
        print(f"  {key:<26}  docs={v['docs']:>10,}  tokens={v['tokens']:>13,}  avg/doc={avg:>7.1f}")

    # ── Target vs actual ──────────────────────────────────────────────────────
    print(f"\n  TARGET VS ACTUAL")
    print(f"  {'─'*60}")
    delta_t = total_toks_sel - total_target
    sign_t  = "+" if delta_t >= 0 else ""
    print(f"  Total  — target: {total_target:>14,}   actual: {total_toks_sel:>14,}   delta: {sign_t}{delta_t:,}")
    if manual_target is not None:
        delta_m = manual_toks_sel - manual_target
        sign_m  = "+" if delta_m >= 0 else ""
        print(f"  Manual — target: {manual_target:>14,}   actual: {manual_toks_sel:>14,}   delta: {sign_m}{delta_m:,}")
    delta_p = public_toks_sel - public_target
    sign_p  = "+" if delta_p >= 0 else ""
    print(f"  Public — target: {public_target:>14,}   actual: {public_toks_sel:>14,}   delta: {sign_p}{delta_p:,}")

    # ── Vocabulary coverage ────────────────────────────────────────────────────
    print(f"\n  VOCABULARY COVERAGE  (over selected ~{total_toks_sel/1e6:.0f}M tokens)")
    print(f"  {'─'*60}")
    print(f"  Vocab size                  : {vf.vocab_size:>12,}")
    print(f"  Total token occurrences     : {vf.total_occurrences:>12,}")
    print(f"  Unique token IDs observed   : {vf.unique_observed:>12,}")
    print(f"  Coverage                    : {vf.coverage_pct():>11.2f}%")
    print(f"  Never-observed tokens       : {vf.zero_count:>12,}")
    print(f"  Tokens occurring exactly = 1: {vf.exactly_once():>12,}")
    print(f"  Tokens occurring < 10       : {vf.fewer_than(10):>12,}")
    print(f"  Tokens occurring < 100      : {vf.fewer_than(100):>12,}")

    print(f"\n  NOTE: No output files have been written.")
    print(f"  Review the vocabulary coverage above, then proceed to the final")
    print(f"  corpus-writing step (WRITE_OUTPUT mode — to be added separately).")
    print()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    t_start = time.time()

    print("=" * 72)
    print(f"  final_tokens_selector.py  [{LANGUAGE.upper()}]  — PRINT ONLY, no files written")
    print(f"  Input   : {TRAIN_JSONL}")
    print(f"  Target  : {TOTAL_TARGET:,} total tokens (~{TOTAL_TARGET/1e6:.0f}M)")
    if MANUAL_TARGET is None:
        print(f"  Manual  : ALL available")
    else:
        print(f"  Manual  : {MANUAL_TARGET:,} (~{MANUAL_TARGET/1e6:.0f}M tokens)")
    print(f"  Seed    : {RANDOM_SEED}")
    print("=" * 72)

    # ── Pre-flight checks ────────────────────────────────────────────────────
    for label, path in [("Input JSONL", TRAIN_JSONL),
                         ("Tokenizer",   TOKENIZER_PATH)]:
        if not os.path.isfile(path):
            sys.exit(f"\n[ERROR] {label} not found: {path}\n"
                     "  Run from the project root directory.\n")

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("\n[ERROR] Install the tokenizers package:  pip install tokenizers\n")

    tokenizer  = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"\nTokenizer loaded: vocab_size={vocab_size:,}")

    rng = random.Random(RANDOM_SEED)

    # ── Pass 1: build index ──────────────────────────────────────────────────
    manual_groups, public_items, other_count, malformed, total_lines = \
        pass1_build_index(TRAIN_JSONL)

    avail_manual_docs = sum(len(v)               for v in manual_groups.values())
    avail_manual_toks = sum(n for v in manual_groups.values() for _, n in v)
    avail_public_docs = len(public_items)
    avail_public_toks = sum(n for _, n in public_items)

    print(f"\n  Available data summary:")
    print(f"    Manual : {avail_manual_docs:>9,} docs  {avail_manual_toks:>14,} tokens")
    for key in sorted(manual_groups):
        d = len(manual_groups[key])
        t = sum(n for _, n in manual_groups[key])
        print(f"      {key:<26}: docs={d:>8,}  tokens={t:>12,}")
    print(f"    Public : {avail_public_docs:>9,} docs  {avail_public_toks:>14,} tokens")
    if other_count:
        print(f"    Other  : {other_count:>9,} docs  (not selected)")
    if malformed:
        print(f"    Malformed JSON records: {malformed}")

    # ── Selection ────────────────────────────────────────────────────────────
    print(f"\n  Performing selection (seed={RANDOM_SEED}) ...")

    sel_manual_lines, manual_per_group, manual_toks_sel, manual_docs_sel = \
        select_manual(manual_groups, MANUAL_TARGET, rng)

    public_target = TOTAL_TARGET - manual_toks_sel
    print(f"    Manual selected : {manual_docs_sel:,} docs  {manual_toks_sel:,} tokens")
    print(f"    Public target   : {public_target:,} tokens "
          f"(= {TOTAL_TARGET:,} − {manual_toks_sel:,})")

    sel_public_lines, public_toks_sel, public_docs_sel = \
        select_public(public_items, public_target, rng)
    print(f"    Public selected : {public_docs_sel:,} docs  {public_toks_sel:,} tokens")

    all_selected = sel_manual_lines | sel_public_lines
    total_toks_selected = manual_toks_sel + public_toks_sel
    print(f"    TOTAL selected  : {len(all_selected):,} docs  {total_toks_selected:,} tokens "
          f"(~{total_toks_selected/1e6:.1f}M)")

    # ── Pass 2: vocab stats ──────────────────────────────────────────────────
    vf, sub_stats = pass2_vocab_stats(TRAIN_JSONL, all_selected, vocab_size)

    # ── Final report ─────────────────────────────────────────────────────────
    print_report(
        language          = LANGUAGE,
        train_jsonl       = TRAIN_JSONL,
        tokenizer_path    = TOKENIZER_PATH,
        vocab_size        = vocab_size,
        total_target      = TOTAL_TARGET,
        manual_target     = MANUAL_TARGET,
        random_seed       = RANDOM_SEED,
        avail_manual_docs = avail_manual_docs,
        avail_manual_toks = avail_manual_toks,
        avail_public_docs = avail_public_docs,
        avail_public_toks = avail_public_toks,
        other_count       = other_count,
        manual_per_group  = manual_per_group,
        manual_toks_sel   = manual_toks_sel,
        manual_docs_sel   = manual_docs_sel,
        public_toks_sel   = public_toks_sel,
        public_docs_sel   = public_docs_sel,
        public_target     = public_target,
        vf                = vf,
        sub_stats         = sub_stats,
    )

    # ── Pass 3: write corpus ──────────────────────────────────────────────────
    output_path, docs_written = pass3_write_corpus(
        src_path       = TRAIN_JSONL,
        selected_lines = all_selected,
        output_dir     = OUTPUT_DIR,
    )

    # ── Write summary ─────────────────────────────────────────────────────────
    total_toks_written = manual_toks_sel + public_toks_sel
    total_docs_written = manual_docs_sel + public_docs_sel
    print(f"\n  CORPUS WRITE SUMMARY")
    print(f"  {'─'*60}")
    print(f"  Output path      : {output_path}")
    print(f"  Manual docs/toks : {manual_docs_sel:,} / {manual_toks_sel:,}")
    print(f"  Public docs/toks : {public_docs_sel:,} / {public_toks_sel:,}")
    print(f"  Total  docs/toks : {total_docs_written:,} / {total_toks_written:,}")
    print(f"  Target tokens    : {TOTAL_TARGET:,}  (actual: {total_toks_written:,}  "
          f"delta: {total_toks_written - TOTAL_TARGET:+,})")
    print(f"  Records modified : NO — original JSONL lines copied verbatim")

    # ── Verify output ─────────────────────────────────────────────────────────
    verify_output_corpus(
        output_path     = output_path,
        expected_docs   = total_docs_written,
        expected_tokens = total_toks_written,
    )

    t_total = time.time() - t_start
    print(f"\n  Total elapsed: {t_total:.1f}s  ({t_total/60:.1f} min)")
    print()


# ==============================================================================
# Pass 3: Write selected corpus
# ==============================================================================

def pass3_write_corpus(
    src_path:       str,
    selected_lines: set,
    output_dir:     str,
) -> tuple:
    """
    Third streaming pass: copy selected JSONL records to the output file.

    Records are written by passing `raw` directly to `fout.write()` — the
    exact bytes read from the input file, including original whitespace and
    line endings, with no json.loads, json.dumps, stripping, reordering,
    or any other modification.

    Returns
    -------
    output_path   : str   path of the written file
    docs_written  : int
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "train.jsonl")

    docs_written = 0
    t0 = time.time()
    print(f"\n[Pass 3] Writing selected corpus ...")
    print(f"  Output : {output_path}")

    with open(src_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, start=1):
            if line_no not in selected_lines:
                continue                     # ← skip: not selected
            if not raw.strip():
                continue
            fout.write(raw)              # ← raw copied directly, unchanged
            docs_written += 1
            if docs_written % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                print(f"  [Pass 3] {docs_written:>10,} docs written  "
                      f"elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [Pass 3] Done. {docs_written:,} docs written in {elapsed:.1f}s  "
          f"({size_mb:.1f} MB)")
    return output_path, docs_written


def verify_output_corpus(
    output_path:    str,
    expected_docs:  int,
    expected_tokens: int,
) -> bool:
    """
    Read the written output file and verify doc count and token count.

    Returns True if both counts match exactly, False otherwise.
    """
    print(f"\n[Verify] Reading output for verification ...")
    t0 = time.time()
    actual_docs   = 0
    actual_tokens = 0

    with open(output_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                print(f"  [Verify] WARNING: malformed line in output!")
                continue
            actual_docs   += 1
            actual_tokens += len(rec.get("tokens", []))

    elapsed = time.time() - t0
    ok_docs = actual_docs   == expected_docs
    ok_toks = actual_tokens == expected_tokens

    print(f"  [Verify] Done in {elapsed:.1f}s")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<22}  {'Expected':>14}  {'Actual':>14}  {'Match':>6}")
    print(f"  {'─'*60}")
    print(f"  {'Documents':<22}  {expected_docs:>14,}  {actual_docs:>14,}  "
          f"{'  OK' if ok_docs else 'FAIL':>6}")
    print(f"  {'Tokens':<22}  {expected_tokens:>14,}  {actual_tokens:>14,}  "
          f"{'  OK' if ok_toks else 'FAIL':>6}")
    print(f"  {'─'*60}")

    if ok_docs and ok_toks:
        print(f"  Verification PASSED — records copied without modification.")
    else:
        print(f"  Verification FAILED — see mismatches above.")
    return ok_docs and ok_toks


# ==============================================================================
if __name__ == "__main__":
    main()
