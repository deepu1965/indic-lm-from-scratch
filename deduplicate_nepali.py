#!/usr/bin/env python3
"""
deduplicate_nepali.py — Two-stage JSONL corpus deduplication for the Nepali corpus.

This file is the Nepali-specific variant of deduplicate.py.
All algorithms are identical; the ONLY behavioural differences are:

  1. Source priority (lower = higher priority, wins deduplication):
       wikimedia       → 0   (highest)
       manual          → 1
       sangraha verified → 2  (lowest)

  2. Statistics report three separate source categories:
       Wikimedia / Manual / Sangraha
     instead of collapsing everything into manual/public.

Do NOT modify deduplicate.py — the Hindi pipeline depends on its current
two-tier (wiki vs. public) priority behaviour.

Stage 1 — Exact dedup:
    Hash each document's text (SHA-256). Keep the first occurrence;
    prefer wikimedia > manual > sangraha verified.

Stage 2 — Near-duplicate dedup (MinHash + LSH):
    Build 3-word shingles → MinHash signature → LSH band buckets.
    All intermediate state lives in SQLite; never loads the corpus into RAM.
    Source priority ordering is respected: wikimedia docs are processed first,
    then manual, then sangraha verified.

Run:
    python3 deduplicate_nepali.py --config clean_dedup_config_nepali.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any, Optional

import yaml
from datasketch import MinHash


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    def __init__(self, raw: dict):
        self.raw = raw

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, *keys, default=None):
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


# ---------------------------------------------------------------------------
# Source priority helper  (Nepali-specific: three tiers)
# ---------------------------------------------------------------------------

def source_priority(source: str) -> int:
    """Lower value = higher priority.

    wikimedia       → 0  (highest — always retained over manual/sangraha)
    manual          → 1
    sangraha verified → 2  (lowest — removed when a duplicate exists at 0 or 1)
    """
    s = (source or "").strip().lower()
    if s.startswith("wiki"):
        return 0
    if s == "manual":
        return 1
    # "sangraha verified" and any unknown source fall here
    return 2


def source_category(source: str) -> str:
    """Map a raw source string to one of three display categories."""
    prio = source_priority(source)
    if prio == 0:
        return "wikimedia"
    if prio == 1:
        return "manual"
    return "sangraha"


# ---------------------------------------------------------------------------
# JSONL streaming
# ---------------------------------------------------------------------------

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream records from a JSONL file one at a time."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {lineno}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Statistics tracker  (Nepali-specific: three source categories)
# ---------------------------------------------------------------------------

class Stats:
    """Tracks doc/word/char counts before and after each stage.

    Reports three categories — wikimedia / manual / sangraha — rather than
    the two-category (manual/public) scheme used in deduplicate.py.
    """

    # Ordered list of categories used in print_stage
    CATEGORIES = ("wikimedia", "manual", "sangraha")

    def __init__(self):
        # {stage: {category: {docs, words, chars}}}
        self._d: Dict[str, Dict[str, Dict[str, int]]] = {}

    def _bucket(self, stage: str, cat: str) -> Dict[str, int]:
        return self._d.setdefault(stage, {}).setdefault(cat, {"docs": 0, "words": 0, "chars": 0})

    def add(self, stage: str, source: str, text: str) -> None:
        cat = source_category(source)
        b = self._bucket(stage, cat)
        b["docs"]  += 1
        b["words"] += len(text.split())
        b["chars"] += len(text)

    def _totals(self, stage: str) -> Dict[str, int]:
        t = {"docs": 0, "words": 0, "chars": 0}
        for b in self._d.get(stage, {}).values():
            for k in t:
                t[k] += b[k]
        return t

    def print_stage(self, before_stage: str, after_stage: str, label: str) -> None:
        bf = self._totals(before_stage)
        af = self._totals(after_stage)
        print(f"\n{'─'*60}")
        print(f"  Statistics after {label}")
        print(f"{'─'*60}")
        for cat in (*self.CATEGORIES, "overall"):
            if cat == "overall":
                b = bf
                a = af
            else:
                b = self._d.get(before_stage, {}).get(cat, {"docs":0,"words":0,"chars":0})
                a = self._d.get(after_stage,  {}).get(cat, {"docs":0,"words":0,"chars":0})
            removed_docs  = b["docs"]  - a["docs"]
            removed_words = b["words"] - a["words"]
            removed_chars = b["chars"] - a["chars"]
            pct_d = 100*removed_docs  / b["docs"]   if b["docs"]   else 0
            pct_w = 100*removed_words / b["words"]  if b["words"]  else 0
            pct_c = 100*removed_chars / b["chars"]  if b["chars"]  else 0
            tag = f"[{cat.upper()}]"
            print(f"  {tag:12s}  kept: docs={a['docs']:,}  words={a['words']:,}  chars={a['chars']:,}")
            print(f"  {' ':12s}  removed: docs={removed_docs:,} ({pct_d:.2f}%)  "
                  f"words={removed_words:,} ({pct_w:.2f}%)  chars={removed_chars:,} ({pct_c:.2f}%)")


# ---------------------------------------------------------------------------
# SQLite-backed store for exact dedup and MinHash/LSH
# ---------------------------------------------------------------------------

class DedupeDB:
    """All disk-backed state for both dedup stages lives here."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-65536")   # ~64 MB page cache
        self._init_tables()

    def _init_tables(self) -> None:
        c = self.conn
        # Exact dedup: stores text_hash → (doc_id, source_priority)
        c.execute("""CREATE TABLE IF NOT EXISTS exact_seen (
            text_hash TEXT PRIMARY KEY,
            doc_id    TEXT,
            priority  INTEGER
        )""")
        # Near-dedup: MinHash signature per doc (stored as JSON array of k ints)
        c.execute("""CREATE TABLE IF NOT EXISTS signatures (
            doc_id    TEXT PRIMARY KEY,
            source    TEXT,
            sig_blob  BLOB
        )""")
        # LSH buckets: (band_id, bucket_hash) → list of doc_ids
        c.execute("""CREATE TABLE IF NOT EXISTS lsh_buckets (
            band_id     INTEGER,
            bucket_hash TEXT,
            doc_id      TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_lsh ON lsh_buckets(band_id, bucket_hash)")
        # Near-dedup removal set: doc_ids marked for removal, disk-backed so
        # the Python process never holds a growing set in RAM.
        c.execute("""CREATE TABLE IF NOT EXISTS near_removed (
            doc_id TEXT PRIMARY KEY
        )""")
        c.commit()

    # --- Exact dedup helpers ---

    def exact_seen(self, text_hash: str) -> Optional[Tuple[str, int]]:
        """Return (doc_id, priority) if text_hash already seen, else None."""
        row = self.conn.execute(
            "SELECT doc_id, priority FROM exact_seen WHERE text_hash=?", (text_hash,)
        ).fetchone()
        return row  # (doc_id, priority) or None

    def exact_insert(self, text_hash: str, doc_id: str, priority: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO exact_seen VALUES (?,?,?)",
            (text_hash, doc_id, priority)
        )

    # --- MinHash / LSH helpers ---

    def insert_signature(self, doc_id: str, source: str, sig: List[int]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO signatures VALUES (?,?,?)",
            (doc_id, source, json.dumps(sig))
        )

    def get_signature(self, doc_id: str) -> Optional[List[int]]:
        row = self.conn.execute(
            "SELECT sig_blob FROM signatures WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def insert_lsh_band(self, band_id: int, bucket_hash: str, doc_id: str) -> None:
        self.conn.execute(
            "INSERT INTO lsh_buckets VALUES (?,?,?)", (band_id, bucket_hash, doc_id)
        )

    def get_bucket_members(self, band_id: int, bucket_hash: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT doc_id FROM lsh_buckets WHERE band_id=? AND bucket_hash=?",
            (band_id, bucket_hash)
        ).fetchall()
        return [r[0] for r in rows]

    def mark_removed(self, doc_id: str) -> None:
        """Record doc_id as a near-duplicate to discard (disk-backed, not in RAM)."""
        self.conn.execute("INSERT OR IGNORE INTO near_removed VALUES (?)", (doc_id,))

    def is_removed(self, doc_id: str) -> bool:
        """Return True if doc_id has been marked for near-dedup removal."""
        return self.conn.execute(
            "SELECT 1 FROM near_removed WHERE doc_id=?", (doc_id,)
        ).fetchone() is not None

    def removed_count(self) -> int:
        """Return the current count of near-removed documents."""
        return self.conn.execute("SELECT COUNT(*) FROM near_removed").fetchone()[0]

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# MinHash  (uses datasketch.MinHash)
# ---------------------------------------------------------------------------


def minhash_signature(shingles: List[int], num_perm: int = 100) -> List[int]:
    """Return a length-num_perm MinHash signature using datasketch.MinHash.

    Each integer shingle is packed as an unsigned 64-bit little-endian byte
    sequence and passed to MinHash.update().  The resulting hashvalues ndarray
    is converted to a plain Python list[int] so it can be JSON-serialised into
    the existing SQLite schema unchanged.

    Returns a list of ``num_perm`` max-uint32 sentinels when shingles is empty
    (consistent with the previous MinHasher behaviour).
    """
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(struct.pack("<Q", sh & 0xFFFFFFFFFFFFFFFF))
    return m.hashvalues.tolist()


def shingle_hashes(text: str, w: int) -> List[int]:
    """Return hashed w-word shingles (Lecture 3 method).

    Returns an empty list if the document has fewer than w words; no
    character-level fallback is used, matching the slide definition.
    """
    words = text.split()
    if len(words) < w:
        return []   # too short for word shingles; treat as empty set
    seen = set()
    result = []
    for i in range(len(words) - w + 1):
        gram = " ".join(words[i:i+w])
        h = int(hashlib.md5(gram.encode()).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF
        if h not in seen:
            seen.add(h)
            result.append(h)
    return result


def jaccard_from_sigs(sig_a: List[int], sig_b: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def band_bucket_hash(band_slice: List[int]) -> str:
    """Hash one band's rows into a bucket key."""
    return hashlib.md5(b"|".join(str(x).encode() for x in band_slice)).hexdigest()


# ---------------------------------------------------------------------------
# Stage 1: Exact deduplication
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_exact_dedup(
    input_path: str,
    output_path: str,
    db: DedupeDB,
    text_field: str,
    stats: Stats,
    commit_every: int = 10_000,
) -> None:
    """Pass 1 — Write only unique documents (by text hash) to output_path.

    Source priority (Nepali): wikimedia (0) > manual (1) > sangraha verified (2).
    If the same text appears in multiple sources the highest-priority record wins.
    """
    print(f"\n[EXACT DEDUP] Input:  {input_path}")
    print(f"[EXACT DEDUP] Output: {output_path}")
    t0 = time.time()

    # Two-sub-pass approach:
    # Sub-pass A: scan all records, record which hash has the best priority.
    #   For each text_hash, keep track of (best_priority, doc_id).
    # Sub-pass B: stream again, write records whose doc_id matches winner.
    #
    # This correctly handles the case where a lower-priority record appears
    # before its higher-priority duplicate in the file.

    print("  Sub-pass A: cataloguing hashes and source priorities...")
    seen = 0
    for i, rec in enumerate(iter_jsonl(input_path)):
        text = rec.get(text_field, "")
        if not text:
            continue
        th = text_hash(text)
        source = rec.get("source", "")
        prio = source_priority(source)
        doc_id = rec.get("id", th)

        stats.add("raw", source, text)

        existing = db.exact_seen(th)
        if existing is None or prio < existing[1]:
            db.exact_insert(th, doc_id, prio)

        seen += 1
        if (i + 1) % commit_every == 0:
            db.commit()
            print(f"    catalogued {i+1:,} records  elapsed={time.time()-t0:.0f}s", flush=True)

    db.commit()
    print(f"  Sub-pass A done: {seen:,} records scanned in {time.time()-t0:.0f}s")

    # Sub-pass B: write winners
    print("  Sub-pass B: writing deduplicated records...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(iter_jsonl(input_path)):
            text = rec.get(text_field, "")
            if not text:
                continue
            th = text_hash(text)
            doc_id = rec.get("id", th)
            winner = db.exact_seen(th)
            if winner and winner[0] == doc_id:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.add("after_exact", rec.get("source",""), text)
                written += 1
            if (i + 1) % commit_every == 0:
                print(f"    written {written:,} / scanned {i+1:,}  elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"  Sub-pass B done: {written:,} unique records written in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Stage 2: Near-duplicate deduplication (MinHash + LSH)
# ---------------------------------------------------------------------------

def build_signatures(
    input_path: str,
    db: DedupeDB,
    text_field: str,
    minhash_size: int,
    shingle_size: int,
    bands: int,
    commit_every: int = 5_000,
) -> None:
    """Pass 1 of near-dedup: compute MinHash signatures and insert LSH bands."""
    rows_per_band = minhash_size // bands
    t0 = time.time()
    print(f"\n[NEAR DEDUP] Building signatures (k={minhash_size}, b={bands}, r={rows_per_band})...")

    for i, rec in enumerate(iter_jsonl(input_path)):
        text = rec.get(text_field, "")
        if not text:
            continue
        doc_id = rec.get("id", text_hash(text))
        source = rec.get("source", "")

        shingles = shingle_hashes(text, shingle_size)
        if not shingles:
            # Fewer than shingle_size words — skip near-dedup entirely.
            # The document is not inserted into signatures/lsh_buckets and
            # will pass through unchanged to the final output.
            continue
        # Use datasketch.MinHash to compute the signature.
        sig = minhash_signature(shingles, num_perm=minhash_size)

        db.insert_signature(doc_id, source, sig)
        # Insert one row per band into lsh_buckets
        for band_idx in range(bands):
            band_slice = sig[band_idx * rows_per_band:(band_idx + 1) * rows_per_band]
            bh = band_bucket_hash(band_slice)
            db.insert_lsh_band(band_idx, bh, doc_id)

        if (i + 1) % commit_every == 0:
            db.commit()
            print(f"  signatures built: {i+1:,}  elapsed={time.time()-t0:.0f}s", flush=True)

    db.commit()
    print(f"  Signature building done in {time.time()-t0:.0f}s")


def run_near_dedup(
    exact_output_path: str,
    final_output_path: str,
    db: DedupeDB,
    text_field: str,
    minhash_size: int,
    shingle_size: int,
    bands: int,
    threshold: float,
    stats: Stats,
    commit_every: int = 5_000,
) -> None:
    """Full near-dedup stage: build signatures, find candidates, write output."""

    # --- Build signatures from exact-dedup output ---
    build_signatures(exact_output_path, db, text_field, minhash_size, shingle_size, bands, commit_every)

    rows_per_band = minhash_size // bands

    # --- Pass 2: determine which docs to REMOVE ---
    # Documents are processed in priority order: wikimedia first (0), then
    # manual (1), then sangraha verified (2).  When a lower-priority doc is a
    # near-duplicate of an already-seen higher-priority doc, it is marked for
    # removal.  Removed IDs live in the near_removed SQLite table — never in a
    # Python set — so memory usage is O(1) regardless of corpus size.
    print("\n[NEAR DEDUP] Identifying near-duplicate candidates...")
    t0 = time.time()

    # Total doc count for progress reporting (single cheap COUNT query).
    total_docs = db.conn.execute("SELECT COUNT(*) FROM signatures").fetchone()[0]

    # Stream signatures in priority order using a cursor (no fetchall / no
    # Python list).  CASE expression maps each source to its numeric priority.
    cursor = db.conn.execute("""
        SELECT doc_id, source FROM signatures
        ORDER BY CASE
            WHEN lower(source) LIKE 'wiki%'         THEN 0
            WHEN lower(trim(source)) = 'manual'     THEN 1
            ELSE 2
        END, doc_id
    """)

    processed = 0
    candidate_pairs = 0
    for doc_id, source in cursor:
        if db.is_removed(doc_id):
            processed += 1
            continue

        sig_a = db.get_signature(doc_id)
        if sig_a is None:
            continue

        # Collect candidate near-duplicates via shared LSH buckets only.
        candidates: set[str] = set()
        for band_idx in range(bands):
            band_slice = sig_a[band_idx * rows_per_band:(band_idx + 1) * rows_per_band]
            bh = band_bucket_hash(band_slice)
            for cid in db.get_bucket_members(band_idx, bh):
                if cid != doc_id and not db.is_removed(cid):
                    candidates.add(cid)

        for cid in candidates:
            candidate_pairs += 1
            sig_b = db.get_signature(cid)
            if sig_b is None:
                continue
            j = jaccard_from_sigs(sig_a, sig_b)
            if j >= threshold:
                # doc_id is retained (higher priority / seen first);
                # persist cid removal to SQLite, not a Python set.
                db.mark_removed(cid)

        processed += 1
        if processed % commit_every == 0:
            db.commit()  # flush near_removed inserts to disk periodically
            n_rem = db.removed_count()
            print(f"  processed {processed:,}/{total_docs:,}  "
                  f"candidate_pairs={candidate_pairs:,}  "
                  f"to_remove={n_rem:,}  "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    db.commit()
    n_removed_final = db.removed_count()
    print(f"  Candidate search done: {n_removed_final:,} documents to remove  elapsed={time.time()-t0:.0f}s")

    # --- Pass 3: write final output (streaming, no corpus in RAM) ---
    print(f"\n[NEAR DEDUP] Writing final output: {final_output_path}")
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    written = 0
    with open(final_output_path, "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(iter_jsonl(exact_output_path)):
            text = rec.get(text_field, "")
            doc_id = rec.get("id", "")
            if db.is_removed(doc_id):   # DB lookup, not Python set
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.add("after_near", rec.get("source",""), text)
            written += 1
            if (i + 1) % 50_000 == 0:
                print(f"  written {written:,}  elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"  Final output written: {written:,} records")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LMA Nepali corpus deduplication pipeline")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    corpus_root  = cfg.get("paths", "corpus_root")
    cleaned_dir  = cfg.get("deduplication", "cleaned_dir",  default="cleaned")
    output_dir   = cfg.get("deduplication", "output_dir",   default="deduplicated")
    input_file   = cfg.get("deduplication", "input_file",   default="final_cleaned.jsonl")
    text_field   = cfg.get("fields", "text_field",          default="text")

    # MinHash / LSH params
    shingle_size  = cfg.get("deduplication", "shingle_size",            default=3)
    minhash_size  = cfg.get("deduplication", "minhash_size",            default=100)
    bands         = cfg.get("deduplication", "bands",                   default=20)
    rows_per_band = minhash_size // bands
    threshold     = cfg.get("deduplication", "near_duplicate_threshold", default=0.80)

    # Validate band × rows == k
    if bands * rows_per_band != minhash_size:
        sys.exit(f"ERROR: bands ({bands}) * rows_per_band ({rows_per_band}) != minhash_size ({minhash_size}). "
                 "Ensure minhash_size is divisible by bands.")

    input_path        = os.path.join(corpus_root, cleaned_dir, input_file)
    exact_output_path = os.path.join(corpus_root, output_dir,  "exact_deduped.jsonl")
    final_output_path = os.path.join(corpus_root, output_dir,  "final_deduped.jsonl")
    db_path           = os.path.join(corpus_root, output_dir,  "dedup_state.sqlite")

    os.makedirs(os.path.join(corpus_root, output_dir), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LMA Nepali Deduplication Pipeline")
    print(f"  Config:   {args.config}")
    print(f"  Input:    {input_path}")
    print(f"  k={minhash_size}  bands={bands}  rows={rows_per_band}  "
          f"shingle={shingle_size}  threshold={threshold}")
    print(f"{'='*60}")

    db    = DedupeDB(db_path)
    stats = Stats()

    # ── Stage 1: Exact dedup ────────────────────────────────────────────────
    run_exact_dedup(input_path, exact_output_path, db, text_field, stats)
    stats.print_stage("raw", "after_exact", "EXACT DEDUPLICATION")

    # ── Stage 2: Near-duplicate dedup ───────────────────────────────────────
    run_near_dedup(
        exact_output_path, final_output_path, db, text_field,
        minhash_size, shingle_size, bands, threshold, stats,
    )
    stats.print_stage("after_exact", "after_near", "NEAR-DUPLICATE DEDUPLICATION")

    db.close()
    print(f"\n[DONE] Final corpus: {final_output_path}")
    print(f"[DONE] SQLite state: {db_path}  (safe to delete after verification)")


if __name__ == "__main__":
    main()
