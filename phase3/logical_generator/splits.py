"""
splits.py — Stratified train / validation / test split management.

Strategy
--------
Records are grouped by (task_family, difficulty).  Within each stratum the
records are shuffled deterministically, then the first 80 % go to train,
the next 10 % to val, and the remaining 10 % to test.

Since every difficulty count in the spec is a multiple of 10, the
proportions come out exactly — no rounding issues.

Extension hooks
---------------
The class exposes stub attributes for future held-out strategies:
    held_out_entity_pools  : Set[str]
    held_out_patterns      : Set[str]
    held_out_templates     : Set[str]

These are not yet used; they are placeholders for Phase 3.2.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from .structures import LogicalRecord


class SplitManager:
    """
    Assigns "train" | "val" | "test" to every LogicalRecord in-place.

    Parameters
    ----------
    train_ratio, val_ratio, test_ratio : must sum to 1.0
    seed : base seed for intra-stratum shuffling
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio:   float = 0.1,
        test_ratio:  float = 0.1,
        seed:        int   = 42,
    ) -> None:
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
            raise ValueError("train + val + test ratios must sum to 1.0")
        self.train_ratio = train_ratio
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed

        # ── Future extension hooks ─────────────────────────────────────────
        self.held_out_entity_pools: Set[str] = set()   # e.g. entity name pools
        self.held_out_patterns:     Set[str] = set()   # e.g. {"P12", "P15"}
        self.held_out_templates:    Set[str] = set()   # template IDs (Phase 3.2)

    def assign_splits(
        self, records: List[LogicalRecord]
    ) -> List[LogicalRecord]:
        """
        Assign split labels to *records* in-place and return them sorted by ID.

        Stratification is by (task_family, difficulty).  Within each stratum,
        records are shuffled with a deterministic seed before assignment so
        that every difficulty level is represented proportionally in every split.
        """
        # Group by stratum
        strata: Dict[Tuple[str, str], List[LogicalRecord]] = defaultdict(list)
        for record in records:
            key = (record.task_family, record.difficulty)
            strata[key].append(record)

        for key, group in strata.items():
            # Deterministic per-stratum shuffle (seed derived from stratum key)
            key_str = f"{key[0]}_{key[1]}".encode("utf-8")
            key_hash = int(hashlib.md5(key_str).hexdigest(), 16)
            stratum_seed = self.seed ^ key_hash
            rng = random.Random(stratum_seed)
            rng.shuffle(group)

            n = len(group)
            n_train = round(n * self.train_ratio)
            n_val   = round(n * self.val_ratio)
            n_test  = n - n_train - n_val   # absorbs rounding remainder

            for i, record in enumerate(group):
                if i < n_train:
                    record.split = "train"
                elif i < n_train + n_val:
                    record.split = "val"
                else:
                    record.split = "test"

        # Re-sort by global ID for deterministic output ordering
        records.sort(key=lambda r: r.id)
        return records

    # ── Summary helpers ────────────────────────────────────────────────────

    def split_counts(
        self, records: List[LogicalRecord]
    ) -> Dict[str, int]:
        """Return {split_name: count} after assignment."""
        counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for r in records:
            if r.split in counts:
                counts[r.split] += 1
        return counts
