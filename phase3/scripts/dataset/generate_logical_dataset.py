#!/usr/bin/env python3
"""
generate_logical_dataset.py — CLI entry point for Phase 3.1 logical generation.

Usage
-----
    python -m phase3.scripts.dataset.generate_logical_dataset
    python -m phase3.scripts.dataset.generate_logical_dataset --config phase3/configs/logical_generator.yaml
    python -m phase3.scripts.dataset.generate_logical_dataset --config phase3/configs/logical_generator.yaml \
        --output-dir phase3/outputs --seed 42

Produces
--------
    <output_dir>/logical_dataset.jsonl        one JSON record per line
    <output_dir>/logical_dataset_stats.json   generation statistics

Must be run from the repo root (individual-project-charan-1230/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Add repo root to path so phase3.* imports work
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from phase3.logical_generator.config import GeneratorConfig
from phase3.logical_generator.generator import LogicalGenerator
from phase3.logical_generator.splits import SplitManager
from phase3.logical_generator.structures import LogicalRecord


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the Phase 3.1 logical reasoning dataset."
    )
    p.add_argument(
        "--config",
        default="phase3/configs/logical_generator.yaml",
        help="Path to logical_generator.yaml (default: phase3/configs/logical_generator.yaml)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(
    records: List[LogicalRecord],
    gen_stats: Dict,
    elapsed: float,
) -> Dict:
    """Compute comprehensive statistics over the generated records."""
    total = len(records)

    # Per-family counts
    by_family: Dict[str, int] = defaultdict(int)
    by_difficulty: Dict[str, int] = defaultdict(int)
    by_attribute: Dict[str, int] = defaultdict(int)
    by_representation: Dict[str, int] = defaultdict(int)
    by_split: Dict[str, int] = defaultdict(int)
    by_pattern: Dict[str, int] = defaultdict(int)

    relation_counts: Dict[str, int] = defaultdict(int)
    gt_relation_counts: Dict[str, int] = defaultdict(int)
    entity_counts: List[int] = []
    path_lengths: List[int] = []
    fact_counts: List[int] = []

    for r in records:
        by_family[r.task_family] += 1
        by_difficulty[r.difficulty] += 1
        by_attribute[r.attribute] += 1
        by_representation[r.representation] += 1
        by_split[r.split] += 1
        by_pattern[r.relation_pattern_id] += 1
        entity_counts.append(len(r.entities))
        path_lengths.append(len(r.reasoning_path))
        fact_counts.append(len(r.facts))
        for f in r.facts:
            relation_counts[f.relation] += 1
        if r.ground_truth.relation is not None:
            gt_relation_counts[r.ground_truth.relation] += 1

    def avg(lst: List[int]) -> float:
        return round(sum(lst) / len(lst), 3) if lst else 0.0

    # Per-family breakdown by difficulty / attribute / representation
    family_breakdown: Dict[str, Dict] = {}
    for r in records:
        tf = r.task_family
        if tf not in family_breakdown:
            family_breakdown[tf] = {
                "difficulty": defaultdict(int),
                "attribute":  defaultdict(int),
                "representation": defaultdict(int),
            }
        family_breakdown[tf]["difficulty"][r.difficulty]     += 1
        family_breakdown[tf]["attribute"][r.attribute]       += 1
        family_breakdown[tf]["representation"][r.representation] += 1

    return {
        "total_examples":          total,
        "generation_time_seconds": round(elapsed, 2),
        "by_task_family":          dict(by_family),
        "by_difficulty":           dict(by_difficulty),
        "by_attribute":            dict(by_attribute),
        "by_representation":       dict(by_representation),
        "by_split":                dict(by_split),
        "by_relation_pattern":     dict(by_pattern),
        "relation_distribution":   dict(relation_counts),
        "ground_truth_relation_distribution": dict(gt_relation_counts),
        "avg_entities_per_example": avg(entity_counts),
        "avg_reasoning_path_length": avg(path_lengths),
        "avg_facts_per_example":   avg(fact_counts),
        "family_breakdown":        {
            k: {sk: dict(sv) for sk, sv in v.items()}
            for k, v in family_breakdown.items()
        },
        "generation_internals": {
            "rejections_per_family": gen_stats.get("rejections_per_family", {}),
            "attempts_per_family":   gen_stats.get("attempts_per_family", {}),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(stats: Dict) -> None:
    sep = "─" * 60
    print(f"\n{'='*60}")
    print("  PHASE 3.1 — LOGICAL DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Total examples  : {stats['total_examples']:,}")
    print(f"  Generation time : {stats['generation_time_seconds']} s")

    print(f"\n{sep}")
    print("  Examples per task family:")
    for tf, cnt in stats["by_task_family"].items():
        print(f"    {tf:<30} {cnt:>5}")

    print(f"\n{sep}")
    print("  Examples per difficulty:")
    for diff, cnt in stats["by_difficulty"].items():
        print(f"    {diff:<12} {cnt:>5}")

    print(f"\n{sep}")
    print("  Examples per attribute:")
    for attr, cnt in stats["by_attribute"].items():
        print(f"    {attr:<12} {cnt:>5}")

    print(f"\n{sep}")
    print("  Examples per representation:")
    for rep, cnt in stats["by_representation"].items():
        print(f"    {rep:<12} {cnt:>5}")

    print(f"\n{sep}")
    print("  Split distribution:")
    for split, cnt in stats["by_split"].items():
        print(f"    {split:<8} {cnt:>5}")

    print(f"\n{sep}")
    print("  Relation distribution (across all facts):")
    for rel, cnt in stats["relation_distribution"].items():
        print(f"    {rel:<20} {cnt:>6}")

    print(f"\n{sep}")
    print("  Ground-truth relation distribution (across all answers):")
    for rel, cnt in stats["ground_truth_relation_distribution"].items():
        print(f"    {rel:<20} {cnt:>6}")

    print(f"\n{sep}")
    print(f"  Avg entities per example    : {stats['avg_entities_per_example']}")
    print(f"  Avg reasoning path length   : {stats['avg_reasoning_path_length']}")
    print(f"  Avg facts per example       : {stats['avg_facts_per_example']}")

    print(f"\n{sep}")
    print("  Rejections per family:")
    for tf, cnt in stats["generation_internals"]["rejections_per_family"].items():
        print(f"    {tf:<30} {cnt:>5}")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Load config ─────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    cfg = GeneratorConfig.from_yaml(config_path)

    # CLI overrides
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir is not None:
        cfg.output_path = args.output_dir

    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path  = output_dir / "logical_dataset.jsonl"
    stats_path  = output_dir / "logical_dataset_stats.json"

    if not args.quiet:
        print(f"Config  : {config_path}")
        print(f"Seed    : {cfg.seed}")
        print(f"Output  : {output_dir}")
        print(f"Generating {cfg.total_examples:,} examples …", flush=True)

    # ── Generate ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    generator = LogicalGenerator(cfg)
    records, gen_stats = generator.generate()
    elapsed = time.perf_counter() - t0

    if not args.quiet:
        print(f"Generated {len(records):,} records in {elapsed:.1f} s", flush=True)

    # ── Assign splits ───────────────────────────────────────────────────────
    split_mgr = SplitManager(seed=cfg.seed)
    records   = split_mgr.assign_splits(records)

    # ── Write JSONL ─────────────────────────────────────────────────────────
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json() + "\n")

    if not args.quiet:
        print(f"Written : {jsonl_path}  ({jsonl_path.stat().st_size / 1024:.1f} KB)")

    # ── Compute & write stats ────────────────────────────────────────────────
    stats = compute_stats(records, gen_stats, elapsed)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"Stats   : {stats_path}")
        print_stats(stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
