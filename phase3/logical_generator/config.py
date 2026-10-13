"""
config.py — Configuration loader for the Phase 3.1 logical generator.

Reads logical_generator.yaml and exposes a typed GeneratorConfig dataclass.
All downstream modules receive a GeneratorConfig instance, never raw dicts.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# Sub-dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttributeRangeCfg:
    """Numeric value range for one attribute."""
    min_val: float
    max_val: float
    unit: str = ""


@dataclass
class TaskFamilyCfg:
    """Difficulty breakdown for one task family."""
    easy:   int = 0
    medium: int = 0
    hard:   int = 0

    @property
    def total(self) -> int:
        return self.easy + self.medium + self.hard


@dataclass
class SplitCfg:
    """Absolute example counts for each split."""
    train: int
    val:   int
    test:  int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


# ─────────────────────────────────────────────────────────────────────────────
# Main config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratorConfig:
    """
    Fully-typed configuration for the logical generator.

    Load with:
        cfg = GeneratorConfig.from_yaml("phase3/configs/logical_generator.yaml")
    """

    seed:                int
    total_examples:      int
    examples_per_family: int
    splits:              SplitCfg
    task_families:       Dict[str, TaskFamilyCfg]
    attributes_per_family: int
    relational_per_family: int
    numerical_per_family:  int
    attribute_ranges:    Dict[str, AttributeRangeCfg]
    output_path:         str
    max_retries:         int = 200

    # Ordered task family names (preserved from YAML insertion order)
    task_family_order: List[str] = field(default_factory=list)

    # ── Canonical attribute / relation lists ──────────────────────────────
    ATTRIBUTES: List[str] = field(default_factory=lambda: [
        "height", "age", "price", "quantity"
    ])
    RELATIONS: List[str] = field(default_factory=lambda: [
        "greater_than", "less_than", "equal"
    ])

    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GeneratorConfig":
        """Load and validate a YAML configuration file."""
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        # Task families
        task_families: Dict[str, TaskFamilyCfg] = {}
        for name, counts in raw["task_families"].items():
            task_families[name] = TaskFamilyCfg(
                easy=counts.get("easy", 0),
                medium=counts.get("medium", 0),
                hard=counts.get("hard", 0),
            )

        # Attribute ranges
        attr_ranges: Dict[str, AttributeRangeCfg] = {}
        for name, rng in raw["attribute_ranges"].items():
            attr_ranges[name] = AttributeRangeCfg(
                min_val=rng["min"],
                max_val=rng["max"],
                unit=rng.get("unit", ""),
            )

        sp = raw["splits"]

        cfg = cls(
            seed=raw["seed"],
            total_examples=raw["total_examples"],
            examples_per_family=raw.get("examples_per_family", 1000),
            splits=SplitCfg(train=sp["train"], val=sp["val"], test=sp["test"]),
            task_families=task_families,
            attributes_per_family=raw.get("attributes_per_family", 250),
            relational_per_family=raw.get("relational_per_family", 750),
            numerical_per_family=raw.get("numerical_per_family", 250),
            attribute_ranges=attr_ranges,
            output_path=raw.get("output_path", "phase3/outputs"),
            max_retries=raw.get("max_retries", 200),
            task_family_order=list(task_families.keys()),
        )
        cfg._validate()
        return cfg

    # ─────────────────────────────────────────────────────────────────────
    def _validate(self) -> None:
        """Sanity-check the loaded configuration."""
        total = sum(tf.total for tf in self.task_families.values())
        if total != self.total_examples:
            raise ValueError(
                f"Sum of task family totals ({total}) ≠ total_examples ({self.total_examples})"
            )

        split_total = self.splits.total
        if split_total != self.total_examples:
            raise ValueError(
                f"Split total ({split_total}) ≠ total_examples ({self.total_examples})"
            )

        for name, tf in self.task_families.items():
            if tf.total != self.examples_per_family:
                raise ValueError(
                    f"Task '{name}' sums to {tf.total}, expected {self.examples_per_family}"
                )

        rep_total = self.relational_per_family + self.numerical_per_family
        if rep_total != self.examples_per_family:
            raise ValueError(
                f"relational+numerical ({rep_total}) ≠ examples_per_family ({self.examples_per_family})"
            )

        attr_total = len(self.ATTRIBUTES) * self.attributes_per_family
        if attr_total != self.examples_per_family:
            raise ValueError(
                f"attributes×per_family ({attr_total}) ≠ examples_per_family ({self.examples_per_family})"
            )
