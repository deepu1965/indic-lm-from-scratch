"""
patterns.py — Canonical relation pattern registry.

Each pattern describes the ordering structure of a set of entities via
a latent value template.  The relation_pattern_id field in every
LogicalRecord is computed here.

Patterns P01–P12 match the spec exactly.  Additional patterns P13–P15
are used for structures that arise in three-hop and mixed-equality tasks.

Pattern ID assignment:
    infer_pattern_id(latent_values) → str

    The function examines the shape of the latent value list and returns
    the matching pattern ID, or "P_custom" for unrecognised shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Pattern descriptor (informational only — not used in hot path)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PatternSpec:
    """Human-readable description of a canonical pattern."""
    id:          str
    description: str
    num_entities: int


PATTERN_REGISTRY: Dict[str, PatternSpec] = {
    "P01": PatternSpec("P01", "A > B",            2),
    "P02": PatternSpec("P02", "A < B",            2),
    "P03": PatternSpec("P03", "A = B",            2),
    "P04": PatternSpec("P04", "A > B > C",        3),
    "P05": PatternSpec("P05", "A < B < C",        3),
    "P06": PatternSpec("P06", "A > B = C",        3),
    "P07": PatternSpec("P07", "A = B > C",        3),
    "P08": PatternSpec("P08", "A > B > C > D",    4),
    "P09": PatternSpec("P09", "A < B < C < D",    4),
    "P10": PatternSpec("P10", "A > B = C > D",    4),
    "P11": PatternSpec("P11", "A = B > C > D",    4),
    "P12": PatternSpec("P12", "A > B = C < D",    4),
    "P13": PatternSpec("P13", "A > B > C = D",    4),
    "P14": PatternSpec("P14", "A = B > C = D",    4),
    "P15": PatternSpec("P15", "A = B > C > D = E", 5),
}


# ─────────────────────────────────────────────────────────────────────────────
# Pattern inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_pattern_id(latent_values: List[float]) -> str:
    """
    Return the pattern ID that matches the given latent value list.

    The latent values are the raw scores assigned to entities in creation
    order: latent_values[i] is the score of the i-th entity.

    Example:
        [3, 2, 1]  → "P04"  (A > B > C)
        [4, 2, 2, 1] → "P10" (A > B = C > D)
    """
    n = len(latent_values)
    v = latent_values  # shorthand

    if n == 2:
        if v[0] > v[1]:
            return "P01"
        if v[0] < v[1]:
            return "P02"
        return "P03"

    if n == 3:
        a, b, c = v
        if a > b > c:
            return "P04"
        if a < b < c:
            return "P05"
        if a > b == c:
            return "P06"
        if a == b > c:
            return "P07"
        return "P_custom"

    if n == 4:
        a, b, c, d = v
        if a > b > c > d:
            return "P08"
        if a < b < c < d:
            return "P09"
        if a > b == c > d:
            return "P10"
        if a == b > c > d:
            return "P11"
        if a > b == c and c < d:   # A > B = C < D
            return "P12"
        if a > b > c == d:
            return "P13"
        if a == b > c == d:
            return "P14"
        return "P_custom"

    if n == 5:
        a, b, c, d, e = v
        if a == b > c > d == e:
            return "P15"
        return "P_custom"

    return "P_custom"
