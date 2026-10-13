"""
relations.py — Relation definitions and algebra for the logical generator.

Three primitive relations are used throughout:
    greater_than  ↔  less_than  (symmetric inverses)
    equal                        (symmetric)

Key operations:
    invert(rel)         → the inverse relation
    derive_relation(a,b)→ relation from latent numeric values
    compose(r1, r2)     → transitivity composition (returns None if indeterminate)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class Relation(str, Enum):
    """The three canonical primitive relations."""
    GREATER_THAN = "greater_than"
    LESS_THAN    = "less_than"
    EQUAL        = "equal"


# ─────────────────────────────────────────────────────────────────────────────
# Algebraic operations
# ─────────────────────────────────────────────────────────────────────────────

def invert(rel: Relation) -> Relation:
    """
    Return the inverse (symmetric complement) of a relation.

    greater_than → less_than
    less_than    → greater_than
    equal        → equal
    """
    if rel == Relation.GREATER_THAN:
        return Relation.LESS_THAN
    if rel == Relation.LESS_THAN:
        return Relation.GREATER_THAN
    return Relation.EQUAL


def derive_relation(va: float, vb: float) -> Relation:
    """
    Derive the relation of entity-A to entity-B from their latent numeric values.

    va > vb → GREATER_THAN
    va < vb → LESS_THAN
    va == vb → EQUAL
    """
    if va > vb:
        return Relation.GREATER_THAN
    if va < vb:
        return Relation.LESS_THAN
    return Relation.EQUAL


def compose(r1: Relation, r2: Relation) -> Optional[Relation]:
    """
    Transitively compose two relations.

    Given A r1 B and B r2 C, return the implied relation of A to C.
    Returns None when the composition is indeterminate (e.g. GT ∘ LT).

    Composition table:
        GT  ∘ GT  = GT
        GT  ∘ EQ  = GT
        EQ  ∘ GT  = GT
        EQ  ∘ EQ  = EQ
        LT  ∘ LT  = LT
        LT  ∘ EQ  = LT
        EQ  ∘ LT  = LT
        GT  ∘ LT  = None   (indeterminate)
        LT  ∘ GT  = None   (indeterminate)
    """
    EQ = Relation.EQUAL
    GT = Relation.GREATER_THAN
    LT = Relation.LESS_THAN

    if r1 == EQ:
        return r2
    if r2 == EQ:
        return r1
    if r1 == r2:          # GT∘GT = GT,  LT∘LT = LT
        return r1
    return None           # GT∘LT or LT∘GT — indeterminate
