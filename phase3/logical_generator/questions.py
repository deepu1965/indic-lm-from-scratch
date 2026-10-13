"""
questions.py — Helper functions for building questions and computing ground truths.

All helpers are pure functions; they do not mutate any state.
The generator calls these instead of duplicating the logic in each sub-generator.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .relations import Relation, derive_relation
from .structures import GroundTruth, Question


# ─────────────────────────────────────────────────────────────────────────────
# Question builders
# ─────────────────────────────────────────────────────────────────────────────

def build_compare_question(e1: str, e2: str) -> Question:
    """Return a compare question between two entities."""
    return Question(type="compare", entities=[e1, e2])


def build_largest_question(entities: List[str]) -> Question:
    """Return a largest question over a set of entities."""
    return Question(type="largest", entities=list(entities))


def build_smallest_question(entities: List[str]) -> Question:
    """Return a smallest question over a set of entities."""
    return Question(type="smallest", entities=list(entities))


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth computation from latent values
# ─────────────────────────────────────────────────────────────────────────────

def compute_compare_ground_truth(
    latent: Dict[str, float],
    e1: str,
    e2: str,
) -> GroundTruth:
    """
    Compute the ground truth for compare(e1, e2) using latent values.

    relation:      relation of e1 to e2
    answer_entity: the entity with the higher latent value (None if equal)
    """
    rel = derive_relation(latent[e1], latent[e2])
    if rel == Relation.GREATER_THAN:
        answer_entity: Optional[str] = e1
    elif rel == Relation.LESS_THAN:
        answer_entity = e2
    else:
        answer_entity = None
    return GroundTruth(relation=rel.value, answer_entity=answer_entity)


def compute_extreme_ground_truth(
    latent: Dict[str, float],
    entities: List[str],
    extreme: str,
) -> Optional[GroundTruth]:
    """
    Compute the ground truth for a largest/smallest question using latent values.

    Returns None if there is no unique extreme (tie).

    extreme: "largest" | "smallest"
    """
    if extreme == "largest":
        best_val = max(latent[e] for e in entities)
        winners = [e for e in entities if latent[e] == best_val]
    else:
        best_val = min(latent[e] for e in entities)
        winners = [e for e in entities if latent[e] == best_val]

    if len(winners) != 1:
        return None   # tied — no unique answer

    return GroundTruth(answer_entity=winners[0])
