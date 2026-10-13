"""
structures.py — Core data structures for the Phase 3.1 logical generator.

All structures are plain dataclasses; no domain logic lives here.
The `to_dict` method produces the canonical JSON-serialisable representation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityRef:
    """A reference to an abstract entity by its unique ID (e.g. 'E01')."""
    id: str

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EntityRef):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return NotImplemented


@dataclass
class Fact:
    """
    A single relation assertion between two entities.

    subject  : entity ID of the left-hand entity
    relation : one of "greater_than" | "less_than" | "equal"
    object   : entity ID of the right-hand entity
    """
    subject:  str
    relation: str   # "greater_than" | "less_than" | "equal"
    object:   str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fact):
            return (self.subject == other.subject
                    and self.relation == other.relation
                    and self.object == other.object)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.subject, self.relation, self.object))


@dataclass
class Question:
    """
    The question posed to the model.

    type     : "compare" | "largest" | "smallest"
    entities : entity IDs relevant to the question
    """
    type:     str         # "compare" | "largest" | "smallest"
    entities: List[str]


@dataclass
class GroundTruth:
    """
    The correct answer to the question.

    For "compare":
        relation      : relation of entities[0] to entities[1]
                        ("greater_than" | "less_than" | "equal")
        answer_entity : entity that is greater (None if equal)

    For "largest" / "smallest":
        answer_entity : the unique extreme entity
        relation      : None
    """
    relation:      Optional[str] = None   # for compare
    answer_entity: Optional[str] = None  # the "winning" or extreme entity


# ─────────────────────────────────────────────────────────────────────────────
# Main record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LogicalRecord:
    """
    A single language-independent logical reasoning example.

    Fields that belong to later stages (prompts, Hindi/Nepali text,
    template_id) are deliberately absent.
    """

    # ── Identity & classification ──────────────────────────────────────────
    id:                  str       # zero-padded 6-digit global sequence number
    task_family:         str       # e.g. "two_hop_transitivity"
    difficulty:          str       # "easy" | "medium" | "hard"

    # ── Representation ─────────────────────────────────────────────────────
    attribute:           str       # "height" | "age" | "price" | "quantity"
    representation:      str       # "relational" | "numerical"

    # ── Logical content ────────────────────────────────────────────────────
    entities:            List[EntityRef]
    facts:               List[Fact]
    question:            Question
    ground_truth:        GroundTruth

    # ── Meta-reasoning ─────────────────────────────────────────────────────
    reasoning_path:      List[str]   # entity IDs along the intended chain
    relation_pattern_id: str         # e.g. "P04"

    # ── Reproducibility ────────────────────────────────────────────────────
    seed:                int

    # ── Split assignment ───────────────────────────────────────────────────
    split:               str = ""   # "train" | "val" | "test" (set by SplitManager)

    # ── Numerical values (only for representation == "numerical") ──────────
    values: Optional[Dict[str, int]] = None   # entity_id → numeric value

    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Canonical JSON-serialisable representation."""
        d: Dict[str, Any] = {
            "id":                  self.id,
            "task_family":         self.task_family,
            "difficulty":          self.difficulty,
            "attribute":           self.attribute,
            "representation":      self.representation,
            "entities":            [{"id": e.id} for e in self.entities],
            "facts": [
                {
                    "subject":  f.subject,
                    "relation": f.relation,
                    "object":   f.object,
                }
                for f in self.facts
            ],
            "question": {
                "type":     self.question.type,
                "entities": self.question.entities,
            },
            "ground_truth": {
                k: v
                for k, v in {
                    "relation":      self.ground_truth.relation,
                    "answer_entity": self.ground_truth.answer_entity,
                }.items()
                if v is not None
            },
            "reasoning_path":      self.reasoning_path,
            "relation_pattern_id": self.relation_pattern_id,
            "seed":                self.seed,
            "split":               self.split,
        }
        if self.values is not None:
            d["values"] = self.values
        return d

    def to_json(self, **kwargs: Any) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
