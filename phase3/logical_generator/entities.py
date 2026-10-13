"""
entities.py — Abstract entity management for the logical generator.

The generator works entirely with abstract entity IDs (E01, E02, …).
Language-specific names (Ravi, Meera, …) are injected at the template stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .structures import EntityRef


@dataclass
class Entity:
    """A named abstract entity."""
    id: str

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.id == other.id
        if isinstance(other, str):
            return self.id == other
        return NotImplemented

    def to_ref(self) -> EntityRef:
        return EntityRef(id=self.id)


class EntityPool:
    """
    Generates pools of abstract entity IDs.

    IDs are zero-padded two-digit strings: E01, E02, …, E26.
    Language-specific substitution is NOT performed here.
    """

    def __init__(self, prefix: str = "E", max_entities: int = 26) -> None:
        self.prefix = prefix
        self.max_entities = max_entities

    def get_ids(self, n: int) -> List[str]:
        """
        Return a list of *n* abstract entity IDs starting at prefix + '01'.

        Raises ValueError if n > max_entities.
        """
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")
        if n > self.max_entities:
            raise ValueError(
                f"Requested {n} entities; maximum is {self.max_entities}"
            )
        return [f"{self.prefix}{i + 1:02d}" for i in range(n)]

    def get_entities(self, n: int) -> List[Entity]:
        """Return a list of *n* Entity objects."""
        return [Entity(eid) for eid in self.get_ids(n)]

    def get_refs(self, n: int) -> List[EntityRef]:
        """Return a list of *n* EntityRef objects suitable for LogicalRecord.entities."""
        return [EntityRef(eid) for eid in self.get_ids(n)]
