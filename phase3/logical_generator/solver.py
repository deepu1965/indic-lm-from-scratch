"""
solver.py — Logical solver for the Phase 3.1 logical generator.

The solver is intentionally INDEPENDENT of the generation logic.
It re-derives the answer from the given facts alone, providing a
genuine validation step separate from the latent-value system.

Public API
----------
ContradictionError          — raised when facts are logically inconsistent
Solver(facts)               — builds transitive closure from a list of Facts
    .query(e1, e2)          → Optional[Relation]
    .find_extreme(entities, extreme) → Optional[str]
    .is_directly_stated(e1, e2)      → bool

compute_shortest_path(facts, e1, e2) → Optional[int]
    BFS hop count in the undirected fact graph.
"""

from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .relations import Relation, invert, compose
from .structures import Fact


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ContradictionError(Exception):
    """Raised when a set of facts is logically contradictory."""


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

class Solver:
    """
    Derives all implied pairwise relations from a list of given Facts via
    transitive closure (Floyd–Warshall style over three-valued relations).

    Construction raises ContradictionError if the facts are inconsistent.
    """

    def __init__(self, facts: List[Fact]) -> None:
        self._facts = facts
        self._entities: Set[str] = set()
        for f in facts:
            self._entities.add(f.subject)
            self._entities.add(f.object)

        # Relation matrix: (a, b) → Relation
        self._rel: Dict[Tuple[str, str], Relation] = {}

        # Direct-fact index: pairs explicitly stated in facts
        self._direct: Set[Tuple[str, str]] = set()

        self._build()

    # ── Public interface ──────────────────────────────────────────────────

    def query(self, e1: str, e2: str) -> Optional[Relation]:
        """
        Return the derived relation of e1 to e2, or None if undetermined.

        Does NOT raise — returns None for undetermined / unknown entities.
        """
        if e1 == e2:
            return Relation.EQUAL
        return self._rel.get((e1, e2))

    def find_extreme(
        self, entities: List[str], extreme: str
    ) -> Optional[str]:
        """
        Return the unique extreme (largest or smallest) entity, or None.

        An entity is a *potential* extreme if, for every other entity, the
        relation is known and it is not beaten (i.e. not LESS_THAN for
        largest, not GREATER_THAN for smallest).

        Returns None if:
            • there are zero or multiple potential extremes (tied), or
            • any relation between candidates is unknown.
        """
        if extreme not in ("largest", "smallest"):
            raise ValueError(f"extreme must be 'largest' or 'smallest', got {extreme!r}")

        losing_rel = Relation.LESS_THAN if extreme == "largest" else Relation.GREATER_THAN

        potential: List[str] = []
        for e in entities:
            beats_all = True
            for other in entities:
                if e == other:
                    continue
                rel = self.query(e, other)
                if rel is None or rel == losing_rel:
                    beats_all = False
                    break
            if beats_all:
                potential.append(e)

        if len(potential) == 1:
            return potential[0]
        # Multiple potential extremes means they're tied — no unique answer
        return None

    def is_directly_stated(self, e1: str, e2: str) -> bool:
        """
        Return True if any given fact directly asserts a relation between
        e1 and e2 (in either direction).
        """
        return (e1, e2) in self._direct or (e2, e1) in self._direct

    @property
    def entities(self) -> Set[str]:
        return set(self._entities)

    # ── Private helpers ───────────────────────────────────────────────────

    def _set_rel(self, a: str, b: str, rel: Relation) -> None:
        """
        Record rel(a,b) and invert(rel)(b,a), raising ContradictionError on
        conflict.  Also sets the reflexive EQ for a==b without error.
        """
        if a == b:
            if rel != Relation.EQUAL:
                raise ContradictionError(
                    f"Self-comparison contradiction: {a} {rel.value} {a}"
                )
            return  # A = A is trivially true; no need to store

        for src, dst, r in [(a, b, rel), (b, a, invert(rel))]:
            key = (src, dst)
            existing = self._rel.get(key)
            if existing is None:
                self._rel[key] = r
            elif existing != r:
                raise ContradictionError(
                    f"Contradiction: {src} {existing.value} {dst} "
                    f"vs {src} {r.value} {dst}"
                )

    def _build(self) -> None:
        """Initialise from direct facts, then run transitive closure."""
        # Phase 1: direct facts
        for f in self._facts:
            self._set_rel(f.subject, f.object, Relation(f.relation))
            self._direct.add((f.subject, f.object))
            self._direct.add((f.object, f.subject))

        # Phase 2: Floyd–Warshall transitive closure
        entities = list(self._entities)
        changed = True
        while changed:
            changed = False
            for a in entities:
                for b in entities:
                    if a == b:
                        continue
                    rab = self._rel.get((a, b))
                    if rab is None:
                        continue
                    for c in entities:
                        if b == c or a == c:
                            continue
                        rbc = self._rel.get((b, c))
                        if rbc is None:
                            continue
                        composed = compose(rab, rbc)
                        if composed is None:
                            continue
                        rac = self._rel.get((a, c))
                        if rac is None:
                            self._set_rel(a, c, composed)
                            changed = True
                        elif rac != composed:
                            raise ContradictionError(
                                f"Contradiction via {a}→{b}→{c}: "
                                f"have {a} {rac.value} {c}, "
                                f"derived {a} {composed.value} {c}"
                            )


# ─────────────────────────────────────────────────────────────────────────────
# BFS hop distance
# ─────────────────────────────────────────────────────────────────────────────

def compute_shortest_path(
    facts: List[Fact], e1: str, e2: str
) -> Optional[int]:
    """
    Compute the shortest path length (hop count) between e1 and e2 in the
    **undirected** fact graph (where each Fact is one undirected edge).

    Returns None if e2 is unreachable from e1.
    Returns 0 if e1 == e2.
    """
    if e1 == e2:
        return 0

    # Build undirected adjacency list
    graph: Dict[str, Set[str]] = defaultdict(set)
    for f in facts:
        graph[f.subject].add(f.object)
        graph[f.object].add(f.subject)

    visited: Set[str] = {e1}
    queue: deque[Tuple[str, int]] = deque([(e1, 0)])

    while queue:
        node, dist = queue.popleft()
        for neighbor in graph[node]:
            if neighbor == e2:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return None  # unreachable
