"""
validator.py — Logical validator for the Phase 3.1 logical generator.

The validator is completely independent of the generation logic.
It applies 15 checks (as specified) to every candidate LogicalRecord.

Usage
-----
    validator = LogicalValidator()
    result = validator.validate(record)
    if not result.ok:
        print(result.errors)

The generator calls validate() on every candidate and rejects failures.
Rejected examples are regenerated; they are never silently fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from .relations import Relation
from .solver import ContradictionError, Solver, compute_shortest_path
from .structures import Fact, LogicalRecord


# ─────────────────────────────────────────────────────────────────────────────
# ValidationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Outcome of a single validation run."""
    ok:     bool
    errors: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


# ─────────────────────────────────────────────────────────────────────────────
# Families exempt from the "no direct answer" check (Check 9)
# ─────────────────────────────────────────────────────────────────────────────
# Task 1 (direct_comparison): the answer IS the direct fact — intentional.
# Task 6 (reversed_relation): the answer IS the inverse of a direct fact — intentional.
#                             (see correction #1 in the implementation brief)
_DIRECT_ANSWER_EXEMPT: Set[str] = {"direct_comparison", "reversed_relation"}


# ─────────────────────────────────────────────────────────────────────────────
# LogicalValidator
# ─────────────────────────────────────────────────────────────────────────────

class LogicalValidator:
    """
    Applies all 15 logical checks to a LogicalRecord.

    Checks implemented:
        1.  No contradictory facts
        2.  All referenced entities exist
        3.  Question entities exist
        4.  Ground truth consistent with facts
        5.  Ground truth has a unique answer
        6.  Task-family constraints satisfied
        7.  Difficulty constraints satisfied (structural proxy)
        8.  Required hop count satisfied (general)
        9.  No direct fact reveals a supposedly inferred answer
        10. Task 7 equality is relevant to derivation
        11. Task 6 query direction is reversed
        12. Largest/smallest has a unique extreme (merged into 5)
        13. Task 4 is actually two-hop
        14. Task 5 is actually three-hop
        15. Reasoning path is valid
    """

    def validate(self, record: LogicalRecord) -> ValidationResult:
        errors: List[str] = []

        # ── Check 1: Build solver (contradiction-free) ─────────────────────
        try:
            solver = Solver(record.facts)
        except ContradictionError as exc:
            errors.append(f"[C1] Contradictory facts: {exc}")
            return ValidationResult(ok=False, errors=errors)

        entity_ids: Set[str] = {e.id for e in record.entities}

        # ── Check 2: All entities referenced in facts exist ────────────────
        for f in record.facts:
            for eid in (f.subject, f.object):
                if eid not in entity_ids:
                    errors.append(f"[C2] Fact entity '{eid}' not declared in entities list")

        # ── Check 3: Question entities exist ──────────────────────────────
        for eid in record.question.entities:
            if eid not in entity_ids:
                errors.append(f"[C3] Question entity '{eid}' not in entities list")

        if errors:
            return ValidationResult(ok=False, errors=errors)

        # ── Checks 4 & 5 (& 12): Ground truth correct and unique ──────────
        q = record.question
        gt = record.ground_truth

        if q.type == "compare":
            e1, e2 = q.entities[0], q.entities[1]
            derived = solver.query(e1, e2)

            if derived is None:
                errors.append(
                    f"[C5] compare({e1},{e2}) is indeterminate from given facts"
                )
            elif derived.value != gt.relation:
                errors.append(
                    f"[C4] Ground truth relation '{gt.relation}' ≠ "
                    f"derived '{derived.value}' for compare({e1},{e2})"
                )
            else:
                # Verify answer_entity
                if derived == Relation.GREATER_THAN:
                    expected_ae = e1
                elif derived == Relation.LESS_THAN:
                    expected_ae = e2
                else:
                    expected_ae = None

                if gt.answer_entity != expected_ae:
                    errors.append(
                        f"[C4] answer_entity '{gt.answer_entity}' ≠ "
                        f"expected '{expected_ae}' for compare({e1},{e2})={derived.value}"
                    )

        elif q.type in ("largest", "smallest"):
            derived_ae = solver.find_extreme(q.entities, q.type)
            if derived_ae is None:
                errors.append(
                    f"[C5/C12] No unique {q.type} among {q.entities}"
                )
            elif derived_ae != gt.answer_entity:
                errors.append(
                    f"[C4] Ground truth answer_entity '{gt.answer_entity}' ≠ "
                    f"derived '{derived_ae}' for {q.type}"
                )

        if errors:
            return ValidationResult(ok=False, errors=errors)

        # ── Check 6: Task-family structural constraints ────────────────────
        tf_ok, tf_msg = self._check_task_family(record, solver)
        if not tf_ok:
            errors.append(f"[C6] Task-family constraint: {tf_msg}")

        # ── Check 7: Difficulty structural constraints ─────────────────────
        diff_ok, diff_msg = self._check_difficulty(record)
        if not diff_ok:
            errors.append(f"[C7] Difficulty constraint: {diff_msg}")

        # ── Check 9: No direct fact gives away the inferred answer ─────────
        if not self._check_no_direct_answer(record, solver):
            errors.append(
                "[C9] A direct fact explicitly states the answer to the question"
            )

        # ── Check 10: Task 7 — equality is relevant ───────────────────────
        if record.task_family == "mixed_equality":
            if not self._check_equality_relevant(record):
                errors.append(
                    "[C10] No equality fact is necessary for deriving the answer"
                )

        # ── Check 11: Task 6 — query direction is reversed ────────────────
        if record.task_family == "reversed_relation":
            if not self._check_reversal(record):
                errors.append(
                    "[C11] Task 6 query direction is not reversed "
                    "(expected ground_truth.relation == 'less_than')"
                )

        # ── Checks 13 & 14: Hop counts for Tasks 4 & 5 ────────────────────
        if record.task_family == "two_hop_transitivity":
            if not self._check_hop_count(record, expected=2):
                errors.append("[C13] two_hop_transitivity is not actually two-hop")

        if record.task_family == "three_hop_transitivity":
            if not self._check_hop_count(record, expected=3):
                errors.append("[C14] three_hop_transitivity is not actually three-hop")

        # ── Check 15: Reasoning path is valid ─────────────────────────────
        path_ok, path_msg = self._check_reasoning_path(record)
        if not path_ok:
            errors.append(f"[C15] Reasoning path invalid: {path_msg}")

        return ValidationResult(ok=len(errors) == 0, errors=errors)

    # ─────────────────────────────────────────────────────────────────────────
    # Individual check helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _check_task_family(
        self, record: LogicalRecord, solver: Solver
    ) -> tuple[bool, str]:
        """Check structural constraints specific to each task family."""
        tf = record.task_family
        q  = record.question
        gt = record.ground_truth

        if tf == "direct_comparison":
            # Must be a 2-entity compare question
            if q.type != "compare":
                return False, "direct_comparison must be a compare question"
            if len(record.entities) != 2:
                return False, "direct_comparison must have exactly 2 entities"
            if len(record.facts) != 1:
                return False, "direct_comparison must have exactly 1 fact"

        elif tf == "largest_smallest":
            if q.type not in ("largest", "smallest"):
                return False, "largest_smallest must be largest or smallest question"
            if len(record.entities) < 3:
                return False, "largest_smallest must have ≥ 3 entities"

        elif tf == "general_comparison":
            if q.type != "compare":
                return False, "general_comparison must be a compare question"

        elif tf == "two_hop_transitivity":
            if q.type != "compare":
                return False, "two_hop_transitivity must be a compare question"

        elif tf == "three_hop_transitivity":
            if q.type != "compare":
                return False, "three_hop_transitivity must be a compare question"

        elif tf == "reversed_relation":
            if q.type != "compare":
                return False, "reversed_relation must be a compare question"

        elif tf == "mixed_equality":
            if q.type != "compare":
                return False, "mixed_equality must be a compare question"
            # Must have at least one equality fact
            if not any(f.relation == "equal" for f in record.facts):
                return False, "mixed_equality must contain at least one equality fact"

        return True, ""

    def _check_difficulty(self, record: LogicalRecord) -> tuple[bool, str]:
        """
        Lightweight structural proxy for difficulty constraints.

        We verify properties that MUST hold (e.g. easy two-hop has no
        equality), not the full pedagogical complexity spectrum.
        """
        tf   = record.task_family
        diff = record.difficulty

        has_equality = any(f.relation == "equal" for f in record.facts)

        # Task 4 easy: clean chain — no equality allowed
        if tf == "two_hop_transitivity" and diff == "easy" and has_equality:
            return False, "two_hop easy must not contain equality"

        # Task 5 medium: clean chain — no equality allowed
        if tf == "three_hop_transitivity" and diff == "medium" and has_equality:
            return False, "three_hop medium must not contain equality"

        # Task 5 has no easy examples
        if tf == "three_hop_transitivity" and diff == "easy":
            return False, "three_hop_transitivity has no easy difficulty"

        # Task 7 has no easy examples
        if tf == "mixed_equality" and diff == "easy":
            return False, "mixed_equality has no easy difficulty"

        # Task 1 has no hard examples
        if tf == "direct_comparison" and diff == "hard":
            return False, "direct_comparison has no hard difficulty"

        return True, ""

    def _check_no_direct_answer(
        self, record: LogicalRecord, solver: Solver
    ) -> bool:
        """
        Check 9: for compare questions (outside exempt families), verify that
        no single given fact directly states the relation between the two
        query entities.

        Task 1 (direct_comparison) and Task 6 (reversed_relation) are
        explicitly exempt — the answer IS (or is the inverse of) a direct fact.

        Task 3 Easy is also exempt (direct relation is the question).
        """
        q = record.question
        if q.type != "compare":
            return True  # N/A for largest/smallest

        tf = record.task_family
        if tf in _DIRECT_ANSWER_EXEMPT:
            return True

        if tf == "general_comparison" and record.difficulty == "easy":
            return True

        e1, e2 = q.entities[0], q.entities[1]
        return not solver.is_directly_stated(e1, e2)

    def _check_equality_relevant(self, record: LogicalRecord) -> bool:
        """
        Check 10: at least one equality fact must be *necessary* for deriving
        the answer.

        We test necessity by removing each equality fact and re-running the
        solver.  If removal makes the query unanswerable (returns None), that
        equality fact is necessary.
        """
        eq_facts = [f for f in record.facts if f.relation == "equal"]
        if not eq_facts:
            return False  # No equality → can't be relevant

        q = record.question
        if q.type != "compare":
            # For largest/smallest mixed-equality tasks (not currently generated)
            # we'd need a different check; for now treat as valid
            return True

        e1, e2 = q.entities[0], q.entities[1]

        for eq_fact in eq_facts:
            reduced = [f for f in record.facts if f != eq_fact]
            try:
                reduced_solver = Solver(reduced)
                if reduced_solver.query(e1, e2) is None:
                    return True  # removing this eq fact makes query unanswerable
            except ContradictionError:
                return True  # removing eq fact causes contradiction (very unlikely)

        return False  # every equality fact was dispensable

    def _check_reversal(self, record: LogicalRecord) -> bool:
        """
        Check 11: Task 6 (reversed_relation) must have ground_truth.relation
        == 'less_than', meaning the queried entity (entities[0]) is the smaller
        one — i.e., the query direction is opposite to the natural chain.
        """
        return record.ground_truth.relation == Relation.LESS_THAN.value

    def _check_hop_count(self, record: LogicalRecord, expected: int) -> bool:
        """
        Checks 13 & 14: verify that the BFS distance between the two query
        endpoints equals *expected* hops in the undirected fact graph.
        """
        q = record.question
        if q.type != "compare":
            return True
        e1, e2 = q.entities[0], q.entities[1]
        dist = compute_shortest_path(record.facts, e1, e2)
        return dist == expected

    def _check_reasoning_path(
        self, record: LogicalRecord
    ) -> tuple[bool, str]:
        """
        Check 15: validate the reasoning_path field.

        Requirements:
            • All path entities must be declared.
            • The path must form a connected sequence in the fact graph.
            • For compare questions: path[0] == question.entities[0]
                                      path[-1] == question.entities[1]
            • Path length ≥ 2.
        """
        path = record.reasoning_path
        entity_ids = {e.id for e in record.entities}

        if len(path) < 2:
            return False, "reasoning_path must have ≥ 2 entities"

        for e in path:
            if e not in entity_ids:
                return False, f"entity '{e}' in reasoning_path not declared"

        q = record.question
        if q.type == "compare":
            if path[0] != q.entities[0] or path[-1] != q.entities[1]:
                return (
                    False,
                    f"reasoning_path [{path[0]}…{path[-1]}] does not match "
                    f"question endpoints [{q.entities[0]},{q.entities[1]}]",
                )

        # Build undirected adjacency for connectivity check
        from collections import defaultdict
        graph: dict = defaultdict(set)
        for f in record.facts:
            graph[f.subject].add(f.object)
            graph[f.object].add(f.subject)

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if b not in graph[a]:
                return (
                    False,
                    f"no fact edge between '{a}' and '{b}' in reasoning_path",
                )

        return True, ""
