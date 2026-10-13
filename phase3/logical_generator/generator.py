"""
generator.py — Central LogicalGenerator and seven task-family sub-generators.

Design
------
1.  Every example starts from a *latent value vector* assigned to entities.
    Relations are derived deterministically from these values — no contradiction
    is possible by construction.

2.  Each sub-generator method returns a LogicalRecord candidate.  The candidate
    is then passed to LogicalValidator.  On failure: retry with a fresh RNG
    state.  On success: keep.

3.  Attribute and representation slots are pre-computed for every family to
    guarantee exact per-family counts (250 per attribute, 750/250 rel/num).

4.  All randomness flows through explicit random.Random instances seeded from
    config.seed.  Global random functions are never used.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .attributes import AttributeRange, sample_ordered_values
from .config import GeneratorConfig
from .entities import EntityPool
from .patterns import infer_pattern_id
from .questions import (
    build_compare_question,
    build_largest_question,
    build_smallest_question,
    compute_compare_ground_truth,
    compute_extreme_ground_truth,
)
from .relations import Relation, derive_relation, invert
from .structures import EntityRef, Fact, GroundTruth, LogicalRecord, Question
from .validator import LogicalValidator


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class GenerationError(RuntimeError):
    """Raised when a task family exhausts max_retries without a valid example."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _vary_facts(
    facts: List[Fact],
    rng: random.Random,
    flip_prob: float = 0.40,
    force_shuffle: bool = True,
) -> List[Fact]:
    """
    Apply random presentation variations:
        • Each fact is independently flipped with probability `flip_prob`.
          Flipping A>B gives B<A (logically equivalent; different surface form).
        • The fact list is shuffled.

    The force_shuffle flag does nothing for single-element lists.
    """
    result: List[Fact] = []
    for f in facts:
        if rng.random() < flip_prob:
            result.append(
                Fact(
                    subject=f.object,
                    relation=invert(Relation(f.relation)).value,
                    object=f.subject,
                )
            )
        else:
            result.append(f)
    if force_shuffle and len(result) > 1:
        rng.shuffle(result)
    return result


def _build_record(
    *,
    task_family: str,
    difficulty: str,
    attribute: str,
    representation: str,
    entity_ids: List[str],
    latent: List[float],
    facts: List[Fact],
    question: Question,
    ground_truth: GroundTruth,
    reasoning_path: List[str],
    rng: random.Random,
    attr_ranges: Dict[str, AttributeRange],
    seed_val: int,
) -> LogicalRecord:
    """Assemble a LogicalRecord from its components."""
    latent_map = {eid: latent[i] for i, eid in enumerate(entity_ids)}

    values: Optional[Dict[str, int]] = None
    if representation == "numerical":
        values = sample_ordered_values(
            rng, entity_ids, latent, attribute, attr_ranges
        )

    return LogicalRecord(
        id="",                    # assigned by generator after validation
        task_family=task_family,
        difficulty=difficulty,
        attribute=attribute,
        representation=representation,
        entities=[EntityRef(eid) for eid in entity_ids],
        facts=facts,
        question=question,
        ground_truth=ground_truth,
        reasoning_path=reasoning_path,
        relation_pattern_id=infer_pattern_id(latent),
        seed=seed_val,
        split="",                 # assigned by SplitManager
        values=values,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LogicalGenerator
# ─────────────────────────────────────────────────────────────────────────────

class LogicalGenerator:
    """
    Orchestrates generation of all 7,000 logical records.

    Usage
    -----
        cfg = GeneratorConfig.from_yaml("phase3/configs/logical_generator.yaml")
        gen = LogicalGenerator(cfg)
        records, stats = gen.generate()
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self.cfg       = config
        self.ep        = EntityPool()
        self.validator = LogicalValidator()

        # Attribute ranges from config
        self._attr_ranges: Dict[str, AttributeRange] = {
            name: AttributeRange(
                min_val=int(arc.min_val),
                max_val=int(arc.max_val),
                unit=arc.unit,
            )
            for name, arc in config.attribute_ranges.items()
        }

        # Counters for reporting
        self._record_counter: int = 0
        self._rejections: Dict[str, int] = defaultdict(int)
        self._attempts:   Dict[str, int] = defaultdict(int)

    # ── Public entry point ────────────────────────────────────────────────

    def generate(self) -> Tuple[List[LogicalRecord], Dict]:
        """
        Generate all examples for all task families.

        Returns
        -------
        records : list of LogicalRecord (unsplit)
        stats   : dict with rejection / attempt counts per family
        """
        # Master RNG — used only to derive family seeds
        master_rng = random.Random(self.cfg.seed)

        all_records: List[LogicalRecord] = []

        for family_idx, family in enumerate(self.cfg.task_family_order):
            family_seed = master_rng.randint(0, 2**31 - 1)
            records = self._generate_family(family, family_seed)
            all_records.extend(records)

        stats = {
            "rejections_per_family": dict(self._rejections),
            "attempts_per_family":   dict(self._attempts),
            "total_records":         self._record_counter,
        }
        return all_records, stats

    # ── Family-level orchestrator ──────────────────────────────────────────

    def _generate_family(
        self, family: str, family_seed: int
    ) -> List[LogicalRecord]:
        """Generate all examples for one task family."""
        cfg   = self.cfg
        tf    = cfg.task_families[family]

        family_rng = random.Random(family_seed)

        # Build (difficulty, attr, rep) slot list for this family
        slots = self._build_slots(tf, family_rng)   # len = examples_per_family

        records: List[LogicalRecord] = []

        for slot_idx, (difficulty, attr, rep) in enumerate(slots):
            # Seed for this specific example slot
            ex_seed = family_seed ^ (slot_idx * 0x9E3779B9)

            generated = False
            for attempt in range(cfg.max_retries):
                # Fresh RNG for each attempt
                attempt_seed = ex_seed ^ (attempt * 0x6B3A5C4F)
                attempt_rng  = random.Random(attempt_seed)

                self._attempts[family] += 1
                try:
                    candidate = self._dispatch(
                        family, difficulty, attr, rep, attempt_rng, attempt_seed
                    )
                except Exception:
                    self._rejections[family] += 1
                    continue

                if candidate is None:
                    self._rejections[family] += 1
                    continue

                result = self.validator.validate(candidate)
                if result.ok:
                    self._record_counter += 1
                    candidate.id   = f"{self._record_counter:06d}"
                    candidate.seed = attempt_seed
                    records.append(candidate)
                    generated = True
                    break
                else:
                    self._rejections[family] += 1

            if not generated:
                raise GenerationError(
                    f"Could not generate a valid '{family}/{difficulty}' example "
                    f"after {cfg.max_retries} attempts."
                )

        return records

    # ── Slot builder ──────────────────────────────────────────────────────

    def _build_slots(
        self,
        tf,
        rng: random.Random,
    ) -> List[Tuple[str, str, str]]:
        """
        Build a shuffled list of (difficulty, attribute, representation) slots
        for one task family.

        Difficulty slots:
            easy   × tf.easy
            medium × tf.medium
            hard   × tf.hard

        Attribute/representation distribution is computed proportionally so
        that the exact counts (250 per attribute, 750/250 rel/num) are hit
        when tf.total == 1000, while still working for smaller totals used
        in smoke tests.

        For tf.total == 1000 the exact target distribution is:
            height   relational 188, numerical 62
            age      relational 188, numerical 62
            price    relational 187, numerical 63
            quantity relational 187, numerical 63
        """
        n = tf.total

        # ── Difficulty slots ─────────────────────────────────────────────
        diff_slots: List[str] = (
            ["easy"]   * tf.easy   +
            ["medium"] * tf.medium +
            ["hard"]   * tf.hard
        )

        # ── Attribute × representation slots ─────────────────────────────
        # Target proportions per attribute: each gets n // 4 (+ remainder)
        # Numerical proportion: n // 4  per family
        ATTRS = ["height", "age", "price", "quantity"]
        n_attrs = len(ATTRS)

        # Distribute n items across 4 attributes (some get 1 extra)
        base_per_attr = n // n_attrs
        extras = n % n_attrs                     # first `extras` attrs get +1

        # Distribute numericals across the family
        n_num   = n // 4                         # ~25% numerical
        n_rel   = n - n_num                      # ~75% relational

        # Split numerical budget across attributes as evenly as possible
        base_num_per_attr = n_num // n_attrs
        num_extras        = n_num % n_attrs      # first `num_extras` attrs get +1

        ar_slots: List[Tuple[str, str]] = []
        for i, attr in enumerate(ATTRS):
            attr_total = base_per_attr + (1 if i < extras else 0)
            attr_num   = base_num_per_attr + (1 if i < num_extras else 0)
            attr_rel   = attr_total - attr_num
            ar_slots.extend([(attr, "relational")] * attr_rel)
            ar_slots.extend([(attr, "numerical")]  * attr_num)

        assert len(ar_slots) == n, (
            f"ar_slots length {len(ar_slots)} ≠ family total {n}"
        )

        # ── Shuffle both lists independently ─────────────────────────────
        rng.shuffle(diff_slots)
        rng.shuffle(ar_slots)

        return [
            (d, a, r)
            for (d, (a, r)) in zip(diff_slots, ar_slots)
        ]

    # ── Dispatch to sub-generator ──────────────────────────────────────────

    def _dispatch(
        self,
        family:     str,
        difficulty: str,
        attr:       str,
        rep:        str,
        rng:        random.Random,
        seed_val:   int,
    ) -> Optional[LogicalRecord]:
        _map = {
            "direct_comparison":     self._gen_direct_comparison,
            "largest_smallest":      self._gen_largest_smallest,
            "general_comparison":    self._gen_general_comparison,
            "two_hop_transitivity":  self._gen_two_hop_transitivity,
            "three_hop_transitivity":self._gen_three_hop_transitivity,
            "reversed_relation":     self._gen_reversed_relation,
            "mixed_equality":        self._gen_mixed_equality,
        }
        fn = _map[family]
        return fn(difficulty, attr, rep, rng, seed_val)

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-generators
    # ─────────────────────────────────────────────────────────────────────────

    # ── Task 1: direct_comparison ──────────────────────────────────────────

    def _gen_direct_comparison(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> LogicalRecord:
        """
        Easy  : A > B  →  compare(A, B)  [stated direction]
        Medium: A > B  →  compare(B, A)  [reversed direction]
        """
        eids = self.ep.get_ids(2)
        E0, E1 = eids

        # Strict ordering: v0 ≠ v1
        rel = rng.choice(["greater_than", "less_than", "equal"])
        if rel == "equal":
            v0 = rng.randint(1, 20)
            v1 = v0
        elif rel == "greater_than":
            v1 = rng.randint(1, 19)
            v0 = rng.randint(v1 + 1, 20)
        else:
            v0 = rng.randint(1, 19)
            v1 = rng.randint(v0 + 1, 20)

        latent = [float(v0), float(v1)]
        latent_map = {E0: latent[0], E1: latent[1]}

        fact = Fact(E0, rel, E1)
        facts = [fact]

        if difficulty == "easy":
            question = build_compare_question(E0, E1)
            gt = compute_compare_ground_truth(latent_map, E0, E1)
            path = [E0, E1]
        else:  # medium — reversed direction
            question = build_compare_question(E1, E0)
            gt = compute_compare_ground_truth(latent_map, E1, E0)
            path = [E1, E0]

        return _build_record(
            task_family="direct_comparison",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )

    # ── Task 2: largest_smallest ───────────────────────────────────────────

    def _gen_largest_smallest(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> Optional[LogicalRecord]:
        """
        Easy  : 3 entities, A > B > C, facts in order.
        Medium: 3 or 4 entities, shuffled/flipped facts.
        Hard  : 4 entities A > B = C > D (P10), always unique extremes.
        """
        if difficulty == "easy":
            eids = self.ep.get_ids(3)
            E0, E1, E2 = eids
            latent = [3.0, 2.0, 1.0]
            facts = [
                Fact(E0, Relation.GREATER_THAN.value, E1),
                Fact(E1, Relation.GREATER_THAN.value, E2),
            ]
            q_type = rng.choice(["largest", "smallest"])
            question = (build_largest_question(eids) if q_type == "largest"
                        else build_smallest_question(eids))
            latent_map = dict(zip(eids, latent))
            gt = compute_extreme_ground_truth(latent_map, eids, q_type)
            if gt is None:
                return None
            path = list(eids)

        elif difficulty == "medium":
            n = rng.choice([3, 4])
            eids = self.ep.get_ids(n)
            latent = [float(n - i) for i in range(n)]  # e.g. [3,2,1] or [4,3,2,1]
            # Canonical facts: strict descending chain
            canonical = [
                Fact(eids[i], Relation.GREATER_THAN.value, eids[i + 1])
                for i in range(n - 1)
            ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            q_type = rng.choice(["largest", "smallest"])
            question = (build_largest_question(eids) if q_type == "largest"
                        else build_smallest_question(eids))
            latent_map = dict(zip(eids, latent))
            gt = compute_extreme_ground_truth(latent_map, eids, q_type)
            if gt is None:
                return None
            path = list(eids)

        else:  # hard — 4 entities with equality, P10: A>B=C>D
            eids = self.ep.get_ids(4)
            E0, E1, E2, E3 = eids
            latent = [4.0, 2.0, 2.0, 1.0]   # E0>E1=E2>E3  unique max=E0, min=E3
            canonical = [
                Fact(E0, Relation.GREATER_THAN.value, E1),
                Fact(E1, Relation.EQUAL.value,         E2),
                Fact(E2, Relation.GREATER_THAN.value, E3),
            ]
            # Allow equivalent flipped forms and shuffle
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            q_type = rng.choice(["largest", "smallest"])
            question = (build_largest_question(eids) if q_type == "largest"
                        else build_smallest_question(eids))
            latent_map = dict(zip(eids, latent))
            gt = compute_extreme_ground_truth(latent_map, eids, q_type)
            if gt is None:
                return None
            path = list(eids)

        return _build_record(
            task_family="largest_smallest",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )

    # ── Task 3: general_comparison ─────────────────────────────────────────

    def _gen_general_comparison(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> LogicalRecord:
        """
        Easy  : 2 entities, direct fact, compare in natural direction.
        Medium: 3 entities, 2-hop chain (various orientations).
        Hard  : 4 entities, 3-hop chain (with possible equality).
        """
        if difficulty == "easy":
            eids = self.ep.get_ids(2)
            E0, E1 = eids
            rel = rng.choice(["greater_than", "less_than", "equal"])
            if rel == "equal":
                v0 = rng.randint(1, 20)
                v1 = v0
            elif rel == "greater_than":
                v1 = rng.randint(1, 19)
                v0 = rng.randint(v1 + 1, 20)
            else:
                v0 = rng.randint(1, 19)
                v1 = rng.randint(v0 + 1, 20)
            latent = [float(v0), float(v1)]
            latent_map = {E0: latent[0], E1: latent[1]}
            facts = [Fact(E0, rel, E1)]
            question = build_compare_question(E0, E1)
            gt = compute_compare_ground_truth(latent_map, E0, E1)
            path = [E0, E1]

        elif difficulty == "medium":
            eids = self.ep.get_ids(3)
            E0, E1, E2 = eids
            q_rel = rng.choice(["greater_than", "less_than", "equal"])
            if q_rel == "greater_than":
                latent = [3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
            elif q_rel == "less_than":
                latent = [1.0, 2.0, 3.0]
                canonical = [
                    Fact(E0, Relation.LESS_THAN.value, E1),
                    Fact(E1, Relation.LESS_THAN.value, E2),
                ]
            else:
                latent = [2.0, 2.0, 2.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value, E1),
                    Fact(E1, Relation.EQUAL.value, E2),
                ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            latent_map = dict(zip(eids, latent))
            # Query non-adjacent pair (2-hop)
            question = build_compare_question(E0, E2)
            gt = compute_compare_ground_truth(latent_map, E0, E2)
            path = [E0, E1, E2]

        else:  # hard — 4 entities, 3-hop
            eids = self.ep.get_ids(4)
            E0, E1, E2, E3 = eids
            q_rel = rng.choice(["greater_than", "less_than", "equal"])
            if q_rel == "greater_than":
                if use_eq:
                    latent = [4.0, 2.0, 2.0, 1.0]   # P10: A>B=C>D
                    canonical = [
                        Fact(E0, Relation.GREATER_THAN.value, E1),
                        Fact(E1, Relation.EQUAL.value,         E2),
                        Fact(E2, Relation.GREATER_THAN.value, E3),
                    ]
                else:
                    latent = [4.0, 3.0, 2.0, 1.0]   # P08: A>B>C>D
                    canonical = [
                        Fact(E0, Relation.GREATER_THAN.value, E1),
                        Fact(E1, Relation.GREATER_THAN.value, E2),
                        Fact(E2, Relation.GREATER_THAN.value, E3),
                    ]
            elif q_rel == "less_than":
                if use_eq:
                    latent = [1.0, 2.0, 2.0, 4.0]
                    canonical = [
                        Fact(E0, Relation.LESS_THAN.value, E1),
                        Fact(E1, Relation.EQUAL.value,         E2),
                        Fact(E2, Relation.LESS_THAN.value, E3),
                    ]
                else:
                    latent = [1.0, 2.0, 3.0, 4.0]
                    canonical = [
                        Fact(E0, Relation.LESS_THAN.value, E1),
                        Fact(E1, Relation.LESS_THAN.value, E2),
                        Fact(E2, Relation.LESS_THAN.value, E3),
                    ]
            else:
                latent = [2.0, 2.0, 2.0, 2.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value, E1),
                    Fact(E1, Relation.EQUAL.value, E2),
                    Fact(E2, Relation.EQUAL.value, E3),
                ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            latent_map = dict(zip(eids, latent))
            question = build_compare_question(E0, E3)
            gt = compute_compare_ground_truth(latent_map, E0, E3)
            path = list(eids)

        return _build_record(
            task_family="general_comparison",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )

    # ── Task 4: two_hop_transitivity ───────────────────────────────────────

    def _gen_two_hop_transitivity(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> LogicalRecord:
        """
        Easy  : clean A>B>C, compare(A,C).
        Medium: 2-hop chain with possible equality & orientation variation.
        Hard  : 2-hop with reversed query and/or equality.
        """
        eids = self.ep.get_ids(3)
        E0, E1, E2 = eids

        if difficulty == "easy":
            latent = [3.0, 2.0, 1.0]
            facts = [
                Fact(E0, Relation.GREATER_THAN.value, E1),
                Fact(E1, Relation.GREATER_THAN.value, E2),
            ]
            question = build_compare_question(E0, E2)
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, E0, E2)
            path = [E0, E1, E2]

        elif difficulty == "medium":
            chain_type = rng.choice(["strict", "eq_mid", "eq_start"])
            if chain_type == "strict":
                latent = [3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
            elif chain_type == "eq_mid":   # A > B = C
                latent = [3.0, 1.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                ]
            else:                          # A = B > C
                latent = [2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            latent_map = dict(zip(eids, latent))
            question = build_compare_question(E0, E2)
            gt = compute_compare_ground_truth(latent_map, E0, E2)
            path = [E0, E1, E2]

        else:  # hard — forward query but with less_than logic
            chain_type = rng.choice(["strict", "eq_mid", "eq_start"])
            if chain_type == "strict":
                latent = [1.0, 2.0, 3.0]
                canonical = [
                    Fact(E0, Relation.LESS_THAN.value, E1),
                    Fact(E1, Relation.LESS_THAN.value, E2),
                ]
            elif chain_type == "eq_mid":
                latent = [1.0, 3.0, 3.0]
                canonical = [
                    Fact(E0, Relation.LESS_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                ]
            else:
                latent = [2.0, 2.0, 3.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.LESS_THAN.value, E2),
                ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            latent_map = dict(zip(eids, latent))
            question = build_compare_question(E0, E2)
            gt = compute_compare_ground_truth(latent_map, E0, E2)
            path = [E0, E1, E2]

        return _build_record(
            task_family="two_hop_transitivity",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )

    # ── Task 5: three_hop_transitivity ─────────────────────────────────────

    def _gen_three_hop_transitivity(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> LogicalRecord:
        """
        Medium: clean A>B>C>D, compare(A,D), facts in order.
        Hard  : reordered facts / reversed query / equality in chain.
        (No easy examples.)
        """
        eids = self.ep.get_ids(4)
        E0, E1, E2, E3 = eids

        if difficulty == "medium":
            latent = [4.0, 3.0, 2.0, 1.0]
            facts = [
                Fact(E0, Relation.GREATER_THAN.value, E1),
                Fact(E1, Relation.GREATER_THAN.value, E2),
                Fact(E2, Relation.GREATER_THAN.value, E3),
            ]
            question = build_compare_question(E0, E3)
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, E0, E3)
            path = list(eids)

        else:  # hard
            hard_type = rng.choice([
                "shuffled",      # A>B>C>D, shuffled presentation
                "eq_mid",        # A>B=C>D, query A vs D
                "eq_start",      # A=B>C>D, query A vs D
                "eq_end",        # A>B>C=D, query A vs D
            ])

            if hard_type == "shuffled":
                latent = [4.0, 3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
                facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
                question = build_compare_question(E0, E3)
                latent_map = dict(zip(eids, latent))
                gt = compute_compare_ground_truth(latent_map, E0, E3)
                path = list(eids)


            elif hard_type == "eq_mid":     # A>B=C>D  (P10)
                latent = [4.0, 2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
                facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
                question = build_compare_question(E0, E3)
                latent_map = dict(zip(eids, latent))
                gt = compute_compare_ground_truth(latent_map, E0, E3)
                path = list(eids)

            elif hard_type == "eq_start":   # A=B>C>D  (P11)
                latent = [3.0, 3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
                facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
                question = build_compare_question(E0, E3)
                latent_map = dict(zip(eids, latent))
                gt = compute_compare_ground_truth(latent_map, E0, E3)
                path = list(eids)

            else:                           # eq_end: A>B>C=D  (P13)
                latent = [4.0, 3.0, 2.0, 2.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.EQUAL.value,         E3),
                ]
                facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
                question = build_compare_question(E0, E3)
                latent_map = dict(zip(eids, latent))
                gt = compute_compare_ground_truth(latent_map, E0, E3)
                path = list(eids)

        return _build_record(
            task_family="three_hop_transitivity",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )

    # ── Task 6: reversed_relation ──────────────────────────────────────────

    def _gen_reversed_relation(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> LogicalRecord:
        """
        Easy  : A>B, query compare(B,A)        → B<A
        Medium: A>B>C, query compare(C,A)       → C<A
        Hard  : A>B=C>D, query compare(D,A)     → D<A

        The answer is always LESS_THAN (Check 11).
        Check 9 is exempt for all Task 6 difficulties (correction #1).
        """
        if difficulty == "easy":
            eids = self.ep.get_ids(2)
            E0, E1 = eids
            # Always E0 > E1
            latent = [3.0, 1.0]
            facts = [Fact(E0, Relation.GREATER_THAN.value, E1)]
            question = build_compare_question(E1, E0)  # reversed
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, E1, E0)
            path = [E1, E0]

        elif difficulty == "medium":
            eids = self.ep.get_ids(3)
            E0, E1, E2 = eids
            chain_type = rng.choice(["rev_strict", "rev_eq_mid", "rev_eq_start"])
            if chain_type == "rev_strict":
                latent = [3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
            elif chain_type == "rev_eq_mid":
                latent = [3.0, 1.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                ]
            else:
                latent = [2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            question = build_compare_question(E2, E0)  # reversed
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, E2, E0)
            path = [E2, E1, E0]

        else:  # hard — reversed query on 3-hop chain
            eids = self.ep.get_ids(4)
            E0, E1, E2, E3 = eids
            chain_type = rng.choice(["rev_strict", "rev_eq_mid", "rev_eq_start", "rev_eq_end"])
            if chain_type == "rev_strict":
                latent = [4.0, 3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
            elif chain_type == "rev_eq_mid":
                latent = [4.0, 2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
            elif chain_type == "rev_eq_start":
                latent = [3.0, 3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
            else:
                latent = [4.0, 3.0, 2.0, 2.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.EQUAL.value,         E3),
                ]
            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            question = build_compare_question(E3, E0)  # reversed
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, E3, E0)
            path = [E3, E2, E1, E0]

        return _build_record(
            task_family="reversed_relation",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )

    # ── Task 7: mixed_equality ─────────────────────────────────────────────

    def _gen_mixed_equality(
        self, difficulty: str, attr: str, rep: str,
        rng: random.Random, seed_val: int,
    ) -> LogicalRecord:
        """
        Medium: 3-entity chain with one equality fact; equality is the bridge.
        Hard  : 4–5 entity chain with equality; equality is necessary.
        (No easy examples.)

        The equality relevance check (Check 10) is enforced by the validator.
        The generator always constructs examples where equality IS the bridge.
        """
        if difficulty == "medium":
            chain_type = rng.choice([
                "eq_start",       # A=B>C, query A vs C
                "eq_end",         # A>B=C, query A vs C
                "eq_start_rev",   # A=B>C, query C vs A  (reversed)
                "eq_end_rev",     # A>B=C, query C vs A  (reversed)
            ])

            eids = self.ep.get_ids(3)
            E0, E1, E2 = eids

            if chain_type == "eq_start":      # A=B>C: A=B, B>C
                latent = [2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
                question = build_compare_question(E0, E2)
                path = [E0, E1, E2]

            elif chain_type == "eq_end":      # A>B=C: A>B, B=C
                latent = [3.0, 1.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                ]
                question = build_compare_question(E0, E2)
                path = [E0, E1, E2]

            elif chain_type == "eq_start_rev":  # reversed query
                latent = [2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                ]
                question = build_compare_question(E2, E0)
                path = [E2, E1, E0]

            else:                              # eq_end_rev
                latent = [3.0, 1.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                ]
                question = build_compare_question(E2, E0)
                path = [E2, E1, E0]

            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, *question.entities)

        else:  # hard
            hard_type = rng.choice([
                "3hop_eq_mid",        # A>B=C>D, query A vs D
                "3hop_eq_start",      # A=B>C>D, query A vs D
                "3hop_eq_end",        # A>B>C=D, query A vs D
                "3hop_eq_mid_rev",    # A>B=C>D, query D vs A
                "4hop_2eq",           # A=B>C>D=E, query A vs E
            ])

            if hard_type == "3hop_eq_mid":     # P10
                eids = self.ep.get_ids(4)
                E0, E1, E2, E3 = eids
                latent = [4.0, 2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
                question = build_compare_question(E0, E3)
                path = list(eids)

            elif hard_type == "3hop_eq_start":  # P11
                eids = self.ep.get_ids(4)
                E0, E1, E2, E3 = eids
                latent = [3.0, 3.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
                question = build_compare_question(E0, E3)
                path = list(eids)

            elif hard_type == "3hop_eq_end":    # P13
                eids = self.ep.get_ids(4)
                E0, E1, E2, E3 = eids
                latent = [4.0, 3.0, 2.0, 2.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.EQUAL.value,         E3),
                ]
                question = build_compare_question(E0, E3)
                path = list(eids)

            elif hard_type == "3hop_eq_mid_rev":  # P10, reversed
                eids = self.ep.get_ids(4)
                E0, E1, E2, E3 = eids
                latent = [4.0, 2.0, 2.0, 1.0]
                canonical = [
                    Fact(E0, Relation.GREATER_THAN.value, E1),
                    Fact(E1, Relation.EQUAL.value,         E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                ]
                question = build_compare_question(E3, E0)
                path = [E3, E2, E1, E0]

            else:  # 4hop_2eq: A=B>C>D=E  (P15)
                eids = self.ep.get_ids(5)
                E0, E1, E2, E3, E4 = eids
                latent = [3.0, 3.0, 2.0, 1.0, 1.0]
                canonical = [
                    Fact(E0, Relation.EQUAL.value,         E1),
                    Fact(E1, Relation.GREATER_THAN.value, E2),
                    Fact(E2, Relation.GREATER_THAN.value, E3),
                    Fact(E3, Relation.EQUAL.value,         E4),
                ]
                question = build_compare_question(E0, E4)
                path = list(eids)

            facts = _vary_facts(canonical, rng, flip_prob=0.4, force_shuffle=True)
            latent_map = dict(zip(eids, latent))
            gt = compute_compare_ground_truth(latent_map, *question.entities)

        return _build_record(
            task_family="mixed_equality",
            difficulty=difficulty, attribute=attr, representation=rep,
            entity_ids=eids, latent=latent, facts=facts,
            question=question, ground_truth=gt, reasoning_path=path,
            rng=rng, attr_ranges=self._attr_ranges, seed_val=seed_val,
        )
