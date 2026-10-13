"""
attributes.py — Attribute definitions and numeric value sampling.

Four attributes are used: height, age, price, quantity.

For "numerical" representation the generator must assign real numeric values
to entities that respect the same ordering as the latent values.

Key rules:
    • Equal latent values   → identical numeric values (exact equality)
    • Strictly ordered      → strictly different numeric values (no accidental ties)
    • Values stay within the configured range for each attribute
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Attribute definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttributeRange:
    """Numeric range for a single attribute."""
    min_val: int
    max_val: int
    unit: str = ""


# Default ranges — kept in sync with phase3/configs/logical_generator.yaml
DEFAULT_ATTRIBUTE_RANGES: Dict[str, AttributeRange] = {
    "height":   AttributeRange(min_val=140, max_val=200, unit="cm"),
    "age":      AttributeRange(min_val=10,  max_val=80,  unit="years"),
    "price":    AttributeRange(min_val=100, max_val=10_000, unit=""),
    "quantity": AttributeRange(min_val=10,  max_val=1_000, unit=""),
}

ATTRIBUTES: List[str] = list(DEFAULT_ATTRIBUTE_RANGES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Value sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_ordered_values(
    rng: random.Random,
    entity_ids: List[str],
    latent_values: List[float],
    attribute: str,
    attr_ranges: Dict[str, AttributeRange] | None = None,
) -> Dict[str, int]:
    """
    Assign concrete integer values to entities that preserve the latent ordering.

    Parameters
    ----------
    rng          : seeded random.Random instance (caller-controlled)
    entity_ids   : list of entity ID strings, same length as latent_values
    latent_values: rank-like floats; equal values → equal output; higher → higher output
    attribute    : one of "height" | "age" | "price" | "quantity"
    attr_ranges  : optional override for attribute ranges

    Returns
    -------
    Dict mapping entity_id → integer value

    Guarantees
    ----------
    • Entities with equal latent values receive the SAME output value.
    • Entities with different latent values receive DIFFERENT output values,
      with order preserved (higher latent → higher output).
    • All values lie within [range.min_val, range.max_val].
    """
    if attr_ranges is None:
        attr_ranges = DEFAULT_ATTRIBUTE_RANGES
    if attribute not in attr_ranges:
        raise ValueError(f"Unknown attribute '{attribute}'")

    ar = attr_ranges[attribute]
    mn, mx = ar.min_val, ar.max_val

    # Unique latent levels sorted ascending: e.g. [1, 2, 4] from [4,2,2,1]
    unique_levels = sorted(set(latent_values))
    n_levels = len(unique_levels)

    # We need n_levels distinct integers from [mn, mx]
    available_range = mx - mn + 1
    if n_levels > available_range:
        # Extremely tight range (shouldn't happen with configured ranges)
        # Fall back: use consecutive integers starting at mn
        chosen = list(range(mn, mn + n_levels))
    else:
        # Sample n_levels distinct values, then sort ascending so ordering aligns
        chosen = sorted(rng.sample(range(mn, mx + 1), n_levels))

    # Map each latent level to a chosen value (ascending ↔ ascending)
    level_to_val: Dict[float, int] = {
        level: val for level, val in zip(unique_levels, chosen)
    }

    return {eid: level_to_val[lv] for eid, lv in zip(entity_ids, latent_values)}
