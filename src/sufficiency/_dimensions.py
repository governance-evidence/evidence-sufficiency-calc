"""Internal constants for evidence dimension names and default weights.

These names are part of the package's semantic model, but this module itself is
not a public API surface. Public callers should continue using the documented
string keys exposed through the stable root API.
"""

from __future__ import annotations

COMPLETENESS = "completeness"
FRESHNESS = "freshness"
RELIABILITY = "reliability"
REPRESENTATIVENESS = "representativeness"

REQUIRED_DIMENSIONS = frozenset({COMPLETENESS, FRESHNESS, RELIABILITY, REPRESENTATIVENESS})


def default_weights() -> dict[str, float]:
    """Return the canonical equal-weight mapping for all dimensions."""
    return {
        COMPLETENESS: 0.25,
        FRESHNESS: 0.25,
        RELIABILITY: 0.25,
        REPRESENTATIVENESS: 0.25,
    }
