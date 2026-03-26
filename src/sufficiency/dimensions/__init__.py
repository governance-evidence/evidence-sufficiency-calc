"""Evidence quality dimension scorers.

Four dimensions: completeness, freshness, reliability, representativeness.
Each module exports a compute_* function returning a DimensionScore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sufficiency.types import DimensionScore


class Dimension(Protocol):
    """Protocol for dimension scoring functions."""

    def __call__(self, **kwargs: object) -> DimensionScore:
        """Compute a dimension score from evidence data."""
        ...  # pragma: no cover
