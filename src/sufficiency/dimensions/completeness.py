"""Completeness dimension: fraction of decisions with confirmed outcome labels.

C(t) = labeled_count / total_count

Confidence interval via Wilson score interval for proportions.
"""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex

from sufficiency.types import DimensionScore


def compute_completeness(
    labeled_count: SupportsIndex,
    total_count: SupportsIndex,
    confidence_level: float = 0.95,
) -> DimensionScore:
    """Compute completeness score with Wilson confidence interval.

    Args:
        labeled_count: Number of decisions with confirmed outcome labels.
        total_count: Total number of decisions in the monitoring window.
        confidence_level: Confidence level for the interval (default 0.95).

    Returns
    -------
        DimensionScore with value = labeled_count / total_count.

    Raises
    ------
        ValueError: If counts are negative or labeled > total.
    """
    labeled = _coerce_count("labeled_count", labeled_count)
    total = _coerce_count("total_count", total_count)

    if labeled < 0 or total < 0:
        msg = f"Counts must be non-negative, got labeled={labeled}, total={total}"
        raise ValueError(msg)
    if labeled > total:
        msg = f"labeled_count ({labeled}) cannot exceed total_count ({total})"
        raise ValueError(msg)
    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(msg)

    if total == 0:
        return DimensionScore(
            value=0.0,
            confidence_low=0.0,
            confidence_high=0.0,
            label="completeness",
        )

    p = labeled / total
    ci_low, ci_high = _wilson_interval(p, total, confidence_level)

    return DimensionScore(
        value=p,
        confidence_low=max(0.0, ci_low),
        confidence_high=min(1.0, ci_high),
        label="completeness",
    )


def _coerce_count(name: str, value: SupportsIndex) -> int:
    """Coerce an integer-like count to int while rejecting bools and floats."""
    if isinstance(value, bool):
        msg = f"{name} must be an integer-like count, got {value}"
        raise TypeError(msg)

    try:
        return operator.index(value)
    except TypeError as exc:
        msg = f"{name} must be an integer-like count, got {value}"
        raise TypeError(msg) from exc


def _wilson_interval(p: float, n: int, confidence_level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than normal approximation for small n or extreme p.
    """
    # z-score for two-tailed confidence level
    alpha = 1.0 - confidence_level
    # Use inverse normal approximation
    z = _normal_quantile(1.0 - alpha / 2.0)
    z2 = z * z

    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator

    return center - spread, center + spread


def _normal_quantile(p: float) -> float:
    """Approximate inverse normal CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0.0 or p >= 1.0:
        msg = f"p must be in (0, 1), got {p}"
        raise ValueError(msg)

    # Rational approximation for 0.5 < p < 1
    if p < 0.5:
        return -_normal_quantile(1.0 - p)

    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
