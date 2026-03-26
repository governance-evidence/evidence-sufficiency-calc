"""Freshness dimension: temporal relevance of evidence.

F(t) = exp(-lambda * delta_t)

where delta_t is elapsed time (days) since the median transaction date
of the most recent confirmed labels.
"""

from __future__ import annotations

import math

from sufficiency.types import DimensionScore


def compute_freshness(
    delta_t_days: float,
    lambda_rate: float = 0.02,
    delta_t_std: float | None = None,
) -> DimensionScore:
    """Compute freshness score as exponential decay of label age.

    Args:
        delta_t_days: Elapsed days since median date of most recent confirmed labels.
        lambda_rate: Decay rate per day (default 0.02, calibrated so F=0.55 at 30 days).
        delta_t_std: Optional standard deviation of label ages for confidence interval.
            If None, a tight interval around the point estimate is returned.

    Returns
    -------
        DimensionScore with value = exp(-lambda * delta_t).

    Raises
    ------
        ValueError: If delta_t_days is negative or lambda_rate is non-positive.
    """
    if not math.isfinite(delta_t_days) or delta_t_days < 0:
        msg = f"delta_t_days must be non-negative, got {delta_t_days}"
        raise ValueError(msg)
    if not math.isfinite(lambda_rate) or lambda_rate <= 0:
        msg = f"lambda_rate must be positive, got {lambda_rate}"
        raise ValueError(msg)
    if delta_t_std is not None and (not math.isfinite(delta_t_std) or delta_t_std < 0):
        msg = f"delta_t_std must be finite and non-negative, got {delta_t_std}"
        raise ValueError(msg)

    value = math.exp(-lambda_rate * delta_t_days)

    if delta_t_std is not None and delta_t_std > 0:
        # Propagate uncertainty in delta_t through exponential
        ci_low = math.exp(-lambda_rate * (delta_t_days + delta_t_std))
        ci_high = math.exp(-lambda_rate * max(0.0, delta_t_days - delta_t_std))
    else:
        ci_low = value
        ci_high = value

    return DimensionScore(
        value=value,
        confidence_low=max(0.0, ci_low),
        confidence_high=min(1.0, ci_high),
        label="freshness",
    )
