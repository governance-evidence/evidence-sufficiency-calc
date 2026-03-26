"""Representativeness dimension: coverage of operational distribution.

P(t) = max(0, 1 - KS / KS_cap)

where KS is the two-sample Kolmogorov-Smirnov statistic between
reference and production score distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

from sufficiency._validation import coerce_1d_float_array
from sufficiency.types import DimensionScore


def compute_representativeness(
    reference: ArrayLike,
    production: ArrayLike,
    ks_cap: float = 0.30,
) -> DimensionScore:
    """Compute representativeness from KS divergence.

    Args:
        reference: Reference distribution samples (labeled period scores).
        production: Production distribution samples (current period scores).
        ks_cap: KS divergence cap for normalization (default 0.30).
            Divergence at or above ks_cap maps to score 0.0.

    Returns
    -------
        DimensionScore with value = max(0, 1 - KS/ks_cap).

    Raises
    ------
        ValueError: If either array is empty or ks_cap is non-positive.
    """
    ref_arr = coerce_1d_float_array("reference", reference)
    prod_arr = coerce_1d_float_array("production", production)

    if len(ref_arr) == 0 or len(prod_arr) == 0:
        msg = "Both reference and production arrays must be non-empty"
        raise ValueError(msg)
    if ks_cap <= 0:
        msg = f"ks_cap must be positive, got {ks_cap}"
        raise ValueError(msg)
    if not np.isfinite(ref_arr).all():
        msg = "reference must contain only finite numeric values"
        raise ValueError(msg)
    if not np.isfinite(prod_arr).all():
        msg = "production must contain only finite numeric values"
        raise ValueError(msg)

    if reference is production:
        return _representativeness_score(0.0, len(ref_arr), len(prod_arr), ks_cap)

    ks_result = stats.ks_2samp(ref_arr, prod_arr)
    ks_stat = float(ks_result.statistic)

    return _representativeness_score(ks_stat, len(ref_arr), len(prod_arr), ks_cap)


def _representativeness_score(
    ks_stat: float,
    ref_count: int,
    prod_count: int,
    ks_cap: float,
) -> DimensionScore:
    """Convert a KS statistic into a normalized score and uncertainty interval."""
    value = max(0.0, 1.0 - ks_stat / ks_cap)

    # Confidence interval from KS test critical values
    # Use p-value to estimate uncertainty in the KS statistic
    n_eff = ref_count * prod_count / (ref_count + prod_count)
    # Approximate 95% CI for KS statistic using Dvoretzky-Kiefer-Wolfowitz inequality
    ks_margin = 1.36 / (n_eff**0.5) if n_eff > 0 else 0.5

    ci_high = max(0.0, 1.0 - max(0.0, ks_stat - ks_margin) / ks_cap)
    ci_low = max(0.0, 1.0 - min(ks_cap, ks_stat + ks_margin) / ks_cap)

    return DimensionScore(
        value=value,
        confidence_low=max(0.0, min(1.0, ci_low)),
        confidence_high=max(0.0, min(1.0, ci_high)),
        label="representativeness",
    )
