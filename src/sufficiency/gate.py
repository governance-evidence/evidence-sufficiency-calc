"""Decision-readiness gate A(t).

A(t) = min(1, C/tau_c) * min(1, R/tau_r)

Multiplicative structure: simultaneous degradation of completeness and
reliability produces a compounding penalty. Prevents high freshness and
representativeness from masking inadequate completeness or reliability.
"""

from __future__ import annotations

import math


def compute_gate(
    completeness: float,
    reliability: float,
    tau_c: float = 0.6,
    tau_r: float = 0.7,
) -> float:
    """Compute the decision-readiness gate value.

    Args:
        completeness: Completeness dimension score C(t) in [0, 1].
        reliability: Reliability dimension score R(t) in [0, 1].
        tau_c: Completeness threshold (default 0.6).
        tau_r: Reliability threshold (default 0.7).

    Returns
    -------
        Gate value A(t) in [0, 1]. Above thresholds: A(t) = 1.0.
        Below thresholds: proportional suppression.
    """
    for name, value in (("completeness", completeness), ("reliability", reliability)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            msg = f"{name} must be finite and in [0, 1], got {value}"
            raise ValueError(msg)

    for name, value in (("tau_c", tau_c), ("tau_r", tau_r)):
        if not math.isfinite(value) or value <= 0.0:
            msg = f"{name} must be finite and positive, got {value}"
            raise ValueError(msg)

    c_component = min(1.0, completeness / tau_c)
    r_component = min(1.0, reliability / tau_r)
    return c_component * r_component
