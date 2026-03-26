"""Composite sufficiency scoring.

S(t) = A(t) * [w_c * C(t) + w_f * F(t) + w_r * R(t) + w_p * P(t)]

Combines four dimension scores through a weighted sum modulated by
the decision-readiness gate.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sufficiency._dimensions import COMPLETENESS, RELIABILITY, REQUIRED_DIMENSIONS
from sufficiency._validation import require_aware_datetime
from sufficiency.gate import compute_gate
from sufficiency.types import DimensionScore, GovernanceConfig, SufficiencyResult


def compute_sufficiency(
    dimensions: dict[str, DimensionScore],
    config: GovernanceConfig,
    timestamp: datetime | None = None,
) -> SufficiencyResult:
    """Compute composite sufficiency score.

    Args:
        dimensions: Per-dimension scores keyed by name. Must contain all four
            required dimensions: completeness, freshness, reliability,
            representativeness.
        config: Governance configuration with weights and thresholds.
        timestamp: Assessment time (default: now UTC).

    Returns
    -------
        SufficiencyResult with composite score, gate, and status.

    Raises
    ------
        ValueError: If required dimensions are missing.
    """
    provided = set(dimensions.keys())
    missing = REQUIRED_DIMENSIONS - provided
    extra = provided - REQUIRED_DIMENSIONS
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        msg = f"Dimension keys must exactly match required set ({', '.join(details)})"
        raise ValueError(msg)

    gate = compute_gate(
        completeness=dimensions[COMPLETENESS].value,
        reliability=dimensions[RELIABILITY].value,
        tau_c=config.tau_c,
        tau_r=config.tau_r,
    )

    weighted_sum = sum(
        config.weights[name] * dimensions[name].value for name in REQUIRED_DIMENSIONS
    )

    composite = gate * weighted_sum
    status = config.thresholds.classify(composite)

    if timestamp is None:
        timestamp = datetime.now(UTC)
    else:
        timestamp = require_aware_datetime(
            timestamp,
            message="timestamp must be timezone-aware",
        )

    return SufficiencyResult(
        composite=composite,
        gate=gate,
        dimensions=dimensions,
        status=status,
        timestamp=timestamp,
    )
