"""Blind period simulation for evidence sufficiency degradation.

Simulates how evidence sufficiency degrades over time when outcome labels
are delayed. Models three drift types with differentiated impact on
each evidence quality dimension (Table 4, Paper 14).

Implementation note
-------------------
This module keeps the public simulator class small and moves the simulation
pipeline into pure helpers. The intended flow is:

1. normalize checkpoint inputs
2. compute baseline freshness and pre-drift dimension values
3. apply drift-specific degradation
4. clamp to the unit interval
5. build immutable ``DimensionScore`` objects

The helpers below are internal-only and exist to keep the numerical model easy
to test and refactor without widening the public API surface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from numbers import Integral

from sufficiency._dimensions import COMPLETENESS, FRESHNESS, RELIABILITY, REPRESENTATIVENESS
from sufficiency._validation import require_unit_interval
from sufficiency.composite import compute_sufficiency
from sufficiency.types import (
    DimensionScore,
    DriftSpec,
    DriftType,
    GovernanceConfig,
    SufficiencyResult,
)


@dataclass(frozen=True)
class _DriftImpact:
    """Per-dimension degradation multipliers for a drift type.

    Values in [0, 1]: 0 = no degradation, 1 = maximum degradation.
    Applied as: dimension(t) = initial * (1 - impact * magnitude * progress)
    """

    completeness: float
    freshness: float  # always 0 — freshness decays independently
    reliability: float
    representativeness: float


# Table 4 from Paper 14 Section 3.2
_DRIFT_IMPACTS: dict[DriftType, _DriftImpact] = {
    DriftType.COVARIATE: _DriftImpact(
        completeness=0.0,
        freshness=0.0,
        reliability=0.2,  # mildly degraded
        representativeness=0.9,  # severely degraded
    ),
    DriftType.REAL_CONCEPT: _DriftImpact(
        completeness=0.0,
        freshness=0.0,
        reliability=0.9,  # severely degraded
        representativeness=0.0,  # unchanged
    ),
    DriftType.PRIOR_PROBABILITY: _DriftImpact(
        completeness=0.6,  # degraded (class composition shifts)
        freshness=0.0,
        reliability=0.5,  # moderately degraded
        representativeness=0.5,  # moderately degraded
    ),
}


@dataclass
class BlindPeriodSimulator:
    """Simulates evidence sufficiency degradation during label delay.

    Attributes
    ----------
        initial_completeness: C(0) at start of blind period.
        initial_reliability: R(0) at start of blind period.
        initial_representativeness: P(0) at start of blind period.
        config: Governance configuration.
        drift_specs: Optional drift events during the blind period.
        start_time: Simulation start time.
    """

    initial_completeness: float = 0.95
    initial_reliability: float = 0.85
    initial_representativeness: float = 0.95
    config: GovernanceConfig = field(default_factory=GovernanceConfig)
    drift_specs: list[DriftSpec] = field(default_factory=list)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        _validate_initial_dimension("initial_completeness", self.initial_completeness)
        _validate_initial_dimension("initial_reliability", self.initial_reliability)
        _validate_initial_dimension("initial_representativeness", self.initial_representativeness)

    def simulate(self, days: list[int] | None = None) -> list[SufficiencyResult]:
        """Run blind period simulation at specified day checkpoints.

        Args:
            days: Days to evaluate (default: [30, 60, 90, 180]).

        Returns
        -------
            SufficiencyResult at each checkpoint day.
        """
        days = _normalize_days(days)

        results: list[SufficiencyResult] = []
        for day in sorted(days):
            dimensions = self._compute_dimensions_at_day(day)
            result = compute_sufficiency(
                dimensions=dimensions,
                config=self.config,
                timestamp=self.start_time + timedelta(days=day),
            )
            results.append(result)

        return results

    def simulate_daily(self, total_days: int) -> list[SufficiencyResult]:
        """Run simulation for every day up to total_days.

        Args:
            total_days: Number of days to simulate.

        Returns
        -------
            Daily SufficiencyResult trajectory.
        """
        _validate_total_days(total_days)

        return self.simulate(list(range(1, total_days + 1)))

    def _compute_dimensions_at_day(self, day: int) -> dict[str, DimensionScore]:
        """Compute all dimension scores at a given day via the helper pipeline."""
        freshness_value = _freshness_at_day(day, self.config.lambda_freshness)
        c_value, r_value, p_value = _baseline_dimension_values(
            day=day,
            initial_completeness=self.initial_completeness,
            initial_reliability=self.initial_reliability,
            initial_representativeness=self.initial_representativeness,
        )
        c_value, r_value, p_value = _apply_drift_impacts(
            day=day,
            c_value=c_value,
            r_value=r_value,
            p_value=p_value,
            drift_specs=self.drift_specs,
        )
        c_value, r_value, p_value = _clamp_dimension_values(c_value, r_value, p_value)

        return _build_dimension_scores(
            c_value=c_value,
            freshness_value=freshness_value,
            r_value=r_value,
            p_value=p_value,
        )


def _normalize_days(days: list[int] | None) -> list[int]:
    """Normalize simulation checkpoints and preserve legacy validation behavior."""
    if days is None:
        return [30, 60, 90, 180]
    if any(not _is_valid_day(day) for day in days):
        msg = f"days must contain only non-negative integers, got {days}"
        raise TypeError(msg)
    if any(day < 0 for day in days):
        msg = f"days must contain only non-negative integers, got {days}"
        raise ValueError(msg)
    return days


def _validate_total_days(total_days: int) -> None:
    """Validate the daily simulation horizon while preserving legacy errors."""
    if not _is_valid_day(total_days):
        msg = f"total_days must be a non-negative integer, got {total_days}"
        raise TypeError(msg)
    if total_days < 0:
        msg = f"total_days must be non-negative, got {total_days}"
        raise ValueError(msg)


def _freshness_at_day(day: int, lambda_freshness: float) -> float:
    """Compute freshness decay independently of drift."""
    return math.exp(-lambda_freshness * day)


def _baseline_dimension_values(
    *,
    day: int,
    initial_completeness: float,
    initial_reliability: float,
    initial_representativeness: float,
) -> tuple[float, float, float]:
    """Return the baseline dimension values before drift-specific degradation."""
    c_value = initial_completeness * max(0.0, 1.0 - 0.005 * day)
    r_value = initial_reliability
    p_value = initial_representativeness
    return c_value, r_value, p_value


def _apply_drift_impacts(
    *,
    day: int,
    c_value: float,
    r_value: float,
    p_value: float,
    drift_specs: list[DriftSpec],
) -> tuple[float, float, float]:
    """Apply all drift-specific degradations for a given day."""
    for drift in drift_specs:
        if day < drift.onset_day:
            continue
        c_value, r_value, p_value = _apply_single_drift(
            day=day,
            c_value=c_value,
            r_value=r_value,
            p_value=p_value,
            drift=drift,
        )
    return c_value, r_value, p_value


def _apply_single_drift(
    *,
    day: int,
    c_value: float,
    r_value: float,
    p_value: float,
    drift: DriftSpec,
) -> tuple[float, float, float]:
    """Apply a single drift event to the current dimension values."""
    impact = _DRIFT_IMPACTS[drift.drift_type]
    progress = _drift_progress(day, drift.onset_day)
    c_value *= 1.0 - impact.completeness * drift.magnitude * progress
    r_value *= 1.0 - impact.reliability * drift.magnitude * progress
    p_value *= 1.0 - impact.representativeness * drift.magnitude * progress
    return c_value, r_value, p_value


def _drift_progress(day: int, onset_day: int) -> float:
    """Return normalized drift progression capped at full effect by day 90."""
    drift_days = day - onset_day
    return min(1.0, drift_days / 90.0)


def _clamp_dimension_values(
    c_value: float,
    r_value: float,
    p_value: float,
) -> tuple[float, float, float]:
    """Clamp mutable dimension values to the closed unit interval."""
    return (
        _clamp_unit_interval(c_value),
        _clamp_unit_interval(r_value),
        _clamp_unit_interval(p_value),
    )


def _build_dimension_scores(
    *,
    c_value: float,
    freshness_value: float,
    r_value: float,
    p_value: float,
) -> dict[str, DimensionScore]:
    """Build the immutable per-dimension score mapping for a simulated day."""
    return {
        COMPLETENESS: _with_relative_confidence(COMPLETENESS, c_value),
        FRESHNESS: DimensionScore(
            value=freshness_value,
            confidence_low=freshness_value,
            confidence_high=freshness_value,
            label=FRESHNESS,
        ),
        RELIABILITY: _with_relative_confidence(RELIABILITY, r_value),
        REPRESENTATIVENESS: _with_relative_confidence(REPRESENTATIVENESS, p_value),
    }


def _with_relative_confidence(label: str, value: float) -> DimensionScore:
    """Construct a score using the simulator's legacy +-10% confidence band."""
    return DimensionScore(
        value=value,
        confidence_low=value * 0.9,
        confidence_high=min(1.0, value * 1.1),
        label=label,
    )


def _clamp_unit_interval(value: float) -> float:
    """Clamp a numeric value to the closed unit interval."""
    return max(0.0, min(1.0, value))


def _is_valid_day(value: object) -> bool:
    """Return whether a day input is an integer-like non-bool value."""
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_initial_dimension(name: str, value: float) -> None:
    """Validate simulator starting values while preserving legacy error text."""
    try:
        require_unit_interval(name, value)
    except ValueError as exc:
        msg = str(exc).replace("must be in", "must be finite and in")
        raise ValueError(msg) from exc
