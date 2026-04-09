"""Core data types for evidence sufficiency scoring.

All types are frozen dataclasses — immutable value objects.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from sufficiency._dimensions import REQUIRED_DIMENSIONS, default_weights
from sufficiency._validation import require_unit_interval

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


class SufficiencyStatus(Enum):
    """Governance sufficiency classification."""

    SUFFICIENT = "sufficient"  # S(t) >= 0.8
    DEGRADED = "degraded"  # 0.5 <= S(t) < 0.8
    INSUFFICIENT = "insufficient"  # S(t) < 0.5


class DriftType(Enum):
    """Three drift types with differentiated impact on evidence dimensions."""

    COVARIATE = "covariate"  # P(X) shift
    REAL_CONCEPT = "real_concept"  # P(Y|X) shift
    PRIOR_PROBABILITY = "prior_probability"  # P(Y) shift


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single evidence quality dimension.

    Attributes
    ----------
        value: Normalized score in [0.0, 1.0].
        confidence_low: Lower bound of confidence interval.
        confidence_high: Upper bound of confidence interval.
        label: Dimension name (completeness, freshness, reliability, representativeness).
    """

    value: float
    confidence_low: float
    confidence_high: float
    label: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not 0.0 <= self.value <= 1.0:
            msg = f"Dimension score must be in [0, 1], got {self.value}"
            raise ValueError(msg)
        require_unit_interval("confidence_low", self.confidence_low)
        require_unit_interval("confidence_high", self.confidence_high)
        if self.confidence_low > self.confidence_high:
            msg = "confidence_low must be <= confidence_high"
            raise ValueError(msg)


@dataclass(frozen=True)
class SufficiencyThresholds:
    """Governance thresholds for sufficiency classification.

    Attributes
    ----------
        sufficient: Minimum score for sufficient evidence (default 0.8).
        degraded: Minimum score for degraded (vs insufficient) evidence (default 0.5).
    """

    sufficient: float = 0.8
    degraded: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.sufficient) or not math.isfinite(self.degraded):
            msg = f"Thresholds must be finite, got {self.degraded}, {self.sufficient}"
            raise ValueError(msg)
        if not 0.0 < self.degraded < self.sufficient <= 1.0:
            msg = f"Need 0 < degraded < sufficient <= 1, got {self.degraded}, {self.sufficient}"
            raise ValueError(msg)

    def classify(self, score: float) -> SufficiencyStatus:
        """Classify a composite score into a governance status."""
        if score >= self.sufficient:
            return SufficiencyStatus.SUFFICIENT
        if score >= self.degraded:
            return SufficiencyStatus.DEGRADED
        return SufficiencyStatus.INSUFFICIENT


@dataclass(frozen=True)
class GovernanceConfig:
    """Configuration for a governance context.

    Defines weights, thresholds, and temporal parameters for sufficiency scoring.
    Weights must sum to 1.0. Default: equal weights (0.25 each).

    Attributes
    ----------
        weights: Per-dimension weights {completeness, freshness, reliability,
            representativeness}. Must sum to 1.0.
        tau_c: Completeness threshold for decision-readiness gate (default 0.6).
        tau_r: Reliability threshold for decision-readiness gate (default 0.7).
        lambda_freshness: Freshness exponential decay rate per day (default 0.02).
        ks_cap: KS divergence cap for representativeness normalization (default 0.30).
        thresholds: Governance sufficiency thresholds.
    """

    weights: Mapping[str, float] = field(default_factory=default_weights)
    tau_c: float = 0.6
    tau_r: float = 0.7
    lambda_freshness: float = 0.02
    ks_cap: float = 0.30
    thresholds: SufficiencyThresholds = field(default_factory=SufficiencyThresholds)

    _REQUIRED_DIMENSIONS: frozenset[str] = field(
        init=False,
        repr=False,
        default=REQUIRED_DIMENSIONS,
    )

    def __post_init__(self) -> None:
        weights = dict(self.weights)
        required_dimensions = set(self._REQUIRED_DIMENSIONS)

        if set(weights.keys()) != required_dimensions:
            got = set(weights.keys())
            msg = f"Weights must have keys {self._REQUIRED_DIMENSIONS}, got {got}"
            raise ValueError(msg)
        for name in required_dimensions:
            weight = weights[name]
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                msg = f"Weight {name} must be finite and in [0, 1], got {weight}"
                raise ValueError(msg)
        total = sum(weights.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
            msg = f"Weights must sum to 1.0, got {total}"
            raise ValueError(msg)
        if not math.isfinite(self.tau_c) or self.tau_c <= 0 or self.tau_c > 1:
            msg = f"tau_c must be in (0, 1], got {self.tau_c}"
            raise ValueError(msg)
        if not math.isfinite(self.tau_r) or self.tau_r <= 0 or self.tau_r > 1:
            msg = f"tau_r must be in (0, 1], got {self.tau_r}"
            raise ValueError(msg)
        if not math.isfinite(self.lambda_freshness) or self.lambda_freshness <= 0:
            msg = f"lambda_freshness must be positive, got {self.lambda_freshness}"
            raise ValueError(msg)
        if not math.isfinite(self.ks_cap) or self.ks_cap <= 0:
            msg = f"ks_cap must be positive, got {self.ks_cap}"
            raise ValueError(msg)
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass(frozen=True)
class SufficiencyResult:
    """Complete sufficiency assessment at a point in time.

    Attributes
    ----------
        composite: Composite sufficiency score S(t) in [0, 1].
        gate: Decision-readiness gate value A(t) in [0, 1].
        dimensions: Per-dimension scores keyed by dimension name.
        status: Governance classification (sufficient/degraded/insufficient).
        timestamp: Assessment time.
    """

    composite: float
    gate: float
    dimensions: Mapping[str, DimensionScore]
    status: SufficiencyStatus
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))


@dataclass(frozen=True)
class DriftSpec:
    """Specification for a drift event in blind period simulation.

    Attributes
    ----------
        drift_type: Type of drift (covariate, real_concept, prior_probability).
        magnitude: Drift magnitude in [0.0, 1.0] (1.0 = maximum severity).
        onset_day: Day within the blind period when drift begins.
    """

    drift_type: DriftType
    magnitude: float = 0.5
    onset_day: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnitude) or not 0.0 <= self.magnitude <= 1.0:
            msg = f"Drift magnitude must be in [0, 1], got {self.magnitude}"
            raise ValueError(msg)
        onset_day = _coerce_nonnegative_index("onset_day", self.onset_day)
        if onset_day < 0:
            msg = f"onset_day must be non-negative, got {onset_day}"
            raise ValueError(msg)
        object.__setattr__(self, "onset_day", onset_day)


def _coerce_nonnegative_index(name: str, value: object) -> int:
    """Coerce an integer-like value to int while rejecting bools and floats."""
    if isinstance(value, bool):
        msg = f"{name} must be an integer-like value, got {value}"
        raise TypeError(msg)

    try:
        return operator.index(cast("Any", value))
    except TypeError as exc:
        msg = f"{name} must be an integer-like value, got {value}"
        raise TypeError(msg) from exc
