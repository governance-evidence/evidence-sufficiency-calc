"""Evidence sufficiency scoring for governance assessments under delayed ground truth.

The package root exposes the stable public API for core scoring, configuration,
and threshold-based monitoring.

Experimental monitoring primitives live under
``sufficiency.experimental.monitoring``.

Decision Event Schema helpers live under ``sufficiency.adapters.des`` rather
than the stable root API surface.
"""

from __future__ import annotations

from sufficiency.blind_period import BlindPeriodSimulator
from sufficiency.composite import compute_sufficiency
from sufficiency.config import credit_scoring_config, default_config, fraud_detection_config
from sufficiency.gate import compute_gate
from sufficiency.sequential import ThresholdMonitor
from sufficiency.types import (
    DimensionScore,
    DriftSpec,
    DriftType,
    GovernanceConfig,
    SufficiencyResult,
    SufficiencyStatus,
    SufficiencyThresholds,
)

__version__ = "0.1.0"
ROOT_API_STABILITY = "stable"

__all__ = [
    "BlindPeriodSimulator",
    "DimensionScore",
    "DriftSpec",
    "DriftType",
    "GovernanceConfig",
    "SufficiencyResult",
    "SufficiencyStatus",
    "SufficiencyThresholds",
    "ThresholdMonitor",
    "compute_gate",
    "compute_sufficiency",
    "credit_scoring_config",
    "default_config",
    "fraud_detection_config",
]
