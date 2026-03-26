"""Tests for core data types."""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np
import pytest

from sufficiency.types import (
    DimensionScore,
    DriftSpec,
    DriftType,
    GovernanceConfig,
    SufficiencyStatus,
    SufficiencyThresholds,
)


class TestDimensionScore:
    def test_valid_score(self) -> None:
        s = DimensionScore(value=0.75, confidence_low=0.7, confidence_high=0.8, label="test")
        assert s.value == 0.75

    def test_boundary_scores(self) -> None:
        DimensionScore(value=0.0, confidence_low=0.0, confidence_high=0.0, label="zero")
        DimensionScore(value=1.0, confidence_low=1.0, confidence_high=1.0, label="one")

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            DimensionScore(value=1.1, confidence_low=0.0, confidence_high=1.0, label="bad")
        with pytest.raises(ValueError, match="must be in"):
            DimensionScore(value=-0.1, confidence_low=0.0, confidence_high=1.0, label="bad")

    def test_rejects_inverted_ci(self) -> None:
        with pytest.raises(ValueError, match="confidence_low must be"):
            DimensionScore(value=0.5, confidence_low=0.8, confidence_high=0.2, label="bad")

    def test_rejects_confidence_low_below_zero(self) -> None:
        with pytest.raises(ValueError, match="confidence_low must be in"):
            DimensionScore(value=0.5, confidence_low=-0.1, confidence_high=0.9, label="bad")

    def test_rejects_confidence_high_above_one(self) -> None:
        with pytest.raises(ValueError, match="confidence_high must be in"):
            DimensionScore(value=0.5, confidence_low=0.1, confidence_high=1.1, label="bad")

    def test_rejects_non_finite_value(self) -> None:
        with pytest.raises(ValueError, match="Dimension score must be in"):
            DimensionScore(value=math.nan, confidence_low=0.1, confidence_high=0.9, label="bad")


class TestSufficiencyThresholds:
    def test_default_thresholds(self) -> None:
        t = SufficiencyThresholds()
        assert t.sufficient == 0.8
        assert t.degraded == 0.5

    def test_classify_sufficient(self) -> None:
        t = SufficiencyThresholds()
        assert t.classify(0.85) == SufficiencyStatus.SUFFICIENT
        assert t.classify(0.8) == SufficiencyStatus.SUFFICIENT

    def test_classify_degraded(self) -> None:
        t = SufficiencyThresholds()
        assert t.classify(0.5) == SufficiencyStatus.DEGRADED
        assert t.classify(0.79) == SufficiencyStatus.DEGRADED

    def test_classify_insufficient(self) -> None:
        t = SufficiencyThresholds()
        assert t.classify(0.49) == SufficiencyStatus.INSUFFICIENT
        assert t.classify(0.0) == SufficiencyStatus.INSUFFICIENT

    def test_rejects_invalid_ordering(self) -> None:
        with pytest.raises(ValueError):
            SufficiencyThresholds(sufficient=0.3, degraded=0.7)

    def test_rejects_non_finite_thresholds(self) -> None:
        with pytest.raises(ValueError, match="Thresholds must be finite"):
            SufficiencyThresholds(sufficient=math.nan)


class TestGovernanceConfig:
    def test_default_config(self) -> None:
        c = GovernanceConfig()
        assert abs(sum(c.weights.values()) - 1.0) < 1e-6
        assert c.tau_c == 0.6
        assert c.tau_r == 0.7

    def test_weights_are_immutable_snapshot(self) -> None:
        weights = {
            "completeness": 0.25,
            "freshness": 0.25,
            "reliability": 0.25,
            "representativeness": 0.25,
        }

        config = GovernanceConfig(weights=weights)

        assert isinstance(config.weights, MappingProxyType)
        weights["freshness"] = 0.5
        assert config.weights["freshness"] == 0.25
        with pytest.raises(TypeError):
            config.weights["freshness"] = 0.5

    def test_rejects_wrong_weight_keys(self) -> None:
        with pytest.raises(ValueError, match="Weights must have keys"):
            GovernanceConfig(weights={"a": 0.5, "b": 0.5})

    def test_rejects_weights_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            GovernanceConfig(
                weights={
                    "completeness": 0.3,
                    "freshness": 0.3,
                    "reliability": 0.3,
                    "representativeness": 0.3,
                }
            )

    def test_rejects_invalid_tau_c(self) -> None:
        with pytest.raises(ValueError, match="tau_c"):
            GovernanceConfig(tau_c=0.0)
        with pytest.raises(ValueError, match="tau_c"):
            GovernanceConfig(tau_c=1.5)

    def test_rejects_invalid_tau_r(self) -> None:
        with pytest.raises(ValueError, match="tau_r"):
            GovernanceConfig(tau_r=0.0)
        with pytest.raises(ValueError, match="tau_r"):
            GovernanceConfig(tau_r=1.5)

    def test_rejects_nonpositive_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_freshness"):
            GovernanceConfig(lambda_freshness=0.0)
        with pytest.raises(ValueError, match="lambda_freshness"):
            GovernanceConfig(lambda_freshness=-1.0)

    def test_rejects_nonpositive_ks_cap(self) -> None:
        with pytest.raises(ValueError, match="ks_cap"):
            GovernanceConfig(ks_cap=0.0)
        with pytest.raises(ValueError, match="ks_cap"):
            GovernanceConfig(ks_cap=-0.5)

    def test_rejects_non_finite_weight(self) -> None:
        with pytest.raises(ValueError, match=r"Weight completeness must be finite and in \[0, 1\]"):
            GovernanceConfig(
                weights={
                    "completeness": math.nan,
                    "freshness": 0.25,
                    "reliability": 0.25,
                    "representativeness": 0.25,
                }
            )

    def test_rejects_negative_weight_even_if_sum_is_one(self) -> None:
        with pytest.raises(ValueError, match=r"Weight completeness must be finite and in \[0, 1\]"):
            GovernanceConfig(
                weights={
                    "completeness": -0.1,
                    "freshness": 0.3,
                    "reliability": 0.4,
                    "representativeness": 0.4,
                }
            )

    def test_rejects_non_finite_tau_c(self) -> None:
        with pytest.raises(ValueError, match="tau_c"):
            GovernanceConfig(tau_c=math.nan)


class TestDriftSpec:
    def test_valid_drift(self) -> None:
        d = DriftSpec(drift_type=DriftType.COVARIATE, magnitude=0.5, onset_day=10)
        assert d.magnitude == 0.5

    def test_rejects_negative_magnitude(self) -> None:
        with pytest.raises(ValueError):
            DriftSpec(drift_type=DriftType.COVARIATE, magnitude=-0.1)

    def test_rejects_negative_onset(self) -> None:
        with pytest.raises(ValueError):
            DriftSpec(drift_type=DriftType.COVARIATE, onset_day=-1)

    def test_accepts_numpy_integer_onset(self) -> None:
        d = DriftSpec(drift_type=DriftType.COVARIATE, onset_day=np.int64(3))
        assert d.onset_day == 3

    def test_rejects_float_onset(self) -> None:
        with pytest.raises(TypeError, match="onset_day must be an integer-like value"):
            DriftSpec(drift_type=DriftType.COVARIATE, onset_day=1.5)

    def test_rejects_boolean_onset(self) -> None:
        with pytest.raises(TypeError, match="onset_day must be an integer-like value"):
            DriftSpec(drift_type=DriftType.COVARIATE, onset_day=True)
