"""Integration tests for end-to-end sufficiency scoring."""

from __future__ import annotations

import numpy as np

from sufficiency.composite import compute_sufficiency
from sufficiency.config import default_config, fraud_detection_config
from sufficiency.dimensions.completeness import compute_completeness
from sufficiency.dimensions.freshness import compute_freshness
from sufficiency.dimensions.reliability import compute_reliability
from sufficiency.dimensions.representativeness import compute_representativeness
from sufficiency.types import SufficiencyStatus


class TestEndToEndScoring:
    def test_healthy_system(self) -> None:
        """A well-monitored system with recent labels should score sufficient."""
        config = default_config()
        rng = np.random.default_rng(42)

        y_true = rng.integers(0, 2, size=500)
        # Good model: ~85% accuracy
        y_pred = y_true.copy()
        flip_idx = rng.choice(500, size=75, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]

        ref = rng.normal(0, 1, size=1000)
        prod = rng.normal(0.02, 1, size=1000)

        dims = {
            "completeness": compute_completeness(450, 500),
            "freshness": compute_freshness(5.0, config.lambda_freshness),
            "reliability": compute_reliability(y_true, y_pred, rng_seed=42),
            "representativeness": compute_representativeness(ref, prod),
        }

        result = compute_sufficiency(dims, config)
        assert result.status == SufficiencyStatus.SUFFICIENT
        assert result.composite >= 0.8

    def test_stale_evidence_degrades(self) -> None:
        """30-day-old labels in fraud detection should degrade score."""
        config = fraud_detection_config()
        rng = np.random.default_rng(42)

        y_true = rng.integers(0, 2, size=200)
        y_pred = y_true.copy()

        ref = rng.normal(0, 1, size=500)
        prod = rng.normal(0, 1, size=500)

        dims = {
            "completeness": compute_completeness(120, 200),
            "freshness": compute_freshness(30.0, config.lambda_freshness),
            "reliability": compute_reliability(y_true, y_pred, rng_seed=42),
            "representativeness": compute_representativeness(ref, prod),
        }

        result = compute_sufficiency(dims, config)
        # Freshness = 0.55, completeness = 0.6, should pull composite down
        assert result.composite < 0.8

    def test_low_completeness_triggers_gate(self) -> None:
        """Very low completeness should suppress composite via gate."""
        config = default_config()
        rng = np.random.default_rng(42)

        y_true = rng.integers(0, 2, size=100)
        y_pred = y_true.copy()

        ref = rng.normal(0, 1, size=500)
        prod = rng.normal(0, 1, size=500)

        dims = {
            "completeness": compute_completeness(20, 100),  # C=0.2 < tau_c=0.6
            "freshness": compute_freshness(1.0, config.lambda_freshness),
            "reliability": compute_reliability(y_true, y_pred, rng_seed=42),
            "representativeness": compute_representativeness(ref, prod),
        }

        result = compute_sufficiency(dims, config)
        assert result.gate < 1.0
        assert result.status == SufficiencyStatus.INSUFFICIENT
