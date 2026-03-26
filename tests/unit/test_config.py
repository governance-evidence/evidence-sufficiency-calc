"""Tests for configuration factories."""

from __future__ import annotations

from sufficiency.config import credit_scoring_config, default_config, fraud_detection_config


class TestConfigFactories:
    def test_default_has_equal_weights(self) -> None:
        c = default_config()
        for v in c.weights.values():
            assert abs(v - 0.25) < 1e-6

    def test_fraud_emphasizes_freshness_reliability(self) -> None:
        c = fraud_detection_config()
        assert c.weights["freshness"] > c.weights["completeness"]
        assert c.weights["reliability"] > c.weights["completeness"]
        assert c.lambda_freshness == 0.02

    def test_credit_emphasizes_representativeness(self) -> None:
        c = credit_scoring_config()
        assert c.weights["representativeness"] > c.weights["freshness"]
        assert c.lambda_freshness == 0.005

    def test_all_configs_weights_sum_to_one(self) -> None:
        for factory in [default_config, fraud_detection_config, credit_scoring_config]:
            c = factory()
            assert abs(sum(c.weights.values()) - 1.0) < 1e-6
