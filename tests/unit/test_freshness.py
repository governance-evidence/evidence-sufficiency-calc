"""Tests for freshness dimension."""

from __future__ import annotations

import math

import pytest

from sufficiency.dimensions.freshness import compute_freshness


class TestFreshness:
    def test_zero_delay(self) -> None:
        s = compute_freshness(0.0)
        assert abs(s.value - 1.0) < 1e-6

    def test_30_day_delay(self) -> None:
        """F(30) = exp(-0.02 * 30) = exp(-0.6) ≈ 0.5488"""
        s = compute_freshness(30.0)
        expected = math.exp(-0.6)
        assert abs(s.value - expected) < 1e-4

    def test_180_day_delay(self) -> None:
        """F(180) = exp(-0.02 * 180) = exp(-3.6) ≈ 0.0273"""
        s = compute_freshness(180.0)
        expected = math.exp(-3.6)
        assert abs(s.value - expected) < 1e-4

    def test_custom_lambda(self) -> None:
        """Credit scoring: lambda=0.005, F(30) = exp(-0.15) ≈ 0.8607"""
        s = compute_freshness(30.0, lambda_rate=0.005)
        expected = math.exp(-0.15)
        assert abs(s.value - expected) < 1e-4

    def test_monotonic_decay(self) -> None:
        scores = [compute_freshness(float(d)).value for d in range(0, 200, 10)]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]

    def test_with_uncertainty(self) -> None:
        s = compute_freshness(30.0, delta_t_std=5.0)
        assert s.confidence_low < s.value < s.confidence_high

    def test_rejects_negative_delay(self) -> None:
        with pytest.raises(ValueError):
            compute_freshness(-1.0)

    def test_rejects_nonpositive_lambda(self) -> None:
        with pytest.raises(ValueError):
            compute_freshness(30.0, lambda_rate=0.0)

    def test_rejects_non_finite_delay(self) -> None:
        with pytest.raises(ValueError, match="delta_t_days must be non-negative"):
            compute_freshness(float("nan"))

    def test_rejects_non_finite_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_rate must be positive"):
            compute_freshness(30.0, lambda_rate=float("inf"))

    def test_rejects_negative_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="delta_t_std must be finite and non-negative"):
            compute_freshness(30.0, delta_t_std=-1.0)

    def test_rejects_non_finite_uncertainty(self) -> None:
        with pytest.raises(ValueError, match="delta_t_std must be finite and non-negative"):
            compute_freshness(30.0, delta_t_std=float("nan"))
