"""Tests for the decision-readiness gate."""

from __future__ import annotations

import math

import pytest

from sufficiency.gate import compute_gate


class TestGate:
    def test_both_above_threshold(self) -> None:
        assert compute_gate(0.8, 0.9) == 1.0

    def test_exactly_at_threshold(self) -> None:
        assert abs(compute_gate(0.6, 0.7) - 1.0) < 1e-6

    def test_completeness_below(self) -> None:
        """C=0.3, R=0.8. C/tau_c = 0.3/0.6 = 0.5, R ok -> A = 0.5"""
        a = compute_gate(0.3, 0.8)
        assert abs(a - 0.5) < 1e-6

    def test_reliability_below(self) -> None:
        """C=0.8, R=0.35. R/tau_r = 0.35/0.7 = 0.5 -> A = 0.5"""
        a = compute_gate(0.8, 0.35)
        assert abs(a - 0.5) < 1e-6

    def test_both_below_compounding(self) -> None:
        """C=0.3, R=0.35. A = 0.5 * 0.5 = 0.25 (compounding penalty)."""
        a = compute_gate(0.3, 0.35)
        assert abs(a - 0.25) < 1e-6

    def test_zero_completeness(self) -> None:
        assert compute_gate(0.0, 0.9) == 0.0

    def test_zero_reliability(self) -> None:
        assert compute_gate(0.9, 0.0) == 0.0

    def test_custom_thresholds(self) -> None:
        a = compute_gate(0.4, 0.4, tau_c=0.8, tau_r=0.8)
        assert abs(a - 0.25) < 1e-6

    def test_rejects_completeness_above_one(self) -> None:
        with pytest.raises(ValueError, match=r"completeness must be finite and in \[0, 1\]"):
            compute_gate(1.1, 0.8)

    def test_rejects_non_finite_reliability(self) -> None:
        with pytest.raises(ValueError, match=r"reliability must be finite and in \[0, 1\]"):
            compute_gate(0.8, math.inf)

    def test_rejects_nonpositive_tau_c(self) -> None:
        with pytest.raises(ValueError, match="tau_c must be finite and positive"):
            compute_gate(0.8, 0.8, tau_c=0.0)

    def test_rejects_non_finite_tau_r(self) -> None:
        with pytest.raises(ValueError, match="tau_r must be finite and positive"):
            compute_gate(0.8, 0.8, tau_r=math.nan)
