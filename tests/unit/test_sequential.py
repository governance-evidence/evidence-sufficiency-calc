"""Tests for sequential testing modules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sufficiency.sequential import EValueAccumulator, ThresholdMonitor
from sufficiency.types import (
    DimensionScore,
    SufficiencyResult,
    SufficiencyStatus,
    SufficiencyThresholds,
)


def _make_result(composite: float, status: SufficiencyStatus, day: int = 1) -> SufficiencyResult:
    dims = {
        "completeness": DimensionScore(0.8, 0.8, 0.8, "completeness"),
        "freshness": DimensionScore(0.8, 0.8, 0.8, "freshness"),
        "reliability": DimensionScore(0.8, 0.8, 0.8, "reliability"),
        "representativeness": DimensionScore(0.8, 0.8, 0.8, "representativeness"),
    }
    return SufficiencyResult(
        composite=composite,
        gate=1.0,
        dimensions=dims,
        status=status,
        timestamp=datetime(2026, 1, day, tzinfo=UTC),
    )


class TestThresholdMonitor:
    def test_no_alert_when_stable(self) -> None:
        mon = ThresholdMonitor()
        alert = mon.observe(_make_result(0.85, SufficiencyStatus.SUFFICIENT, 1))
        assert alert is None
        alert = mon.observe(_make_result(0.82, SufficiencyStatus.SUFFICIENT, 2))
        assert alert is None

    def test_alert_on_degradation(self) -> None:
        mon = ThresholdMonitor()
        mon.observe(_make_result(0.85, SufficiencyStatus.SUFFICIENT, 1))
        alert = mon.observe(_make_result(0.65, SufficiencyStatus.DEGRADED, 2))
        assert alert is not None
        assert alert.status == SufficiencyStatus.DEGRADED

    def test_alert_on_insufficient(self) -> None:
        mon = ThresholdMonitor()
        mon.observe(_make_result(0.65, SufficiencyStatus.DEGRADED, 1))
        alert = mon.observe(_make_result(0.40, SufficiencyStatus.INSUFFICIENT, 2))
        assert alert is not None
        assert alert.status == SufficiencyStatus.INSUFFICIENT

    def test_no_alert_on_improvement(self) -> None:
        mon = ThresholdMonitor()
        mon.observe(_make_result(0.65, SufficiencyStatus.DEGRADED, 1))
        alert = mon.observe(_make_result(0.85, SufficiencyStatus.SUFFICIENT, 2))
        assert alert is None

    def test_history_tracked(self) -> None:
        mon = ThresholdMonitor()
        mon.observe(_make_result(0.85, SufficiencyStatus.SUFFICIENT, 1))
        mon.observe(_make_result(0.65, SufficiencyStatus.DEGRADED, 2))
        assert len(mon.history) == 2
        assert len(mon.alerts) == 1

    def test_uses_monitor_thresholds_not_result_status(self) -> None:
        mon = ThresholdMonitor(thresholds=SufficiencyThresholds(sufficient=0.9, degraded=0.7))

        mon.observe(_make_result(0.95, SufficiencyStatus.SUFFICIENT, 1))
        alert = mon.observe(_make_result(0.75, SufficiencyStatus.SUFFICIENT, 2))

        assert alert is not None
        assert alert.status == SufficiencyStatus.DEGRADED

    def test_inconsistent_result_status_does_not_override_monitor_policy(self) -> None:
        mon = ThresholdMonitor(thresholds=SufficiencyThresholds(sufficient=0.8, degraded=0.5))

        mon.observe(_make_result(0.85, SufficiencyStatus.INSUFFICIENT, 1))
        alert = mon.observe(_make_result(0.45, SufficiencyStatus.SUFFICIENT, 2))

        assert alert is not None
        assert alert.status == SufficiencyStatus.INSUFFICIENT


class TestEValueAccumulator:
    def test_initial_state(self) -> None:
        acc = EValueAccumulator()
        assert acc.e_value == 1.0
        assert not acc.rejected

    def test_rejects_nonpositive_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be in"):
            EValueAccumulator(threshold=0.0)

    def test_rejects_nonpositive_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be in"):
            EValueAccumulator(alpha=0.0)

    def test_accumulates_evidence_below_threshold(self) -> None:
        acc = EValueAccumulator(threshold=0.8)
        for _ in range(50):
            acc.observe(0.4)
        assert acc.e_value > 1.0

    def test_eventually_rejects_with_low_scores(self) -> None:
        acc = EValueAccumulator(threshold=0.8, alpha=0.05)
        for _ in range(100):
            if acc.observe(0.3):
                break
        assert acc.rejected

    def test_does_not_reject_at_threshold(self) -> None:
        acc = EValueAccumulator(threshold=0.8)
        for _ in range(20):
            acc.observe(0.85)
        assert not acc.rejected

    def test_rejects_score_out_of_range(self) -> None:
        acc = EValueAccumulator()
        with pytest.raises(ValueError, match="score must be in"):
            acc.observe(1.5)

    def test_e_value_saturates_to_inf_for_large_log_value(self) -> None:
        acc = EValueAccumulator()
        acc.log_e_value = 1000.0

        assert acc.e_value == float("inf")

    def test_rejected_works_when_e_value_saturates(self) -> None:
        acc = EValueAccumulator(alpha=0.05)
        acc.log_e_value = 1000.0

        assert acc.rejected
