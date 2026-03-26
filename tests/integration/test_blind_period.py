"""Integration tests for blind period simulation."""

from __future__ import annotations

import pytest

from sufficiency.blind_period import BlindPeriodSimulator
from sufficiency.config import default_config
from sufficiency.types import DriftSpec, DriftType


class TestBlindPeriodSimulation:
    def test_degradation_over_time(self) -> None:
        """Sufficiency should monotonically degrade during a blind period."""
        sim = BlindPeriodSimulator(config=default_config())
        results = sim.simulate([30, 60, 90, 180])
        scores = [r.composite for r in results]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]

    def test_180_day_drops_below_threshold(self) -> None:
        """180-day blind period should produce insufficient evidence."""
        sim = BlindPeriodSimulator(
            config=default_config(),
            drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=0.5)],
        )
        results = sim.simulate([180])
        assert results[0].composite < 0.5

    def test_covariate_drift_hits_representativeness(self) -> None:
        """Covariate drift should primarily degrade representativeness."""
        sim = BlindPeriodSimulator(
            config=default_config(),
            drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=0.8)],
        )
        results = sim.simulate([90])
        dims = results[0].dimensions
        assert dims["representativeness"].value < dims["reliability"].value

    def test_concept_drift_hits_reliability(self) -> None:
        """Real concept drift should primarily degrade reliability."""
        sim = BlindPeriodSimulator(
            config=default_config(),
            drift_specs=[DriftSpec(DriftType.REAL_CONCEPT, magnitude=0.8)],
        )
        results = sim.simulate([90])
        dims = results[0].dimensions
        assert dims["reliability"].value < dims["representativeness"].value

    def test_no_drift_still_degrades(self) -> None:
        """Even without drift, freshness decay should degrade sufficiency."""
        sim = BlindPeriodSimulator(config=default_config())
        results = sim.simulate([90])
        assert results[0].composite < 0.8

    def test_daily_simulation(self) -> None:
        """Daily simulation should produce expected number of results."""
        sim = BlindPeriodSimulator(config=default_config())
        results = sim.simulate_daily(30)
        assert len(results) == 30

    def test_default_days(self) -> None:
        """Simulate with default days (None -> [30, 60, 90, 180])."""
        sim = BlindPeriodSimulator(config=default_config())
        results = sim.simulate()
        assert len(results) == 4

    def test_drift_onset_after_observation(self) -> None:
        """Drift that starts after the observation day should not affect score."""
        sim_no_drift = BlindPeriodSimulator(config=default_config())
        sim_late_drift = BlindPeriodSimulator(
            config=default_config(),
            drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=1.0, onset_day=100)],
        )
        r_no = sim_no_drift.simulate([30])
        r_late = sim_late_drift.simulate([30])
        assert abs(r_no[0].composite - r_late[0].composite) < 1e-6

    def test_rejects_negative_checkpoint_day(self) -> None:
        sim = BlindPeriodSimulator(config=default_config())
        with pytest.raises(ValueError, match="days must contain only non-negative integers"):
            sim.simulate([-1, 30])

    def test_rejects_negative_daily_horizon(self) -> None:
        sim = BlindPeriodSimulator(config=default_config())
        with pytest.raises(ValueError, match="total_days must be non-negative"):
            sim.simulate_daily(-1)
