"""Unit tests for blind period simulator input validation."""

from __future__ import annotations

import pytest

from sufficiency.blind_period import BlindPeriodSimulator
from sufficiency.config import default_config
from sufficiency.types import DriftSpec, DriftType


class TestBlindPeriodSimulatorValidation:
    def test_rejects_initial_completeness_above_one(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"initial_completeness must be finite and in \[0, 1\]",
        ):
            BlindPeriodSimulator(initial_completeness=1.1, config=default_config())

    def test_rejects_initial_reliability_nan(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"initial_reliability must be finite and in \[0, 1\]",
        ):
            BlindPeriodSimulator(initial_reliability=float("nan"), config=default_config())

    def test_rejects_initial_representativeness_below_zero(self) -> None:
        with pytest.raises(
            ValueError,
            match=r"initial_representativeness must be finite and in \[0, 1\]",
        ):
            BlindPeriodSimulator(initial_representativeness=-0.1, config=default_config())

    def test_rejects_non_integer_checkpoint_day(self) -> None:
        sim = BlindPeriodSimulator(config=default_config())
        with pytest.raises(TypeError, match="days must contain only non-negative integers"):
            sim.simulate([30, 60.5])

    def test_rejects_boolean_checkpoint_day(self) -> None:
        sim = BlindPeriodSimulator(config=default_config())
        with pytest.raises(TypeError, match="days must contain only non-negative integers"):
            sim.simulate([True])

    def test_rejects_non_integer_daily_horizon(self) -> None:
        sim = BlindPeriodSimulator(config=default_config())
        with pytest.raises(TypeError, match="total_days must be a non-negative integer"):
            sim.simulate_daily(30.0)

    def test_accepts_valid_drift_spec_list(self) -> None:
        sim = BlindPeriodSimulator(
            config=default_config(),
            drift_specs=[DriftSpec(DriftType.COVARIATE, magnitude=0.4, onset_day=10)],
        )

        results = sim.simulate([30])

        assert len(results) == 1
