"""Tests for composite sufficiency scoring."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from sufficiency.composite import compute_sufficiency
from sufficiency.types import DimensionScore, GovernanceConfig, SufficiencyStatus


def _make_dimensions(c: float, f: float, r: float, p: float) -> dict[str, DimensionScore]:
    return {
        "completeness": DimensionScore(c, c, c, "completeness"),
        "freshness": DimensionScore(f, f, f, "freshness"),
        "reliability": DimensionScore(r, r, r, "reliability"),
        "representativeness": DimensionScore(p, p, p, "representativeness"),
    }


class TestComposite:
    def test_perfect_scores(self) -> None:
        dims = _make_dimensions(1.0, 1.0, 1.0, 1.0)
        result = compute_sufficiency(dims, GovernanceConfig())
        assert abs(result.composite - 1.0) < 1e-6
        assert result.gate == 1.0
        assert result.status == SufficiencyStatus.SUFFICIENT

    def test_equal_weights_average(self) -> None:
        dims = _make_dimensions(0.8, 0.6, 0.9, 0.7)
        result = compute_sufficiency(dims, GovernanceConfig())
        # gate: C=0.8 >= tau_c=0.6, R=0.9 >= tau_r=0.7 -> A=1.0
        expected = 0.25 * (0.8 + 0.6 + 0.9 + 0.7)  # = 0.75
        assert abs(result.composite - expected) < 1e-6
        assert result.status == SufficiencyStatus.DEGRADED

    def test_gate_suppression(self) -> None:
        """Low completeness suppresses composite regardless of other dimensions."""
        dims = _make_dimensions(0.3, 1.0, 1.0, 1.0)
        result = compute_sufficiency(dims, GovernanceConfig())
        # gate: C/tau_c = 0.3/0.6 = 0.5, R ok -> A=0.5
        # weighted_sum = 0.25*(0.3+1.0+1.0+1.0) = 0.825
        # composite = 0.5 * 0.825 = 0.4125
        assert abs(result.composite - 0.4125) < 1e-4
        assert result.status == SufficiencyStatus.INSUFFICIENT

    def test_missing_dimension_raises(self) -> None:
        dims = {
            "completeness": DimensionScore(0.8, 0.8, 0.8, "completeness"),
            "freshness": DimensionScore(0.8, 0.8, 0.8, "freshness"),
        }
        with pytest.raises(ValueError, match="Dimension keys must exactly match required set"):
            compute_sufficiency(dims, GovernanceConfig())

    def test_unexpected_dimension_raises(self) -> None:
        dims = _make_dimensions(0.8, 0.8, 0.8, 0.8)
        dims["freshnes"] = DimensionScore(0.1, 0.1, 0.1, "freshnes")
        with pytest.raises(ValueError, match="unexpected"):
            compute_sufficiency(dims, GovernanceConfig())

    def test_custom_timestamp(self) -> None:
        dims = _make_dimensions(1.0, 1.0, 1.0, 1.0)
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        result = compute_sufficiency(dims, GovernanceConfig(), timestamp=ts)
        assert result.timestamp == ts

    def test_naive_timestamp_raises(self) -> None:
        dims = _make_dimensions(1.0, 1.0, 1.0, 1.0)

        with pytest.raises(ValueError, match="timestamp must be timezone-aware"):
            compute_sufficiency(dims, GovernanceConfig(), timestamp=datetime(2026, 1, 1))

    def test_result_dimensions_are_immutable_copy(self) -> None:
        dims = _make_dimensions(1.0, 1.0, 1.0, 1.0)

        result = compute_sufficiency(dims, GovernanceConfig())

        assert isinstance(result.dimensions, MappingProxyType)
        dims["freshness"] = DimensionScore(0.1, 0.1, 0.1, "freshness")
        assert result.dimensions["freshness"].value == 1.0
        with pytest.raises(TypeError):
            result.dimensions["freshness"] = DimensionScore(0.2, 0.2, 0.2, "freshness")
