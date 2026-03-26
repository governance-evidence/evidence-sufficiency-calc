"""Tests for completeness dimension."""

from __future__ import annotations

import numpy as np
import pytest

from sufficiency.dimensions.completeness import compute_completeness


class TestCompleteness:
    def test_full_completeness(self) -> None:
        s = compute_completeness(100, 100)
        assert s.value == 1.0
        assert s.label == "completeness"

    def test_zero_completeness(self) -> None:
        s = compute_completeness(0, 100)
        assert s.value == 0.0

    def test_partial_completeness(self) -> None:
        s = compute_completeness(60, 100)
        assert abs(s.value - 0.6) < 1e-6

    def test_empty_window(self) -> None:
        s = compute_completeness(0, 0)
        assert s.value == 0.0

    def test_confidence_interval_contains_value(self) -> None:
        s = compute_completeness(50, 100)
        assert s.confidence_low <= s.value <= s.confidence_high

    def test_larger_sample_tighter_ci(self) -> None:
        small = compute_completeness(50, 100)
        large = compute_completeness(500, 1000)
        small_width = small.confidence_high - small.confidence_low
        large_width = large.confidence_high - large.confidence_low
        assert large_width < small_width

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError):
            compute_completeness(-1, 100)

    def test_rejects_labeled_exceeding_total(self) -> None:
        with pytest.raises(ValueError):
            compute_completeness(101, 100)

    def test_accepts_numpy_integer_counts(self) -> None:
        s = compute_completeness(np.int64(60), np.int64(100))
        assert abs(s.value - 0.6) < 1e-6

    def test_rejects_float_labeled_count(self) -> None:
        with pytest.raises(TypeError, match="labeled_count must be an integer-like count"):
            compute_completeness(0.5, 1)

    def test_rejects_boolean_total_count(self) -> None:
        with pytest.raises(TypeError, match="total_count must be an integer-like count"):
            compute_completeness(1, True)

    def test_rejects_invalid_confidence_level(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in"):
            compute_completeness(50, 100, confidence_level=1.0)

    def test_wilson_ci_low_proportion(self) -> None:
        """Wilson interval for low proportion exercises the p < 0.5 branch."""
        s = compute_completeness(5, 100)
        assert s.value == 0.05
        assert s.confidence_low < s.value < s.confidence_high


class TestNormalQuantile:
    """Tests for the internal _normal_quantile function."""

    def test_rejects_out_of_range(self) -> None:
        from sufficiency.dimensions.completeness import _normal_quantile

        with pytest.raises(ValueError, match="must be in"):
            _normal_quantile(0.0)
        with pytest.raises(ValueError, match="must be in"):
            _normal_quantile(1.0)

    def test_symmetry(self) -> None:
        from sufficiency.dimensions.completeness import _normal_quantile

        q_low = _normal_quantile(0.025)
        q_high = _normal_quantile(0.975)
        assert abs(q_low + q_high) < 0.01  # symmetric around 0
