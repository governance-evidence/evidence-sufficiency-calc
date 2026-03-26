"""Tests for reliability dimension."""

from __future__ import annotations

import numpy as np
import pytest

from sufficiency.dimensions.reliability import _bootstrap_batch_size, compute_reliability


class TestReliability:
    def test_perfect_predictions(self) -> None:
        y = [1, 1, 0, 0, 1, 0]
        s = compute_reliability(y, y, rng_seed=42)
        assert abs(s.value - 1.0) < 1e-6

    def test_all_wrong(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [0, 0, 0, 1, 1, 1]
        s = compute_reliability(y_true, y_pred, rng_seed=42)
        assert s.value == 0.0

    def test_known_f1(self) -> None:
        """tp=2, fp=1, fn=1 -> P=2/3, R=2/3, F1=2/3"""
        y_true = [1, 1, 1, 0, 0]
        y_pred = [1, 1, 0, 1, 0]
        s = compute_reliability(y_true, y_pred, rng_seed=42)
        assert abs(s.value - 2 / 3) < 1e-6

    def test_ci_contains_point(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).tolist()
        y_pred = rng.integers(0, 2, size=200).tolist()
        s = compute_reliability(y_true, y_pred, rng_seed=42)
        assert s.confidence_low <= s.value <= s.confidence_high

    def test_seeded_results_are_reproducible(self) -> None:
        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=300).tolist()
        y_pred = rng.integers(0, 2, size=300).tolist()

        s1 = compute_reliability(y_true, y_pred, n_bootstrap=257, rng_seed=123)
        s2 = compute_reliability(y_true, y_pred, n_bootstrap=257, rng_seed=123)

        assert s1.value == s2.value
        assert s1.confidence_low == s2.confidence_low
        assert s1.confidence_high == s2.confidence_high

    def test_chunked_bootstrap_path(self) -> None:
        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 2, size=128).tolist()
        y_pred = rng.integers(0, 2, size=128).tolist()

        s = compute_reliability(y_true, y_pred, n_bootstrap=513, rng_seed=99)

        assert 0.0 <= s.value <= 1.0
        assert 0.0 <= s.confidence_low <= s.confidence_high <= 1.0

    def test_batch_size_prefers_larger_chunks_for_small_samples(self) -> None:
        assert _bootstrap_batch_size(n_samples=2_048, n_bootstrap=1_000) == 512

    def test_batch_size_prefers_smaller_chunks_for_large_samples(self) -> None:
        assert _bootstrap_batch_size(n_samples=2_049, n_bootstrap=1_000) == 64

    def test_batch_size_never_exceeds_bootstrap_count(self) -> None:
        assert _bootstrap_batch_size(n_samples=100, n_bootstrap=20) == 20

    def test_empty_arrays(self) -> None:
        s = compute_reliability([], [])
        assert s.value == 0.0
        assert s.confidence_high == 1.0

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_reliability([1, 0], [1])

    def test_rejects_non_binary_true_labels(self) -> None:
        with pytest.raises(ValueError, match="y_true must contain only binary labels"):
            compute_reliability([0, 2, 1], [0, 1, 1])

    def test_rejects_non_binary_pred_labels(self) -> None:
        with pytest.raises(ValueError, match="y_pred must contain only binary labels"):
            compute_reliability([0, 1, 1], [0, 2, 1])

    def test_accepts_float_binary_labels(self) -> None:
        s = compute_reliability([0.0, 1.0, 1.0], [0.0, 1.0, 0.0], rng_seed=42)
        assert 0.0 <= s.value <= 1.0

    def test_accepts_boolean_labels(self) -> None:
        s = compute_reliability([True, False, True], [True, False, False], rng_seed=42)
        assert 0.0 <= s.value <= 1.0

    def test_rejects_non_binary_floats_without_silent_truncation(self) -> None:
        with pytest.raises(ValueError, match="y_true must contain only binary labels"):
            compute_reliability([0.9, 0.1, 1.0], [1.0, 0.0, 1.0])

    def test_rejects_probabilities_like_scores(self) -> None:
        with pytest.raises(ValueError, match="y_pred must contain only binary labels"):
            compute_reliability([0, 1, 1], [0.8, 0.1, 0.6])

    def test_rejects_string_labels(self) -> None:
        with pytest.raises(ValueError, match="y_true must contain only binary labels"):
            compute_reliability(["0", "1"], [0, 1])

    def test_rejects_object_dtype_with_unsupported_values(self) -> None:
        y_true = np.array([0, object()], dtype=object)

        with pytest.raises(ValueError, match="y_true must contain only binary labels"):
            compute_reliability(y_true, [0, 1])

    def test_accepts_object_dtype_binary_numeric_values(self) -> None:
        y_true = np.array([0, 1, 1], dtype=object)
        y_pred = np.array([0, 1, 0], dtype=object)

        s = compute_reliability(y_true, y_pred, rng_seed=42)
        assert 0.0 <= s.value <= 1.0

    def test_rejects_nonfinite_labels(self) -> None:
        with pytest.raises(ValueError, match="y_true must contain only finite binary labels"):
            compute_reliability([0.0, float("nan")], [0, 1])

    def test_rejects_non_real_numeric_labels(self) -> None:
        with pytest.raises(ValueError, match="y_true must contain only binary labels"):
            compute_reliability([0j, 1 + 0j], [0, 1])

    def test_rejects_nonpositive_bootstrap_count(self) -> None:
        with pytest.raises(ValueError, match="n_bootstrap must be positive"):
            compute_reliability([0, 1], [0, 1], n_bootstrap=0)

    def test_rejects_invalid_confidence_level(self) -> None:
        with pytest.raises(ValueError, match="confidence_level must be in"):
            compute_reliability([0, 1], [0, 1], confidence_level=1.0)
