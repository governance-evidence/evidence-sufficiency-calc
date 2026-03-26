"""Tests for representativeness dimension."""

from __future__ import annotations

import numpy as np
import pytest

from sufficiency.dimensions import representativeness as rep
from sufficiency.dimensions.representativeness import compute_representativeness


class TestRepresentativeness:
    def test_identical_array_object_skips_ks_computation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_if_called(*args: object, **kwargs: object) -> object:
            raise AssertionError("ks_2samp should not be called for the same array object")

        monkeypatch.setattr(rep.stats, "ks_2samp", fail_if_called)

        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=1000)

        s = compute_representativeness(data, data)

        assert s.value == 1.0

    def test_identical_distributions(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, size=1000)
        s = compute_representativeness(data, data)
        assert s.value > 0.9

    def test_very_different_distributions(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=500)
        prod = rng.normal(5, 1, size=500)
        s = compute_representativeness(ref, prod)
        assert s.value == 0.0  # KS >> ks_cap

    def test_moderate_shift(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=500)
        prod = rng.normal(0.3, 1, size=500)
        s = compute_representativeness(ref, prod)
        assert 0.0 < s.value < 1.0

    def test_ci_ordering(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, size=200)
        prod = rng.normal(0.2, 1, size=200)
        s = compute_representativeness(ref, prod)
        assert s.confidence_low <= s.confidence_high

    def test_rejects_empty_arrays(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_representativeness([], [1.0, 2.0])

    def test_rejects_nonpositive_cap(self) -> None:
        with pytest.raises(ValueError):
            compute_representativeness([1.0], [1.0], ks_cap=0.0)

    def test_rejects_nan_in_reference(self) -> None:
        with pytest.raises(ValueError, match="reference must contain only finite"):
            compute_representativeness([1.0, float("nan")], [1.0, 2.0])

    def test_rejects_inf_in_production(self) -> None:
        with pytest.raises(ValueError, match="production must contain only finite"):
            compute_representativeness([1.0, 2.0], [1.0, float("inf")])

    def test_rejects_multidimensional_reference(self) -> None:
        with pytest.raises(ValueError, match="reference must be a one-dimensional"):
            compute_representativeness([[1.0, 2.0]], [1.0, 2.0])

    def test_rejects_multidimensional_production(self) -> None:
        with pytest.raises(ValueError, match="production must be a one-dimensional"):
            compute_representativeness([1.0, 2.0], [[1.0, 2.0]])

    def test_rejects_scalar_reference(self) -> None:
        with pytest.raises(ValueError, match="reference must be a one-dimensional"):
            compute_representativeness(1.0, [1.0, 2.0])

    def test_rejects_scalar_production(self) -> None:
        with pytest.raises(ValueError, match="production must be a one-dimensional"):
            compute_representativeness([1.0, 2.0], 1.0)
