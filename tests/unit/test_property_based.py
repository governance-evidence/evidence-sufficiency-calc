"""Property-based tests using Hypothesis for dimension scorers and validators."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sufficiency._validation import coerce_binary_labels, require_unit_interval
from sufficiency.dimensions.completeness import compute_completeness
from sufficiency.dimensions.freshness import compute_freshness
from sufficiency.dimensions.representativeness import compute_representativeness
from sufficiency.gate import compute_gate

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
positive_float = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
small_positive_float = st.floats(
    min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False
)


# ---------------------------------------------------------------------------
# require_unit_interval
# ---------------------------------------------------------------------------


class TestRequireUnitIntervalProperties:
    @given(value=unit_float)
    def test_accepts_all_unit_interval_values(self, value: float) -> None:
        require_unit_interval("x", value)  # should not raise

    @given(value=st.floats(min_value=1.001, max_value=1e10))
    def test_rejects_above_one(self, value: float) -> None:
        with pytest.raises(ValueError):
            require_unit_interval("x", value)

    @given(value=st.floats(max_value=-0.001, min_value=-1e10))
    def test_rejects_below_zero(self, value: float) -> None:
        with pytest.raises(ValueError):
            require_unit_interval("x", value)


# ---------------------------------------------------------------------------
# compute_completeness
# ---------------------------------------------------------------------------


class TestCompletenessProperties:
    @given(
        labeled=st.integers(min_value=0, max_value=10000),
        extra=st.integers(min_value=0, max_value=10000),
    )
    def test_score_always_in_unit_interval(self, labeled: int, extra: int) -> None:
        total = labeled + extra
        if total == 0:
            return  # empty window is a special case
        score = compute_completeness(labeled, total)
        assert 0.0 <= score.value <= 1.0
        assert 0.0 <= score.confidence_low <= 1.0
        assert 0.0 <= score.confidence_high <= 1.0
        # Wilson CI confidence_high can be < value by float epsilon when labeled==total
        assert score.confidence_low <= score.confidence_high + 1e-12

    @given(total=st.integers(min_value=1, max_value=10000))
    def test_full_completeness_is_one(self, total: int) -> None:
        score = compute_completeness(total, total)
        assert score.value == 1.0

    @given(total=st.integers(min_value=1, max_value=10000))
    def test_zero_completeness_is_zero(self, total: int) -> None:
        score = compute_completeness(0, total)
        assert score.value == 0.0


# ---------------------------------------------------------------------------
# compute_freshness
# ---------------------------------------------------------------------------


class TestFreshnessProperties:
    @given(delta_t=positive_float)
    def test_score_always_in_unit_interval(self, delta_t: float) -> None:
        score = compute_freshness(delta_t)
        assert 0.0 <= score.value <= 1.0

    @given(delta_t=positive_float, lambda_rate=small_positive_float)
    def test_monotonic_decay(self, delta_t: float, lambda_rate: float) -> None:
        score_0 = compute_freshness(0.0, lambda_rate=lambda_rate)
        score_t = compute_freshness(delta_t, lambda_rate=lambda_rate)
        assert score_t.value <= score_0.value + 1e-10  # allow float tolerance

    def test_zero_delay_is_one(self) -> None:
        score = compute_freshness(0.0)
        assert score.value == 1.0


# ---------------------------------------------------------------------------
# compute_gate
# ---------------------------------------------------------------------------


class TestGateProperties:
    @given(completeness=unit_float, reliability=unit_float)
    def test_gate_always_in_unit_interval(self, completeness: float, reliability: float) -> None:
        gate = compute_gate(completeness, reliability)
        assert 0.0 <= gate <= 1.0

    @given(completeness=unit_float, reliability=unit_float)
    def test_gate_increases_with_inputs(self, completeness: float, reliability: float) -> None:
        gate = compute_gate(completeness, reliability)
        gate_max = compute_gate(1.0, 1.0)
        assert gate <= gate_max + 1e-10


# ---------------------------------------------------------------------------
# compute_representativeness
# ---------------------------------------------------------------------------


class TestRepresentativenessProperties:
    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=20)
    def test_identical_distributions_score_one(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        data = rng.standard_normal(50)
        score = compute_representativeness(data, data)
        assert score.value == 1.0

    @given(
        shift=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(max_examples=20)
    def test_score_always_in_unit_interval(self, shift: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ref = rng.standard_normal(50)
        prod = rng.standard_normal(50) + shift
        score = compute_representativeness(ref, prod)
        assert 0.0 <= score.value <= 1.0


# ---------------------------------------------------------------------------
# coerce_binary_labels
# ---------------------------------------------------------------------------


class TestCoerceBinaryLabelsProperties:
    @given(data=st.lists(st.sampled_from([0, 1]), min_size=1, max_size=100))
    def test_accepts_valid_binary_lists(self, data: list[int]) -> None:
        result = coerce_binary_labels("test", data)
        assert result.dtype == np.int64
        assert set(result.tolist()).issubset({0, 1})

    @given(data=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=100))
    def test_accepts_float_binary_lists(self, data: list[float]) -> None:
        result = coerce_binary_labels("test", data)
        assert result.dtype == np.int64
