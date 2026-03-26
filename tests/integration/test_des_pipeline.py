"""End-to-end integration test: DES events -> sufficiency score.

Validates the full pipeline from raw Decision Event Schema events through
adapter extraction to composite sufficiency scoring.
"""

from __future__ import annotations

from typing import Any

import pytest

from sufficiency import compute_sufficiency, default_config
from sufficiency.adapters.des import (
    extract_completeness_inputs,
    extract_freshness_inputs,
    validate_events,
)
from sufficiency.dimensions.completeness import compute_completeness
from sufficiency.dimensions.freshness import compute_freshness
from sufficiency.types import DimensionScore


class TestDesEndToEndPipeline:
    """Full pipeline: DES events -> dimension scores -> composite."""

    def test_knight_capital_pipeline(self, knight_capital_event: dict[str, Any]) -> None:
        """Knight Capital: no ground truth -> low completeness -> gate suppression."""
        events = [knight_capital_event]

        # Step 1: validate
        errors = validate_events(events)
        assert errors == []

        # Step 2: extract
        labeled, total = extract_completeness_inputs(events)
        delta_t = extract_freshness_inputs(events)

        # Step 3: compute dimensions
        completeness = compute_completeness(labeled, total)
        freshness = compute_freshness(delta_t)

        # Proxy scores for dimensions we can't extract from single event
        reliability = DimensionScore(0.5, 0.4, 0.6, "reliability")
        representativeness = DimensionScore(0.7, 0.6, 0.8, "representativeness")

        # Step 4: composite
        dimensions = {
            "completeness": completeness,
            "freshness": freshness,
            "reliability": reliability,
            "representativeness": representativeness,
        }
        result = compute_sufficiency(dimensions, default_config())

        # Knight Capital had ground_truth_available=false -> labeled=0
        assert labeled == 0
        assert completeness.value == pytest.approx(0.0)

        # Zero completeness should suppress composite via gate
        assert result.composite < 0.5
        assert result.gate < 1.0

    def test_mixed_batch_pipeline(
        self,
        all_des_events: list[dict[str, Any]],
    ) -> None:
        """Mixed batch of real DES events produces valid sufficiency score."""
        # Step 1: validate all events
        errors = validate_events(all_des_events)
        assert errors == []

        # Step 2: extract from batch
        labeled, total = extract_completeness_inputs(all_des_events)
        delta_t = extract_freshness_inputs(all_des_events)

        assert total == len(all_des_events)
        assert 0 <= labeled <= total
        assert delta_t >= 0.0

        # Step 3: compute
        completeness = compute_completeness(labeled, total)
        freshness = compute_freshness(delta_t)

        dimensions = {
            "completeness": completeness,
            "freshness": freshness,
            "reliability": DimensionScore(0.8, 0.7, 0.9, "reliability"),
            "representativeness": DimensionScore(0.85, 0.8, 0.9, "representativeness"),
        }

        result = compute_sufficiency(dimensions, default_config())

        # Basic sanity
        assert 0.0 <= result.composite <= 1.0
        assert 0.0 <= result.gate <= 1.0
        assert result.status is not None

    def test_pipeline_with_all_config_presets(
        self,
        minimal_valid_event: dict[str, Any],
    ) -> None:
        """Verify pipeline works with every config factory."""
        from sufficiency.config import (
            credit_scoring_config,
            fraud_detection_config,
        )

        events = [minimal_valid_event]
        labeled, total = extract_completeness_inputs(events)
        delta_t = extract_freshness_inputs(events)

        completeness = compute_completeness(labeled, total)
        freshness = compute_freshness(delta_t)

        dimensions = {
            "completeness": completeness,
            "freshness": freshness,
            "reliability": DimensionScore(0.9, 0.85, 0.95, "reliability"),
            "representativeness": DimensionScore(0.9, 0.85, 0.95, "representativeness"),
        }

        for config_fn in (default_config, fraud_detection_config, credit_scoring_config):
            result = compute_sufficiency(dimensions, config_fn())
            assert 0.0 <= result.composite <= 1.0
