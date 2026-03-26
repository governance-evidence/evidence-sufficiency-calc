"""Tests for the canonical Decision Event Schema adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from sufficiency.adapters import des
from sufficiency.adapters.des import (
    CompatError,
    extract_completeness_inputs,
    extract_freshness_inputs,
    validate_events,
)

DES_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "des"


def _load_example(name: str) -> dict[str, object]:
    return cast("dict[str, object]", json.loads((DES_FIXTURES / name).read_text()))


def _raise_import_error(_name: str) -> ModuleType:
    raise ImportError("mocked")


class TestValidateEvents:
    def test_packaged_schema_resource_exists(self) -> None:
        assert des._get_schema_resource().is_file()

    def test_valid_event(self) -> None:
        event = _load_example("knight-capital-2012.json")
        errors = validate_events([event])
        assert errors == []

    def test_all_examples_valid(self) -> None:
        events = [_load_example(file.name) for file in sorted(DES_FIXTURES.glob("*.json"))]
        errors = validate_events(events)
        assert errors == [], f"Validation errors: {errors}"

    def test_invalid_event_missing_required(self) -> None:
        event = {"decision_type": "automated"}
        errors = validate_events([event])
        assert len(errors) >= 2

    def test_invalid_decision_type(self) -> None:
        event = {
            "decision_id": "test-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "invalid_type",
        }
        errors = validate_events([event])
        assert len(errors) == 1
        assert "invalid_type" in errors[0]


class TestExtractCompletenessInputs:
    def test_knight_capital_no_ground_truth(self) -> None:
        event = _load_example("knight-capital-2012.json")
        labeled, total = extract_completeness_inputs([event])
        assert total == 1
        assert labeled == 0

    def test_batch_with_mixed_labels(self) -> None:
        labeled_event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "automated",
            "decision_quality_indicators": {"ground_truth_available": True},
        }
        unlabeled_event = {
            "decision_id": "d2",
            "timestamp": "2026-01-02T00:00:00Z",
            "decision_type": "automated",
            "decision_quality_indicators": {"ground_truth_available": False},
        }
        no_indicators = {
            "decision_id": "d3",
            "timestamp": "2026-01-03T00:00:00Z",
            "decision_type": "automated",
        }

        labeled, total = extract_completeness_inputs(
            [labeled_event, unlabeled_event, no_indicators]
        )

        assert total == 3
        assert labeled == 1

    def test_empty_batch(self) -> None:
        labeled, total = extract_completeness_inputs([])
        assert total == 0
        assert labeled == 0


class TestValidateEdgeCases:
    def test_missing_jsonschema_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        des._get_validator_cached.cache_clear()
        des._load_schema_cached.cache_clear()
        monkeypatch.setattr(des, "import_module", _raise_import_error)

        with pytest.raises(CompatError, match="jsonschema is required"):
            validate_events([])

    def test_unsupported_jsonschema_version_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_jsonschema = ModuleType("jsonschema")
        cast("Any", fake_jsonschema).__version__ = "3.2.0"
        des._get_validator_cached.cache_clear()
        des._load_schema_cached.cache_clear()
        monkeypatch.setattr(des, "import_module", lambda name: fake_jsonschema)

        with pytest.raises(CompatError, match="jsonschema>=4.20 is required"):
            validate_events([])

    def test_missing_schema_file_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(des, "_get_schema_resource", lambda: Path("/nonexistent/schema.json"))
        des._load_schema_cached.cache_clear()
        des._get_validator_cached.cache_clear()
        event = {
            "decision_id": "x",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "human",
        }

        with pytest.raises(CompatError, match="schema not found"):
            validate_events([event])

    def test_validator_cached_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        des._get_validator_cached.cache_clear()
        des._load_schema_cached.cache_clear()

        class FakeValidator:
            instances = 0

            def __init__(self, schema: object) -> None:
                self.schema = schema
                FakeValidator.instances += 1

            def iter_errors(self, _event: object) -> list[object]:
                return []

        fake_jsonschema = ModuleType("jsonschema")
        cast("Any", fake_jsonschema).Draft202012Validator = FakeValidator
        monkeypatch.setattr(des, "import_module", lambda name: fake_jsonschema)

        event = _load_example("knight-capital-2012.json")
        assert validate_events([event]) == []
        assert validate_events([event]) == []
        assert FakeValidator.instances == 1


class TestExtractFreshnessInputs:
    def test_knight_capital_label_latency(self) -> None:
        event = _load_example("knight-capital-2012.json")
        delta_t = extract_freshness_inputs([event])
        assert 75 < delta_t < 77

    def test_no_labeled_events(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "automated",
        }
        delta_t = extract_freshness_inputs([event])
        assert delta_t == 0.0

    def test_no_timestamps_returns_zero(self) -> None:
        event = {
            "decision_id": "d1",
            "decision_type": "automated",
            "temporal_metadata": {
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
            },
        }

        delta_t = extract_freshness_inputs([event])
        assert delta_t == 0.0

    def test_with_reference_time(self) -> None:
        event = _load_example("knight-capital-2012.json")
        ref = datetime(2013, 10, 16, tzinfo=UTC)
        delta_t = extract_freshness_inputs([event], reference_time=ref)
        assert 364 < delta_t < 367

    def test_reference_time_before_ground_truth(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "ground_truth_arrival_timestamp": "2026-02-01T00:00:00Z",
            },
        }
        ref = datetime(2026, 1, 15, tzinfo=UTC)
        delta_t = extract_freshness_inputs([event], reference_time=ref)
        assert 30 < delta_t < 32

    def test_negative_latency_skipped(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-02-01T00:00:00Z",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-02-01T00:00:00Z",
                "ground_truth_arrival_timestamp": "2026-01-01T00:00:00Z",
            },
        }
        delta_t = extract_freshness_inputs([event])
        assert delta_t == 0.0

    def test_reference_time_skips_events_without_gt(self) -> None:
        labeled = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
            },
        }
        no_gt = {
            "decision_id": "d2",
            "timestamp": "2026-01-02T00:00:00+00:00",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-02T00:00:00+00:00",
            },
        }
        ref = datetime(2026, 2, 1, tzinfo=UTC)
        delta_t = extract_freshness_inputs([labeled, no_gt], reference_time=ref)
        assert 20 < delta_t < 22

    def test_multiple_events_returns_median(self) -> None:
        events = [
            {
                "decision_id": f"d{index}",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-01T00:00:00Z",
                    "ground_truth_arrival_timestamp": (f"2026-01-{11 + index * 10:02d}T00:00:00Z"),
                },
            }
            for index in range(3)
        ]

        delta_t = extract_freshness_inputs(events)
        assert abs(delta_t - 20.0) < 0.1

    def test_rejects_naive_decision_timestamp(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00",
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
            },
        }

        with pytest.raises(CompatError, match="timezone offset or Z suffix"):
            extract_freshness_inputs([event])

    def test_rejects_naive_ground_truth_timestamp(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00",
            },
        }

        with pytest.raises(CompatError, match="timezone offset or Z suffix"):
            extract_freshness_inputs([event])

    def test_rejects_naive_reference_time(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00Z",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00Z",
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
            },
        }

        with pytest.raises(CompatError, match="timezone offset or Z suffix"):
            extract_freshness_inputs([event], reference_time=datetime(2026, 2, 1))

    def test_accepts_explicit_offset_timestamps(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00+00:00",
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00+00:00",
            },
        }

        delta_t = extract_freshness_inputs([event], reference_time=datetime(2026, 2, 1, tzinfo=UTC))
        assert 20 < delta_t < 22

    def test_even_number_of_events_returns_true_median(self) -> None:
        events = [
            {
                "decision_id": "d1",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-01T00:00:00Z",
                    "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
                },
            },
            {
                "decision_id": "d2",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-01T00:00:00Z",
                    "ground_truth_arrival_timestamp": "2026-01-21T00:00:00Z",
                },
            },
        ]

        delta_t = extract_freshness_inputs(events)
        assert abs(delta_t - 15.0) < 0.1

    def test_reference_time_even_number_of_events_returns_true_median(self) -> None:
        events = [
            {
                "decision_id": "d1",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-01T00:00:00Z",
                    "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
                },
            },
            {
                "decision_id": "d2",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-01T00:00:00Z",
                    "ground_truth_arrival_timestamp": "2026-01-21T00:00:00Z",
                },
            },
        ]
        ref = datetime(2026, 1, 31, tzinfo=UTC)
        delta_t = extract_freshness_inputs(events, reference_time=ref)
        assert abs(delta_t - 15.0) < 0.1

    def test_default_reference_time_uses_latest_decision_timestamp(self) -> None:
        events = [
            {
                "decision_id": "d1",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-01T00:00:00Z",
                    "ground_truth_arrival_timestamp": "2026-01-11T00:00:00Z",
                },
            },
            {
                "decision_id": "d2",
                "timestamp": "2026-01-20T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "decision_timestamp": "2026-01-20T00:00:00Z",
                    "ground_truth_arrival_timestamp": "2026-01-21T00:00:00Z",
                },
            },
        ]

        delta_t = extract_freshness_inputs(events)
        assert abs(delta_t - 9.0) < 0.1

    def test_default_reference_time_uses_top_level_timestamp_fallback(self) -> None:
        events = [
            {
                "decision_id": "d1",
                "timestamp": "2026-01-01T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "ground_truth_arrival_timestamp": "2026-01-04T00:00:00Z",
                },
            },
            {
                "decision_id": "d2",
                "timestamp": "2026-01-10T00:00:00Z",
                "decision_type": "automated",
                "temporal_metadata": {
                    "ground_truth_arrival_timestamp": "2026-01-10T00:00:00Z",
                },
            },
        ]

        delta_t = extract_freshness_inputs(events)
        assert abs(delta_t - 3.0) < 0.1

    def test_iso_timestamps_without_z_suffix(self) -> None:
        event = {
            "decision_id": "d1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "decision_type": "automated",
            "temporal_metadata": {
                "decision_timestamp": "2026-01-01T00:00:00+00:00",
                "ground_truth_arrival_timestamp": "2026-01-11T00:00:00+00:00",
            },
        }
        delta_t = extract_freshness_inputs([event])
        assert abs(delta_t - 10.0) < 0.1
