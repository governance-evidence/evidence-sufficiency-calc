"""Example: validate Decision Event Schema records and score a monitoring window.

Demonstrates the adapter boundary end-to-end: validate a DES-shaped batch,
extract completeness/freshness inputs, then combine them with explicit proxy
scores for the remaining dimensions to compute an overall sufficiency result.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sufficiency import DimensionScore, compute_sufficiency, default_config
from sufficiency.adapters.des import (
    extract_completeness_inputs,
    extract_freshness_inputs,
    validate_events,
)
from sufficiency.dimensions.completeness import compute_completeness
from sufficiency.dimensions.freshness import compute_freshness


def _mean_confidence(events: list[dict[str, object]]) -> float:
    scores: list[float] = []
    for event in events:
        indicators = event.get("decision_quality_indicators", {})
        if not isinstance(indicators, dict):
            continue
        confidence = indicators.get("confidence_score")
        if isinstance(confidence, float):
            scores.append(confidence)

    if not scores:
        return 0.75

    return sum(scores) / len(scores)


def _make_des_event(
    event_id: str,
    *,
    decision_time: str,
    domain: str,
    logic_type: str,
    output: object,
    ground_truth_available: bool,
    confidence_score: float,
    ground_truth_arrival_timestamp: str | None = None,
    sequence_number: int = 1,
) -> dict[str, object]:
    event: dict[str, object] = {
        "schema_version": "0.3.0",
        "decision_id": event_id,
        "timestamp": decision_time,
        "decision_type": "automated" if logic_type != "human_decision" else "human",
        "decision_context": {
            "decision_id": event_id,
            "decision_type": domain,
        },
        "decision_logic": {
            "logic_type": logic_type,
            "output": output,
        },
        "decision_quality_indicators": {
            "ground_truth_available": ground_truth_available,
            "confidence_score": confidence_score,
        },
        "human_override_record": {
            "override_occurred": False,
            "override_actor": {
                "actor_id": "analyst-01" if logic_type == "human_decision" else "system",
                "actor_role": "analyst" if logic_type == "human_decision" else "service",
                "authorization_level": "standard",
            },
            "override_rationale": "Recorded decision event",
        },
        "temporal_metadata": {
            "event_timestamp": decision_time,
            "decision_timestamp": decision_time,
            "sequence_number": sequence_number,
            "hash_chain": {
                "previous_hash": None,
                "current_hash": f"{event_id}-hash",
                "algorithm": "SHA-256",
            },
            "evidence_tier": "lightweight",
        },
    }
    temporal_metadata = event["temporal_metadata"]
    assert isinstance(temporal_metadata, dict)
    if ground_truth_arrival_timestamp is not None:
        temporal_metadata["ground_truth_arrival_timestamp"] = ground_truth_arrival_timestamp
    return event


def main() -> None:
    config = default_config()

    events: list[dict[str, object]] = [
        _make_des_event(
            "des-example-001",
            decision_time="2026-01-02T09:15:00Z",
            domain="fraud_detection",
            logic_type="ml_inference",
            output=0.91,
            ground_truth_available=True,
            confidence_score=0.91,
            ground_truth_arrival_timestamp="2026-01-06T09:15:00Z",
            sequence_number=1,
        ),
        _make_des_event(
            "des-example-002",
            decision_time="2026-01-05T14:40:00Z",
            domain="manual_case_review",
            logic_type="human_decision",
            output="approve_after_review",
            ground_truth_available=True,
            confidence_score=0.84,
            ground_truth_arrival_timestamp="2026-01-12T14:40:00Z",
            sequence_number=2,
        ),
        _make_des_event(
            "des-example-003",
            decision_time="2026-01-08T18:05:00Z",
            domain="fraud_detection",
            logic_type="rule_based",
            output="queue_for_review",
            ground_truth_available=False,
            confidence_score=0.79,
            sequence_number=3,
        ),
    ]

    errors = validate_events(events)
    if errors:
        raise SystemExit("Schema validation failed:\n- " + "\n- ".join(errors))

    labeled_count, total_count = extract_completeness_inputs(events)
    freshness_days = extract_freshness_inputs(
        events,
        reference_time=datetime(2026, 1, 20, tzinfo=UTC),
    )

    reliability_proxy = _mean_confidence(events)
    representativeness_proxy = 0.88

    dimensions = {
        "completeness": compute_completeness(labeled_count, total_count),
        "freshness": compute_freshness(freshness_days, config.lambda_freshness),
        "reliability": DimensionScore(
            reliability_proxy,
            reliability_proxy,
            reliability_proxy,
            "reliability",
        ),
        "representativeness": DimensionScore(
            representativeness_proxy,
            representativeness_proxy,
            representativeness_proxy,
            "representativeness",
        ),
    }

    result = compute_sufficiency(dimensions, config)

    print("=== DES Adapter Example ===\n")
    print(f"Validated events: {total_count}")
    print(f"Ground truth available: {labeled_count}/{total_count}")
    print(f"Median freshness age: {freshness_days:.1f} days")
    print(f"Reliability proxy: {reliability_proxy:.2f}")
    print(f"Representativeness proxy: {representativeness_proxy:.2f}")
    print()
    print(f"Composite score: {result.composite:.3f}")
    print(f"Gate A(t):       {result.gate:.3f}")
    print(f"Status:          {result.status.value}")


if __name__ == "__main__":
    main()
