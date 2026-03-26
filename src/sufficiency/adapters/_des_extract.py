"""Internal extraction helpers for the Decision Event Schema adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sufficiency._validation import require_aware_datetime

_TIMEZONE_MESSAGE = "DES timestamps must include a timezone offset or Z suffix"


def extract_completeness_inputs(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Extract labeled/total counts from a batch of DES events."""
    total = len(events)
    labeled = sum(
        1
        for event in events
        if event.get("decision_quality_indicators", {}).get("ground_truth_available") is True
    )
    return labeled, total


def extract_freshness_inputs(
    events: list[dict[str, Any]],
    reference_time: datetime | None = None,
    *,
    error_cls: type[Exception],
) -> float:
    """Extract median label age in days from a batch of DES events."""
    if reference_time is None:
        reference_time = _latest_decision_time(events, error_cls=error_cls)
    else:
        reference_time = require_aware_datetime(
            reference_time,
            message=_TIMEZONE_MESSAGE,
            error_cls=error_cls,
        )

    labeled_ages: list[float] = []

    for event in events:
        temporal = event.get("temporal_metadata", {})
        gt_ts = temporal.get("ground_truth_arrival_timestamp")
        dec_ts = temporal.get("decision_timestamp") or event.get("timestamp")

        if gt_ts is None or dec_ts is None:
            continue

        gt_time = _parse_iso(gt_ts, error_cls=error_cls)
        dec_time = _parse_iso(dec_ts, error_cls=error_cls)
        age_days = (gt_time - dec_time).total_seconds() / 86400.0

        if age_days >= 0:
            labeled_ages.append(age_days)

    if not labeled_ages:
        return 0.0

    assert reference_time is not None

    ages_from_ref: list[float] = []
    for event in events:
        temporal = event.get("temporal_metadata", {})
        gt_ts = temporal.get("ground_truth_arrival_timestamp")
        if gt_ts is None:
            continue
        gt_time = _parse_iso(gt_ts, error_cls=error_cls)
        age = (reference_time - gt_time).total_seconds() / 86400.0
        if age >= 0:
            ages_from_ref.append(age)
    if ages_from_ref:
        return _median(ages_from_ref)

    return _median(labeled_ages)


def _latest_decision_time(
    events: list[dict[str, Any]],
    *,
    error_cls: type[Exception],
) -> datetime | None:
    """Return the latest decision timestamp available in a batch."""
    decision_times: list[datetime] = []

    for event in events:
        temporal = event.get("temporal_metadata", {})
        decision_ts = temporal.get("decision_timestamp") or event.get("timestamp")
        if decision_ts is None:
            continue
        decision_times.append(_parse_iso(decision_ts, error_cls=error_cls))

    if not decision_times:
        return None

    return max(decision_times)


def _median(values: list[float]) -> float:
    """Return the mathematical median of a non-empty list."""
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _parse_iso(ts: str, *, error_cls: type[Exception]) -> datetime:
    """Parse ISO 8601 timestamp, handling Z suffix."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return require_aware_datetime(
        datetime.fromisoformat(ts),
        message=_TIMEZONE_MESSAGE,
        error_cls=error_cls,
    )
