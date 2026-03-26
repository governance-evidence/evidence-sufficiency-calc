"""Sequential monitoring primitives for sufficiency assessments.

Stable: ThresholdMonitor — daily comparison against governance thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sufficiency.experimental import evalue as _experimental_evalue
from sufficiency.types import SufficiencyResult, SufficiencyStatus, SufficiencyThresholds

if TYPE_CHECKING:
    from datetime import datetime


THRESHOLD_MONITOR_STABILITY = "stable"
# Backward-compatible re-export: the experimental implementation now lives
# under sufficiency.experimental.evalue, but existing imports from
# sufficiency.sequential remain supported for local callers.
EVALUE_ACCUMULATOR_STABILITY = _experimental_evalue.EVALUE_ACCUMULATOR_STABILITY
EValueAccumulator = _experimental_evalue.EValueAccumulator


@dataclass(frozen=True)
class Alert:
    """Sufficiency threshold breach alert.

    Attributes
    ----------
        timestamp: When the breach was detected.
        composite_score: The composite score that triggered the alert.
        status: Governance status at the time of the alert.
        message: Human-readable alert description.
    """

    timestamp: datetime
    composite_score: float
    status: SufficiencyStatus
    message: str


@dataclass
class ThresholdMonitor:
    """Daily threshold-based sufficiency monitoring.

    Tracks sufficiency scores over time and emits alerts when thresholds
    are crossed. This is the monitoring approach evaluated in Paper 14.

    Attributes
    ----------
        thresholds: Governance sufficiency thresholds.
        history: Sequence of observed SufficiencyResults.
        alerts: Emitted alerts.
    """

    thresholds: SufficiencyThresholds = field(default_factory=SufficiencyThresholds)
    history: list[SufficiencyResult] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    _prev_status: SufficiencyStatus | None = field(default=None, repr=False)

    def observe(self, result: SufficiencyResult) -> Alert | None:
        """Record a new sufficiency observation and check for threshold crossings.

        Args:
            result: Latest sufficiency assessment.

        Returns
        -------
            Alert if a threshold was crossed, None otherwise.
        """
        self.history.append(result)
        current_status = self.thresholds.classify(result.composite)

        alert: Alert | None = None

        if (
            self._prev_status is not None
            and current_status != self._prev_status
            and _severity(current_status) > _severity(self._prev_status)
        ):
            alert = Alert(
                timestamp=result.timestamp,
                composite_score=result.composite,
                status=current_status,
                message=(
                    f"Sufficiency degraded from {self._prev_status.value} "
                    f"to {current_status.value} "
                    f"(S={result.composite:.3f})"
                ),
            )
            self.alerts.append(alert)

        self._prev_status = current_status
        return alert


def _severity(status: SufficiencyStatus) -> int:
    """Map status to severity level (higher = worse)."""
    return {
        SufficiencyStatus.SUFFICIENT: 0,
        SufficiencyStatus.DEGRADED: 1,
        SufficiencyStatus.INSUFFICIENT: 2,
    }[status]
