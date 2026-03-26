"""Experimental heuristic e-value accumulation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

EVALUE_ACCUMULATOR_STABILITY = "experimental"


@dataclass
class EValueAccumulator:
    """Experimental heuristic e-value-style accumulation for sufficiency monitoring.

    Scaffold implementation for accumulating a monotone evidence signal when
    scores repeatedly fall below a governance threshold.

    The current update rule is heuristic. It is useful for ranking or alerting,
    but it does not by itself establish an anytime-valid e-process or formal
    Type I error guarantees.

    Attributes
    ----------
        threshold: Null hypothesis threshold (sufficiency >= threshold).
        alpha: Significance level (default 0.05).
        log_e_value: Accumulated log e-value (starts at 0 = e-value of 1).
    """

    threshold: float = 0.8
    alpha: float = 0.05
    log_e_value: float = 0.0
    _observations: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            msg = f"threshold must be in (0, 1), got {self.threshold}"
            raise ValueError(msg)
        if not 0.0 < self.alpha < 1.0:
            msg = f"alpha must be in (0, 1), got {self.alpha}"
            raise ValueError(msg)

    def observe(self, score: float) -> bool:
        """Accumulate a heuristic signal against the configured threshold.

        Args:
            score: Observed composite sufficiency score.

        Returns
        -------
            True if the accumulated score exceeds the alert cutoff 1/alpha.
        """
        if not 0.0 <= score <= 1.0:
            msg = f"score must be in [0, 1], got {score}"
            raise ValueError(msg)

        self._observations += 1

        # Simple betting strategy: bet proportional to deviation from threshold
        # e_t = 1 + lambda * (threshold - score) when score < threshold
        bet_fraction = 0.5  # conservative bet size
        if score < self.threshold:
            e_t = 1.0 + bet_fraction * (self.threshold - score) / self.threshold
        else:
            e_t = 1.0 - bet_fraction * (score - self.threshold) / (1.0 - self.threshold + 1e-10)
            e_t = max(e_t, 0.01)  # prevent log of zero

        import math

        self.log_e_value += math.log(e_t)

        # Trigger when the accumulated score exceeds the configured cutoff.
        return self.log_e_value >= math.log(1.0 / self.alpha)

    @property
    def e_value(self) -> float:
        """Current accumulated e-value."""
        import math
        import sys

        if self.log_e_value >= math.log(sys.float_info.max):
            return float("inf")
        return math.exp(self.log_e_value)

    @property
    def rejected(self) -> bool:
        """Whether H0 has been rejected."""
        import math

        return self.log_e_value >= math.log(1.0 / self.alpha)
