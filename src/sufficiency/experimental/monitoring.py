"""Canonical namespace for experimental monitoring primitives."""

from __future__ import annotations

from sufficiency.experimental.evalue import EVALUE_ACCUMULATOR_STABILITY, EValueAccumulator

API_STABILITY = EVALUE_ACCUMULATOR_STABILITY
CANONICAL_NAMESPACE = "sufficiency.experimental.monitoring"

__all__ = ["EValueAccumulator"]
