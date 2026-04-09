"""Shared test fixtures for the evidence-sufficiency-calc test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sufficiency.types import DimensionScore, GovernanceConfig

DES_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "des"


# ---------------------------------------------------------------------------
# DES event fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def des_fixture_dir() -> Path:
    """Path to the DES example fixtures directory."""
    return DES_FIXTURES_DIR


@pytest.fixture
def knight_capital_event() -> dict[str, Any]:
    """Knight Capital 2012 DES event."""
    return _load_des_fixture("knight-capital-2012.json")


@pytest.fixture
def cloudflare_event() -> dict[str, Any]:
    """Cloudflare 2025 DES event."""
    return _load_des_fixture("cloudflare-2025.json")


@pytest.fixture
def all_des_events() -> list[dict[str, Any]]:
    """All DES example events from fixtures directory."""
    return [_load_des_fixture(f.name) for f in sorted(DES_FIXTURES_DIR.glob("*.json"))]


@pytest.fixture
def minimal_valid_event() -> dict[str, Any]:
    """Minimal valid DES event with only required fields."""
    return {
        "schema_version": "0.3.0",
        "decision_id": "test-minimal",
        "timestamp": "2026-01-01T00:00:00Z",
        "decision_type": "automated",
        "decision_context": {
            "decision_id": "test-minimal",
            "decision_type": "fraud_detection",
        },
        "decision_logic": {
            "logic_type": "rule_based",
            "output": "approve",
        },
        "human_override_record": {
            "override_occurred": False,
        },
        "temporal_metadata": {
            "event_timestamp": "2026-01-01T00:00:00Z",
            "decision_timestamp": "2026-01-01T00:00:00Z",
            "sequence_number": 1,
            "hash_chain": {
                "previous_hash": None,
                "current_hash": "test-minimal-hash",
                "algorithm": "SHA-256",
            },
            "evidence_tier": "lightweight",
        },
    }


# ---------------------------------------------------------------------------
# Dimension score fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_scores() -> dict[str, DimensionScore]:
    """All dimensions at 1.0."""
    return {
        "completeness": DimensionScore(1.0, 1.0, 1.0, "completeness"),
        "freshness": DimensionScore(1.0, 1.0, 1.0, "freshness"),
        "reliability": DimensionScore(1.0, 1.0, 1.0, "reliability"),
        "representativeness": DimensionScore(1.0, 1.0, 1.0, "representativeness"),
    }


@pytest.fixture
def degraded_scores() -> dict[str, DimensionScore]:
    """Scores representing a degraded system (completeness low, freshness stale)."""
    return {
        "completeness": DimensionScore(0.3, 0.2, 0.4, "completeness"),
        "freshness": DimensionScore(0.4, 0.3, 0.5, "freshness"),
        "reliability": DimensionScore(0.85, 0.8, 0.9, "reliability"),
        "representativeness": DimensionScore(0.9, 0.85, 0.95, "representativeness"),
    }


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_governance_config() -> GovernanceConfig:
    """Default governance config with equal weights."""
    from sufficiency.config import default_config

    return default_config()


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic numpy random generator for reproducible tests."""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers (not fixtures)
# ---------------------------------------------------------------------------


def _load_des_fixture(name: str) -> dict[str, Any]:
    """Load a DES fixture JSON file."""
    return cast("dict[str, Any]", json.loads((DES_FIXTURES_DIR / name).read_text()))
