"""Pin the bundled decision-event schema so updates are intentional."""

from __future__ import annotations

import hashlib
from pathlib import Path

BUNDLED_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "sufficiency"
    / "adapters"
    / "schemas"
    / "decision-event.schema.json"
)
PINNED_SHA256 = (
    "84d665324b4c22286ce6bfeda32e4099579e2dce5b0a744b2b5776dc6b7342d1"  # pragma: allowlist secret
)


def test_bundled_des_schema_matches_pinned_hash() -> None:
    """The bundled schema mirrors decision-event-schema v0.3.0.

    If this fails, the bundled copy changed: re-sync it intentionally from
    the decision-event-schema repository, update fixtures and this pin in
    the same change, and note the sync in the changelog.
    """
    digest = hashlib.sha256(BUNDLED_SCHEMA.read_bytes()).hexdigest()
    assert digest == PINNED_SHA256
