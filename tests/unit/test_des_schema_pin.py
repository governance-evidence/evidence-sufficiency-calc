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
    "365585a04f3f3913d5fc5bfd7708569054aa56a80388b89152e48cc98d147282"  # pragma: allowlist secret
)


def test_bundled_des_schema_matches_pinned_hash() -> None:
    """The bundled schema mirrors decision-event-schema v0.3.1.

    If this fails, the bundled copy changed: re-sync it intentionally from
    the decision-event-schema repository, update fixtures and this pin in
    the same change, and note the sync in the changelog.
    """
    digest = hashlib.sha256(BUNDLED_SCHEMA.read_bytes()).hexdigest()
    assert digest == PINNED_SHA256
