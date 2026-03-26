"""Canonical namespace for Decision Event Schema adapter helpers.

Converts DES event dicts into sufficiency dimension inputs. Requires the ``des``
optional extra: ``pip install evidence-sufficiency-calc[des]``

This module is an integration boundary, not part of the core scoring model.
Its responsibility is to validate and transform DES-shaped records into the
primitive inputs expected by the scoring functions.

Implementation note
-------------------
The public facade stays in this module so imports, monkeypatch-based tests, and
module markers remain stable. Internal responsibilities are split between
``_des_schema`` and ``_des_extract``.

DES property mapping
--------------------
- ``decision_quality_indicators.ground_truth_available`` -> completeness input
- ``temporal_metadata.{decision_timestamp, ground_truth_arrival_timestamp}`` -> freshness Dt
- ``decision_quality_indicators.confidence_score`` -> reliability proxy (when labels unavailable)
- Feature-level distribution data (external) -> representativeness input

The adapter extracts aggregate statistics from a batch of DES events,
not individual events. Sufficiency is a property of a monitoring window,
not a single decision.
"""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

from sufficiency.adapters import _des_extract, _des_schema

if TYPE_CHECKING:
    from datetime import datetime
    from importlib.resources.abc import Traversable
    from types import ModuleType

MODULE_LAYER = "adapter"
ADAPTER_NAME = "des"
CANONICAL_NAMESPACE = "sufficiency.adapters.des"


class CompatError(Exception):
    """Raised when DES events cannot be processed."""


_ValidatorLike = _des_schema.ValidatorLike
_ValidatorFactory = _des_schema.ValidatorFactory


def validate_events(events: list[dict[str, Any]]) -> list[str]:
    """Validate a batch of DES events against the JSON Schema."""
    validator = _get_validator()
    errors: list[str] = []

    for i, event in enumerate(events):
        errors.extend(f"Event {i}: {err.message}" for err in validator.iter_errors(event))

    return errors


def extract_completeness_inputs(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Extract labeled/total counts from a batch of DES events."""
    return _des_extract.extract_completeness_inputs(events)


def extract_freshness_inputs(
    events: list[dict[str, Any]],
    reference_time: datetime | None = None,
) -> float:
    """Extract median label age in days from a batch of DES events."""
    return _des_extract.extract_freshness_inputs(
        events,
        reference_time,
        error_cls=CompatError,
    )


def _schema_cache_key() -> str:
    """Return a stable cache key for the current schema path."""
    return _des_schema.schema_cache_key()


def _get_schema_resource() -> Traversable:
    """Return the packaged JSON Schema resource for DES validation."""
    return _des_schema.get_schema_resource()


def _get_validator() -> _ValidatorLike:
    """Return a cached JSON Schema validator for the current schema path."""
    jsonschema = _import_jsonschema()
    validator_cls = getattr(jsonschema, "Draft202012Validator", None)
    if validator_cls is None:
        version = getattr(jsonschema, "__version__", "unknown")
        msg = (
            "jsonschema>=4.20 is required for DES validation because the schema uses "
            f"draft 2020-12. Found version {version!r}. "
            "Install with: pip install evidence-sufficiency-calc[des]"
        )
        raise CompatError(msg)

    return _get_validator_cached(_schema_cache_key(), validator_cls)


def _import_jsonschema() -> ModuleType:
    """Import jsonschema or raise a dependency-focused adapter error."""
    return _des_schema.import_jsonschema(import_module, CompatError)


@cache
def _get_validator_cached(schema_path: str, validator_cls: _ValidatorFactory) -> _ValidatorLike:
    """Build and cache a validator for a schema path and validator class."""
    return _des_schema.build_validator(schema_path, validator_cls, _load_schema_cached)


@cache
def _load_schema_cached(schema_path: str) -> dict[str, object]:
    """Load and cache the DES JSON Schema for a resolved path."""
    return _des_schema.load_schema(schema_path, _get_schema_resource, CompatError)


__all__ = [
    "CompatError",
    "extract_completeness_inputs",
    "extract_freshness_inputs",
    "validate_events",
]
