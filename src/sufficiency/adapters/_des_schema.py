"""Internal schema and compatibility helpers for the DES adapter."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from importlib.resources.abc import Traversable
    from types import ModuleType


class ValidationErrorLike(Protocol):
    """Minimal protocol for schema validation errors."""

    message: str


class ValidatorLike(Protocol):
    """Minimal protocol for schema validators."""

    def iter_errors(
        self, instance: object
    ) -> Iterable[ValidationErrorLike]: ...  # pragma: no cover


class ValidatorFactory(Protocol):
    """Minimal protocol for validator constructors."""

    def __call__(self, schema: dict[str, object]) -> ValidatorLike: ...  # pragma: no cover


SCHEMA_RESOURCE_PARTS = ("schemas", "decision-event.schema.json")


def schema_cache_key() -> str:
    """Return a stable cache key for the current schema path."""
    return f"sufficiency.adapters:{'/'.join(SCHEMA_RESOURCE_PARTS)}"


def get_schema_resource() -> Traversable:
    """Return the packaged JSON Schema resource for DES validation."""
    return files("sufficiency.adapters").joinpath(*SCHEMA_RESOURCE_PARTS)


def import_jsonschema(
    import_module_func: Callable[[str], ModuleType],
    compat_error_cls: type[Exception],
) -> ModuleType:
    """Import jsonschema or raise a dependency-focused adapter error."""
    try:
        jsonschema = import_module_func("jsonschema")
    except ImportError as exc:
        msg = (
            "jsonschema is required for DES validation. "
            "Install with: pip install evidence-sufficiency-calc[des]"
        )
        raise compat_error_cls(msg) from exc

    return jsonschema


def build_validator(
    schema_path: str,
    validator_cls: ValidatorFactory,
    load_schema_func: Callable[[str], dict[str, object]],
) -> ValidatorLike:
    """Build a validator for a schema path and validator class."""
    schema = load_schema_func(schema_path)
    return validator_cls(schema)


def load_schema(
    schema_path: str,
    get_schema_resource_func: Callable[[], Traversable],
    compat_error_cls: type[Exception],
) -> dict[str, object]:
    """Load the DES JSON Schema for a resolved path."""
    resource = get_schema_resource_func()
    if not resource.is_file():
        msg = (
            f"Packaged DES schema not found for {schema_path}. "
            "Reinstall the package or check setuptools package-data configuration."
        )
        raise compat_error_cls(msg)
    return cast("dict[str, object]", json.loads(resource.read_text(encoding="utf-8")))
