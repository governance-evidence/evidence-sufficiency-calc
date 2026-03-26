"""Internal validation helpers shared across sufficiency modules.

This module is intentionally internal-only. It centralizes low-level input
contract checks so public modules can share behavior without duplicating
validation logic or drifting in error semantics.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import ArrayLike


def require_unit_interval(name: str, value: float) -> None:
    """Require a finite numeric value in the closed unit interval."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        msg = f"{name} must be in [0, 1], got {value}"
        raise ValueError(msg)


def require_aware_datetime(
    value: datetime,
    *,
    message: str,
    error_cls: type[Exception] = ValueError,
) -> datetime:
    """Require a timezone-aware datetime and preserve caller-specific errors."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise error_cls(message)
    return value


def coerce_binary_labels(name: str, values: ArrayLike) -> np.ndarray:
    """Validate binary labels without silently truncating non-binary numerics."""
    arr = np.asarray(values)

    if arr.dtype.kind in {"U", "S"}:
        msg = f"{name} must contain only binary labels 0/1"
        raise ValueError(msg)

    if arr.dtype.kind == "O":
        flat_values = np.ravel(arr)
        if not all(
            isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))
            for value in flat_values
        ):
            msg = f"{name} must contain only binary labels 0/1"
            raise ValueError(msg)

    try:
        numeric_arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must contain only binary labels 0/1"
        raise ValueError(msg) from exc

    if not np.isfinite(numeric_arr).all():
        msg = f"{name} must contain only finite binary labels 0/1"
        raise ValueError(msg)

    if not np.all((numeric_arr == 0.0) | (numeric_arr == 1.0)):
        unique_values = sorted({float(value) for value in np.ravel(numeric_arr).tolist()})
        msg = f"{name} must contain only binary labels 0/1, got {unique_values}"
        raise ValueError(msg)

    return numeric_arr.astype(np.int64, copy=False)


def coerce_1d_float_array(name: str, values: ArrayLike) -> np.ndarray:
    """Coerce a numeric sample to a one-dimensional float array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        msg = f"{name} must be a one-dimensional numeric sample"
        raise ValueError(msg)
    return arr
