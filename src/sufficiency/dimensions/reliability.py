"""Reliability dimension: accuracy of evidence against ground truth.

R(t) = F1-score on retroactively labeled data.

Confidence interval via bootstrap resampling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sufficiency._validation import coerce_binary_labels
from sufficiency.types import DimensionScore

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


_SMALL_SAMPLE_BATCH_SIZE = 512
_LARGE_SAMPLE_BATCH_SIZE = 64
_SMALL_SAMPLE_THRESHOLD = 2_048


def compute_reliability(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    rng_seed: int | None = None,
) -> DimensionScore:
    """Compute reliability as F1-score with bootstrap confidence interval.

    Args:
        y_true: True binary labels (0/1) from retroactively confirmed outcomes.
        y_pred: Predicted binary labels (0/1) from the model.
        n_bootstrap: Number of bootstrap resamples for CI (default 1000).
        confidence_level: Confidence level for the interval (default 0.95).
        rng_seed: Optional random seed for reproducibility.

    Returns
    -------
        DimensionScore with value = F1-score on the provided data.

    Raises
    ------
        ValueError: If arrays have different lengths or are empty.
    """
    y_true_arr = coerce_binary_labels("y_true", y_true)
    y_pred_arr = coerce_binary_labels("y_pred", y_pred)

    if y_true_arr.shape != y_pred_arr.shape:
        msg = f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
        raise ValueError(msg)

    if n_bootstrap <= 0:
        msg = f"n_bootstrap must be positive, got {n_bootstrap}"
        raise ValueError(msg)

    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}"
        raise ValueError(msg)

    if len(y_true_arr) == 0:
        return DimensionScore(
            value=0.0,
            confidence_low=0.0,
            confidence_high=1.0,
            label="reliability",
        )

    f1 = _f1_score(y_true_arr, y_pred_arr)
    ci_low, ci_high = _bootstrap_ci(y_true_arr, y_pred_arr, n_bootstrap, confidence_level, rng_seed)

    return DimensionScore(
        value=float(f1),
        confidence_low=max(0.0, float(ci_low)),
        confidence_high=min(1.0, float(ci_high)),
        label="reliability",
    )


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1-score without sklearn dependency."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
    rng_seed: int | None,
) -> tuple[float, float]:
    """Bootstrap confidence interval for F1-score."""
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    batch_size = _bootstrap_batch_size(n, n_bootstrap)

    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        idx = rng.integers(0, n, size=(stop - start, n))
        batch_true = y_true[idx]
        batch_pred = y_pred[idx]

        tp = np.sum((batch_true == 1) & (batch_pred == 1), axis=1, dtype=np.int64)
        fp = np.sum((batch_true == 0) & (batch_pred == 1), axis=1, dtype=np.int64)
        fn = np.sum((batch_true == 1) & (batch_pred == 0), axis=1, dtype=np.int64)
        denominator = 2 * tp + fp + fn

        scores[start:stop] = np.divide(
            2 * tp,
            denominator,
            out=np.zeros(stop - start, dtype=np.float64),
            where=denominator > 0,
        )

    alpha = 1.0 - confidence_level
    return float(np.percentile(scores, 100 * alpha / 2)), float(
        np.percentile(scores, 100 * (1 - alpha / 2))
    )


def _bootstrap_batch_size(n_samples: int, n_bootstrap: int) -> int:
    """Choose a bootstrap batch size that balances CPU overhead and memory pressure."""
    preferred = (
        _SMALL_SAMPLE_BATCH_SIZE
        if n_samples <= _SMALL_SAMPLE_THRESHOLD
        else _LARGE_SAMPLE_BATCH_SIZE
    )
    return min(preferred, n_bootstrap)
