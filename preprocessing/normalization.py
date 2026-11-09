"""Fit per-feature Z-score statistics on naturally observed TRAIN values only."""

import numpy as np


def fit_normalizer(
    train_values: np.ndarray, train_observed_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-feature mean, population std (ddof=0), and counts.

    CHANGED FROM ORIGINAL CSDI: the old code fit mean/std using ALL patients,
    before train/test splitting (test leakage). Only training patients are
    allowed to determine this normalizer. Scores may therefore change.
    """
    if train_values.shape != train_observed_mask.shape or train_values.ndim != 3:
        raise ValueError("Expected matching (patients, 48, 35) arrays")
    n_features = train_values.shape[-1]
    means = np.empty(n_features, dtype=np.float64)
    stds = np.empty(n_features, dtype=np.float64)
    counts = np.empty(n_features, dtype=np.int64)

    for feature in range(n_features):
        known = train_values[..., feature][train_observed_mask[..., feature]]
        counts[feature] = len(known)
        if len(known) == 0:
            raise ValueError(f"No training observations for feature {feature}")
        means[feature] = known.mean()
        # SAME AS ORIGINAL: np.std uses population std, with ddof=0.
        stds[feature] = known.std(ddof=0)
        # CHANGE: guard constant features against division by zero / NaN.
        # Their standardized observed values become 0 by construction.
        if stds[feature] < 1e-12:
            stds[feature] = 1.0

    return means, stds, counts


def transform(
    values: np.ndarray,
    observed_mask: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Apply train-fitted stats to ALL splits, then put 0 at truly missing positions."""
    if values.shape != observed_mask.shape:
        raise ValueError("values and observed_mask must have identical shapes")
    if values.shape[-1] != len(means) or len(means) != len(stds):
        raise ValueError("Expected one mean/std per feature")

    standardized = (values - means) / stds
    # CHANGE: preserve NaN until normalization, then use 0 only as a numerical
    # placeholder at naturally absent values. Masks still say these are missing.
    result = np.where(observed_mask, standardized, 0.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite normalized values")
    return result


def inverse_transform(normalized: np.ndarray, means: np.ndarray, stds: np.ndarray):
    """Recover original units for plotting or original-unit RMSE/MAE."""
    return normalized * stds + means
