"""Shared point-imputation metrics for *all* methods.

Only positions artificially held out from a known original measurement are
scored. Original units have different scales across physiological features;
normalized errors are the primary comparable aggregate metric here.
"""

import numpy as np


def compute_metrics(predictions, targets, eval_mask, mean=None, std=None):
    """Calculate global MAE/RMSE over eval_mask, shape (N, 48, 35).

    `predictions` are complete numerical arrays in *normalized* space.
    Targets include true held-out values, and must not be fed to imputers as
    visible observations. Empty masks are rejected rather than reporting NaN.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    mask = np.asarray(eval_mask, dtype=bool)
    if predictions.shape != targets.shape or mask.shape != targets.shape:
        raise ValueError("predictions, targets and eval_mask must share the same shape")
    if targets.ndim != 3:
        raise ValueError("expected arrays of shape (patients, hours, features)")
    if not mask.any():
        raise ValueError("eval_mask contains no withheld, known measurements")
    if not np.isfinite(predictions[mask]).all() or not np.isfinite(targets[mask]).all():
        raise ValueError("non-finite predictions or targets at evaluation positions")

    error = (predictions - targets)[mask]
    result = {
        "n_eval_points": int(mask.sum()),
        "mae_normalized": float(np.abs(error).mean()),
        "rmse_normalized": float(np.sqrt(np.square(error).mean())),
    }
    if (mean is None) != (std is None):
        raise ValueError("mean and std must both be provided or both omitted")
    if std is not None:
        mean, std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
        if mean.shape != (targets.shape[-1],) or std.shape != mean.shape:
            raise ValueError("normalizer arrays must match the feature dimension")
        # Error in original units; the mean cancels when computing differences.
        # NOTE: mixing clinical units heavily weights large-scale attributes.
        original_error = ((predictions - targets) * std)[mask]
        result["mae_original_mixed_units"] = float(np.abs(original_error).mean())
        result["rmse_original_mixed_units"] = float(np.sqrt(np.square(original_error).mean()))
    return result
