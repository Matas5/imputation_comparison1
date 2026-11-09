"""Generate reproducible artificial MCAR-style evaluation masks."""

import numpy as np


def generate_mcar_input_mask(
    observed_mask: np.ndarray, missing_ratio: float = 0.1, seed: int = 1
) -> np.ndarray:
    """Hide floor(ratio * observed_count) values independently per patient.

    SAME AS ORIGINAL: draw without replacement from REAL observed positions;
    choose a fixed proportion per patient, not per whole dataset. This is
    conditional random masking of observed values (MCAR-style benchmark).

    NumPy RandomState preserves CSDI's original random-choice algorithm,
    provided patient order and successfully parsed patients stay the same.
    CHANGE: independent local RNG, so global random state remains unchanged.
    """
    if observed_mask.ndim != 3:
        raise ValueError("Expected (patients, hours, features) observed mask")
    if not 0 <= missing_ratio < 1:
        raise ValueError("missing_ratio must be in [0,1)")
    rng = np.random.RandomState(seed)
    input_mask = observed_mask.astype(bool, copy=True)

    for patient in input_mask:
        flat = patient.reshape(-1)  # view: edits original patient's mask
        observed_indices = np.flatnonzero(flat)
        count = int(len(observed_indices) * missing_ratio)
        # SAME AS ORIGINAL: use choice even when count=0; keeps RNG behaviour.
        hide = rng.choice(observed_indices, size=count, replace=False)
        flat[hide] = False

    return input_mask


def get_eval_mask(observed_mask: np.ndarray, input_mask: np.ndarray) -> np.ndarray:
    """True only where a real recorded value was artificially hidden."""
    if observed_mask.shape != input_mask.shape:
        raise ValueError("Mask shapes must match")
    if np.any(input_mask & ~observed_mask):
        raise ValueError("Input mask cannot reveal a naturally missing position")
    return observed_mask & ~input_mask
