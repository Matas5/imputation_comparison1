"""Patient-level splitting. Folds match CSDI's get_dataloader() indexing."""

import numpy as np


def split_patient_indices(
    n_patients: int, fold: int = 0, seed: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, test indices for one of the five folds.

    SAME AS ORIGINAL: shuffled 20% test, ~70% train, remaining ~10% val.
    We use local RandomState objects to reproduce the original seed-dependent
    NumPy shuffle without changing the global NumPy random state.
    """
    if not 0 <= fold < 5:
        raise ValueError("fold must be one of 0, 1, 2, 3, 4")
    if n_patients < 5:
        raise ValueError("Need at least five patients for five-fold splitting")

    shuffled = np.arange(n_patients)
    np.random.RandomState(seed).shuffle(shuffled)

    start = int(fold * 0.2 * n_patients)
    end = int((fold + 1) * 0.2 * n_patients)
    test = shuffled[start:end].copy()
    remaining = np.delete(shuffled, np.arange(start, end))

    # SAME AS ORIGINAL: reseed before shuffling the non-test patients.
    np.random.RandomState(seed).shuffle(remaining)
    train_count = int(n_patients * 0.7)
    train = remaining[:train_count].copy()
    valid = remaining[train_count:].copy()

    if not len(valid) or not len(test):
        raise ValueError("Validation/test split is empty")
    return train, valid, test
