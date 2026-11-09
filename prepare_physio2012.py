"""Prepare reproducible, model-independent PhysioNet 2012 Set A experiments.

Example from project root:
    python prepare_physio2012.py --folds 0 --ratios 0.1 0.3 0.5 --seed 1
    python prepare_physio2012.py --folds 0 1 2 3 4 --ratios 0.1 --seed 1

Requires numpy and pandas; PyTorch is NOT needed to prepare the data.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from datasets.physio2012 import ATTRIBUTES, HOURS, load_physio2012
from preprocessing.splitting import split_patient_indices
from preprocessing.normalization import fit_normalizer, transform
from preprocessing.missingness import generate_mcar_input_mask, get_eval_mask

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "data" / "raw" / "physio2012" / "set-a"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "physio2012"


def ratio_dir(ratio: float) -> str:
    """Stable directory name, e.g. mcar_0.1, mcar_0.3."""
    return f"mcar_{ratio:g}"


def save_split(path: Path, indices: np.ndarray, values: np.ndarray,
               observed_mask: np.ndarray, input_mask: np.ndarray,
               patient_ids: np.ndarray) -> None:
    """Save model-agnostic arrays with both truth and masks.

    IMPORTANT: values includes TRUE values even where the evaluation input
    mask hides them! This is intentional, for computing errors against truth.
    Non-CSDI methods MUST construct their inputs using `input_mask` (see README)
    or they would accidentally see the held-out answers (data leakage).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    observed = observed_mask[indices]
    inputs = input_mask[indices]
    evaluation = get_eval_mask(observed, inputs)
    np.savez_compressed(
        path,
        values=values[indices],
        observed_mask=observed,
        input_mask=inputs,
        eval_mask=evaluation,
        patient_ids=patient_ids[indices],
    )


def prepare(raw_dir: Path, output_dir: Path, folds: list[int],
            ratios: list[float], seed: int, reparse: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_file = output_dir / "parsed.npz"

    # CHANGE: cache original parsed arrays separately from the random masks
    # and normalization. In original CSDI these are coupled in one pickle.
    if parsed_file.exists() and not reparse:
        print(f"Using cached parsed dataset: {parsed_file}")
        with np.load(parsed_file, allow_pickle=False) as saved:
            original_values = saved["values"]
            observed_mask = saved["observed_mask"].astype(bool)
            patient_ids = saved["patient_ids"]
            if list(saved["attributes"]) != ATTRIBUTES:
                raise ValueError("Cached feature order differs; use --reparse")
    else:
        print(f"Parsing patient records in {raw_dir} ...")
        original_values, observed_mask, patient_ids = load_physio2012(raw_dir)
        np.savez_compressed(
            parsed_file,
            values=original_values,  # NaN indicates natural missingness
            observed_mask=observed_mask,
            patient_ids=patient_ids,
            attributes=np.asarray(ATTRIBUTES),
        )

    if original_values.ndim != 3 or original_values.shape[1:] != (HOURS, len(ATTRIBUTES)):
        raise ValueError("Unexpected dataset dimensions")
    if not np.array_equal(~np.isnan(original_values), observed_mask):
        raise ValueError("Observed mask disagrees with NaNs in parsed values")

    # CHANGE: fixed evaluation masks use their own seed, independent of folds.
    # This way a given patient/ratio gets the same hidden positions in every
    # fold; the original CSDI generated masks once before splitting too.
    masks_by_ratio = {
        ratio: generate_mcar_input_mask(observed_mask, ratio, seed)
        for ratio in ratios
    }

    for fold in folds:
        train_ix, valid_ix, test_ix = split_patient_indices(len(patient_ids), fold, seed)
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # CHANGE vs original: ONLY training patients contribute to statistics.
        # Original calculated mean/std over train+validation+test patients;
        # removing that leakage WILL change normalized values and scores.
        mean, std, counts = fit_normalizer(
            original_values[train_ix], observed_mask[train_ix]
        )

        # SAME stats for EVERY patient and every mask ratio in this fold.
        normalized = transform(original_values, observed_mask, mean, std)

        np.savez_compressed(
            fold_dir / "normalizer.npz",
            mean=mean, std=std, observed_count=counts,
            attributes=np.asarray(ATTRIBUTES),
        )
        np.savez_compressed(
            fold_dir / "splits.npz",
            train_indices=train_ix, valid_indices=valid_ix, test_indices=test_ix,
            train_ids=patient_ids[train_ix], valid_ids=patient_ids[valid_ix],
            test_ids=patient_ids[test_ix],
        )

        # SAME AS CSDI'S TRAINING BEHAVIOUR: the artificial test masks are not
        # passed in as its training mask; CSDI samples training masks inside
        # main_model.py. Keep train input_mask = observed_mask for that purpose.
        save_split(
            fold_dir / "train.npz", train_ix, normalized,
            observed_mask, observed_mask, patient_ids,
        )

        for ratio, full_input_mask in masks_by_ratio.items():
            mask_dir = fold_dir / ratio_dir(ratio)
            save_split(
                mask_dir / "valid.npz", valid_ix, normalized,
                observed_mask, full_input_mask, patient_ids,
            )
            save_split(
                mask_dir / "test.npz", test_ix, normalized,
                observed_mask, full_input_mask, patient_ids,
            )
            print(
                f"fold={fold}, ratio={ratio:g}: "
                f"train={len(train_ix)}, valid={len(valid_ix)}, test={len(test_ix)}"
            )

        metadata = {
            "dataset": "PhysioNet Challenge 2012 - Set A",
            "n_patients": len(patient_ids),
            "shape_per_patient": [HOURS, len(ATTRIBUTES)],
            "features": ATTRIBUTES,
            "fold": fold,
            "seed": seed,
            "missing_ratios": ratios,
            "normalization": "Z-score, ddof=0, training observed measurements only",
            "split": "Original CSDI 70/10/20 shuffled patient-level scheme",
            "missingness": "Per-patient fixed-count MCAR-style masking of observed entries",
            "training_mask": "Unmodified; CSDI generates its own inside the model",
            "outcome_changing_notes": [
                "Training-only statistics replace original all-patient normalization",
                "For zero-variance training features, use std=1 instead of dividing by zero",
                "Malformed patients raise errors instead of being silently skipped",
                "NaNs preserved in parsed.npz; standardized missing locations get zero placeholders",
                "Normalized values saved as float32 (matching CSDI's model input dtype)",
            ],
        }
        (fold_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    print(f"Finished. Prepared files are in: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--reparse", action="store_true")
    args = parser.parse_args()
    if not args.folds or any(f not in range(5) for f in args.folds):
        parser.error("--folds must contain numbers from 0 through 4")
    if not args.ratios or any(not (0 <= r < 1) for r in args.ratios):
        parser.error("--ratios must contain values from 0 through <1")
    if len(set(args.ratios)) != len(args.ratios):
        parser.error("--ratios must be unique")
    prepare(args.raw_dir, args.output_dir, args.folds, args.ratios, args.seed, args.reparse)


if __name__ == "__main__":
    main()
