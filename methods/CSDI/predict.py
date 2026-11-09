
"""Convert original CSDI probabilistic samples into shared model predictions."""
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from evaluation.imputation_metrics import compute_metrics


@torch.no_grad()
def predict_test(model, test_loader, test_npz, normalizer_npz, output_dir, n_samples=10,
                 save_samples=False):
    """Predict ALL test batches and save arrays in (N, hours, features) order.

    ORIGINAL CSDI: model.evaluate() returns n_samples reverse-diffusion
    realizations, and point scores use torch.median(dim=1), which for an even
    sample count takes the lower of the middle two values (not NumPy's mean).

    CHANGE: No early break after batch one. Shared metrics instead of the
    original model-specific utils.evaluate(). This can change reported scores.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    model.eval()
    predictions_list, completed_list, samples_list = [], [], []
    for batch in tqdm(test_loader, desc="CSDI inference"):
        # CSDI itself uses gt_mask (= input_mask) to condition on visible values;
        # the held-out target values remain available solely for scoring.
        samples, target, target_mask, _, _ = model.evaluate(batch, n_samples)
        samples = samples.permute(0, 1, 3, 2)           # B, samples, hours, features
        point = samples.median(dim=1).values            # B, hours, features
        visible = batch["gt_mask"].to(point.device).bool()
        original_data = batch["observed_data"].to(point.device)
        complete = torch.where(visible, original_data, point)
        predictions_list.append(point.cpu().numpy().astype(np.float32))
        completed_list.append(complete.cpu().numpy().astype(np.float32))
        if save_samples:
            # MAY BE HUGE: N * n_samples * 48 * 35 float32 entries.
            samples_list.append(samples.cpu().numpy().astype(np.float32))

    if not predictions_list:
        raise RuntimeError("No test patients found")
    predictions = np.concatenate(predictions_list)
    completed = np.concatenate(completed_list)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(test_npz, allow_pickle=False) as saved:
        targets = saved["values"]
        eval_mask = saved["eval_mask"].astype(bool)
        observed_mask = saved["observed_mask"].astype(bool)
        input_mask = saved["input_mask"].astype(bool)
        ids = saved["patient_ids"]
    if len(predictions) != len(targets):
        raise RuntimeError("Test DataLoader must have shuffle=False and drop_last=False")
    if np.any(input_mask & ~observed_mask):
        raise ValueError("An input mask contains locations not originally observed")

    with np.load(normalizer_npz, allow_pickle=False) as norm:
        mean, std = norm["mean"], norm["std"]
    metrics = compute_metrics(predictions, targets, eval_mask, mean=mean, std=std)
    # Save truth and masks for transparent offline evaluation by all models.
    np.savez_compressed(
        output_dir / "predictions.npz",
        predictions=predictions,
        completed=completed,
        targets=targets,
        observed_mask=observed_mask,
        input_mask=input_mask,
        eval_mask=eval_mask,
        patient_ids=ids,
    )
    if save_samples:
        np.savez_compressed(
            output_dir / "samples.npz",
            samples=np.concatenate(samples_list),
            patient_ids=ids,
        )
    return metrics
