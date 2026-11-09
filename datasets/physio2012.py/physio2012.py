"""OPTIONAL adapter for existing CSDI, requiring PyTorch.

This module does NOT prepare or alter the dataset. It maps saved universal
arrays to the exact dict expected by main_model.CSDI_Physio.process_data().
"""

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader

from datasets.physio2012 import HOURS


class PreparedPhysioDataset(Dataset):
    def __init__(self, npz_path: str | Path):
        with np.load(npz_path, allow_pickle=False) as data:
            self.values = data["values"].astype(np.float32)
            self.observed_mask = data["observed_mask"].astype(np.float32)
            self.input_mask = data["input_mask"].astype(np.float32)
        if self.values.shape != self.observed_mask.shape or self.values.shape != self.input_mask.shape:
            raise ValueError("Saved data/masks have inconsistent shapes")

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, index):
        return {
            # CSDI NEEDS ground-truth values at artificially hidden positions
            # for calculating validation losses and test evaluation scores.
            # Its conditioning in main_model.py uses gt_mask, NOT these targets.
            "observed_data": self.values[index],
            "observed_mask": self.observed_mask[index],
            "gt_mask": self.input_mask[index],  # CSDI's name for visible-input mask
            "timepoints": np.arange(HOURS, dtype=np.float32),
        }


def get_prepared_dataloaders(
    output_dir: str | Path = "data/processed/physio2012",
    fold: int = 0,
    ratio: float = 0.1,
    batch_size: int = 16,
    seed: int = 1,
):
    """Drop-in return shape for CSDI exe_physio.py's old get_dataloader()."""
    folder = Path(output_dir) / f"fold_{fold}"
    mask_folder = folder / f"mcar_{ratio:g}"
    generator = __import__("torch").Generator().manual_seed(seed)
    return (
        DataLoader(PreparedPhysioDataset(folder / "train.npz"),
                   batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(PreparedPhysioDataset(mask_folder / "valid.npz"),
                   batch_size=batch_size, shuffle=False),
        DataLoader(PreparedPhysioDataset(mask_folder / "test.npz"),
                   batch_size=batch_size, shuffle=False),
    )
