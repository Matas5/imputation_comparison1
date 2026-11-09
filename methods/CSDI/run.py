
"""Entry point for the CSDI method only, using shared preprocessed NPZ files.

Run from the repository root:
  python -m methods.CSDI.run --mode train --device auto --fold 0 --ratio 0.1
  python -m methods.CSDI.run --mode predict --checkpoint results/.../final.pth

No downloading, parsing, artificial mask generation, or normalization occurs here.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from .csdi_adapter import get_prepared_dataloaders
from .main_model import CSDI_Physio
from .training import train
from .predict import predict_test

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "physio2012"
DEFAULT_RESULTS = ROOT / "results" / "CSDI" / "physio2012"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "base.yaml"


def choose_device(requested):
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    # CHANGE: original exe_physio.py parsed a seed but did not fully seed
    # PyTorch and NumPy model randomness before training. Seeding improves
    # repeatability but does not guarantee bitwise deterministic GPU/MPS runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run CSDI on prepared PhysioNet 2012")
    p.add_argument("--mode", choices=["train", "predict"], default="train")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda, cuda:0")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--epochs", type=int, help="DEBUG override; normal setting is 200")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-steps", type=int, help="DEBUG override; normal setting is 50")
    p.add_argument("--max-train-batches", type=int, help="DEBUG cap per epoch")
    p.add_argument("--validation-every", type=int, default=20)
    p.add_argument("--checkpoint-selection", choices=["final", "best"], default="final")
    p.add_argument("--checkpoint", type=Path, help="Required for --mode predict")
    p.add_argument("--save-samples", action="store_true", help="Can generate very large files")
    p.add_argument("--run-dir", type=Path, help="Optional exact output directory")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fold not in range(5):
        raise ValueError("fold must be 0..4")
    if not (0 <= args.ratio < 1):
        raise ValueError("ratio must be in [0,1)")
    if args.n_samples < 1 or args.validation_every < 1:
        raise ValueError("n-samples and validation-every must be positive")
    if args.mode == "predict" and not args.checkpoint:
        raise ValueError("--mode predict requires --checkpoint path")
    if args.mode == "predict" and any(x is not None for x in (args.epochs, args.num_steps)):
        raise ValueError("Do not override model architecture when loading checkpoint")

    seed_everything(args.seed)
    device = choose_device(args.device)
    print("Using device:", device)

    if args.mode == "predict":
        # Read the SAME architecture settings that created the checkpoint.
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config_path = checkpoint.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Expected saved configuration beside checkpoint: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        original_run_info = checkpoint.parent / "run_info.json"
        if original_run_info.is_file():
            previous = json.loads(original_run_info.read_text(encoding="utf-8"))
            if previous.get("fold") != args.fold or float(previous.get("ratio", -1)) != args.ratio:
                raise ValueError("Checkpoint was trained using another fold or missingness ratio. "
                                 "Pass the corresponding --fold and --ratio explicitly.")
    else:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.epochs is not None:
            config["train"]["epochs"] = args.epochs
        if args.num_steps is not None:
            config["diffusion"]["num_steps"] = args.num_steps
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size

    train_loader, valid_loader, test_loader = get_prepared_dataloaders(
        output_dir=args.data_dir, fold=args.fold, ratio=args.ratio,
        batch_size=int(config["train"]["batch_size"]), seed=args.seed,
    )
    model = CSDI_Physio(config, device).to(device)
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        # Fresh output directory for every run, no overwriting previous scores.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = args.results_dir / f"fold_{args.fold}" / f"mcar_{args.ratio:g}" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    # The CSDI constructor adds `side_dim` to diffusion config; this is derived
    # and does not change the architecture relative to original CSDI.
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.mode == "train":
        print("Training CSDI...")
        checkpoint_paths = train(
            model, config["train"], train_loader, valid_loader, run_dir,
            valid_epoch_interval=args.validation_every,
            max_train_batches=args.max_train_batches,
        )
        chosen = checkpoint_paths[args.checkpoint_selection]
        if chosen is None:
            # Common in quick experiments with epochs < validation interval.
            print("No best-validation checkpoint exists; using final.pth")
            chosen = checkpoint_paths["final"]
        checkpoint = chosen
    else:
        print("Loading checkpoint", checkpoint)

    # Explicitly reload selected checkpoint; do not silently use in-memory weights.
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    data_fold = args.data_dir / f"fold_{args.fold}"
    metadata_file = data_fold / "metadata.json"
    prepared_seed = None
    if metadata_file.is_file():
        prepared_seed = json.loads(metadata_file.read_text(encoding="utf-8")).get("seed")
        if prepared_seed != args.seed:
            print(f"NOTE: saved dataset seed={prepared_seed}; model/sampling seed={args.seed}. "
                  "These are separate sources of randomness.")
    metrics = predict_test(
        model, test_loader,
        test_npz=data_fold / f"mcar_{args.ratio:g}" / "test.npz",
        normalizer_npz=data_fold / "normalizer.npz",
        output_dir=run_dir, n_samples=args.n_samples, save_samples=args.save_samples,
    )
    metrics.update({
        "method": "CSDI", "dataset": "physio2012",
        "fold": args.fold, "missing_ratio": args.ratio,
        "n_samples": args.n_samples, "seed": args.seed,
        "prepared_dataset_seed": prepared_seed,
        "checkpoint": str(checkpoint), "device": str(device),
        "checkpoint_selection": args.checkpoint_selection if args.mode == "train" else "external",
        "NOTE": "Metrics use saved eval_mask. Normalized metrics aggregate features on standardized scales.",
    })
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "run_info.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved results in", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
