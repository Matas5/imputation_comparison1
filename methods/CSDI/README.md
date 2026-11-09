# CSDI on the shared PhysioNet 2012 benchmark

This module uses the existing `methods/CSDI/csdi_adapter.py` and the
model-independent `data/processed/physio2012/` files. It **does not download,
reparse, normalize, or regenerate test masks**.

## Files

- `main_model.py` — CSDI's diffusion equations, random training masks, training loss, and iterative sample generation. PhysioNet-specific `process_data()` rotates `(B,48,35)` to `(B,35,48)`.
- `diff_models.py` — original attention network that predicts added diffusion noise.
- `csdi_adapter.py` — your existing dataset adapter; `gt_mask` is the *visible input mask*, NOT the evaluation targets.
- `training.py` — Adam, MultiStepLR, validation, `final.pth` and `best.pth` checkpoints.
- `predict.py` — all test batches, multiple diffusion samples, torch-median point prediction, completed data, `.npz` and metrics.
- `run.py` — command-line entry point; configuration and output paths.
- `config/base.yaml` — original 200-epoch/50-step CSDI PhysioNet hyperparameters.
- `LICENSE-CSDI` — MIT license retained from the original implementation.

## Install (from repository root)

```sh
source .venv/bin/activate
python -m pip install -r requirements-preprocessing.txt
python -m pip install -r methods/CSDI/requirements.txt
```

On Apple Silicon, `--device auto` chooses PyTorch MPS if available, otherwise CPU. CUDA is chosen on compatible NVIDIA systems. Use `--device cpu` to force CPU.

## Prepare one fold at 10% artificial missingness

```sh
python prepare_physio2012.py --folds 0 --ratios 0.1 --seed 1
```

## Small integration check (NOT comparable research results)

```sh
python -m methods.CSDI.run --mode train --device auto --fold 0 --ratio 0.1 \
  --epochs 1 --num-steps 2 --max-train-batches 1 --validation-every 1 --n-samples 1
```

**This still runs inference over the entire test fold**. For a faster check of only model shapes use `python -m unittest discover -s tests -v` and the accompanying CSDI test.

## Standard experiment

```sh
python -m methods.CSDI.run --mode train --device auto --fold 0 --ratio 0.1 \
  --n-samples 10 --checkpoint-selection final
```

The source CSDI defaults (200 epochs, 50 diffusion steps, batch size 16) come from `base.yaml`. Sampling 10 trajectories is much cheaper than sampling 100, but may give a different point estimate. Run across folds and ratios using repeated commands.

`--checkpoint-selection final` matches the original fork's save-at-end behavior.
`--checkpoint-selection best` instead selects the lowest validation loss checkpoint when available and **may change the measured results**. Both checkpoints are saved.

## Evaluate an existing checkpoint, without retraining

```sh
python -m methods.CSDI.run --mode predict --device auto --fold 0 --ratio 0.1 \
  --checkpoint results/CSDI/physio2012/fold_0/mcar_0.1/RUN_DIRECTORY/final.pth \
  --n-samples 10
```

The exact path printed by training should replace `RUN_DIRECTORY`. The checkpoint's `config.json` is automatically loaded from its directory.

## Outputs

A timestamped directory under `results/CSDI/physio2012/fold_0/mcar_0.1/` contains:

- `final.pth` — final epoch (for trained runs only)
- `best.pth` — lowest tracked validation loss (when validation occurred)
- `config.json` — exact model settings (also stored for evaluation-only runs)
- `predictions.npz` — `predictions`, `completed`, `targets`, masks and patient IDs; all in `(N,48,35)` order
- `metrics.json` — normalized MAE/RMSE and mixed-original-unit MAE/RMSE, evaluated ONLY on `eval_mask`
- `run_info.json` — command-line settings
- `samples.npz` — optional (`--save-samples`), possibly very large `(N,S,48,35)`

`predictions` are CSDI's point estimates at *every* location, including naturally missing values. `completed` preserves the values available to the imputer and fills all other positions. Only artificially hidden observed locations are scored.

## Differences to document in the coursework

1. **Preprocessing** is different from the original CSDI script: training-only normalization prevents data leakage. This changes inputs and may change scores.
2. **Artificial test masks** are generated once by shared preprocessing and reused across methods, instead of being regenerated inside CSDI's dataset class. Training masks are STILL random inside CSDI.
3. **Point estimate** uses `torch.median(dim=1)` like the original; for even `n_samples`, this is the lower-middle sample.
4. **All test batches** are used. Your original CSDI fork had a debug limit stopping after one batch.
5. **Saved checkpoints**: your original fork saves final weights, not the best-validation weights. We save both and make selection explicit.
6. **Model architecture, training loss and reverse diffusion** are preserved from original CSDI. `diff_models.py` lazy-imports optional linear attention, which is not used with the default config.
7. **Reported metrics** are from the model-independent evaluator, not the source CSDI CRPS code. CRPS and classification are future work.
8. **RNG seeds** are set for NumPy, Python and Torch, but exact equality across machines/accelerators is not guaranteed.
9. **New, smaller debug overrides** like `--num-steps 2` and `--epochs 1` are ONLY for integration tests, not scientific comparisons.

## Provenance

Source fork: https://github.com/Matas5/CSDI (MIT-licensed, 2021 Yusuke Tashiro).
Model and diffusion code derives from `main_model.py` and `diff_models.py` in that fork. The original code contains additional PM2.5/forecasting subclasses not needed by this PhysioNet-only integration. Original paper: Tashiro et al., *Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation*, NeurIPS 2021.
