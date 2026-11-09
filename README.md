
# Imputation Comparison — PhysioNet 2012

A reproducible experiment workspace for comparing missing-data imputation methods. The first dataset is **PhysioNet Challenge 2012, Set A** (hourly clinical time series).

The long-term goal is to compare different imputation methods (such as CSDI, SAITS, MatrixFormer, and other tabular or time-series methods) based on:

- **Imputation accuracy:** RMSE, MAE, and potentially other metrics.
- **Classification performance:** How imputation affects downstream classification.
- **Computational efficiency:** Training time, inference time, and resource requirements.
- **Data representation:** How different time-series-to-tabular conversions affect imputation.

> **Current status:** Dataset parsing, preprocessing, normalization, splitting, and artificial missingness generation are implemented. The preprocessing pipeline has passed synthetic-data tests. Model training, imputation, classification, and shared evaluation are not yet implemented in this repository.

---

## 1. Project Structure

```text
Imputation-Comparison1/
│
├── config/
│   └──                         # Future model and experiment configurations
│
├── data/
│   ├── raw/
│   │   └── physio2012/
│   │       └── set-a/          # Original PhysioNet 2012 patient files
│   │
│   └── processed/
│       └── physio2012/        # Generated data ready for experiments
│
├── datasets/
│   ├── __init__.py
│   └── physio2012.py           # Parses original patient files
│
├── evaluation/
│   └──                         # Future shared evaluation methods
│
├── experiments/
│   └──                         # Future experiment execution scripts
│
├── methods/
│   ├── __init__.py
│   └── CSDI/
│       ├── __init__.py
│       └── csdi_adapter.py     # Converts prepared data to CSDI format
│
├── preprocessing/
│   ├── __init__.py
│   ├── missingness.py          # Artificially hides known measurements
│   ├── normalization.py        # Standardizes measurements
│   └── splitting.py            # Creates train/validation/test splits
│
├── results/
│   └──                         # Future model predictions and metrics
│
├── tests/
│   └── test_pipeline.py        # Tests preprocessing functionality
│
├── .gitignore
├── download.py                 # Downloads original datasets
├── prepare_physio2012.py       # Runs the preprocessing pipeline
└── requirements-preprocessing.txt
```

### Difference between `data/` and `datasets/`

**`data/` contains actual data files.**

- `data/raw/` stores original measurements downloaded from PhysioNet.
- `data/processed/` stores generated NumPy arrays, masks, normalization statistics, and splits.

**`datasets/` contains Python code.**

For example, `datasets/physio2012.py` reads the original patient files and converts them into structured numerical arrays.

This separation allows multiple imputation methods to use identical prepared datasets.

### Why have a separate `methods/` directory?

Each imputation method will eventually have its own implementation.

For example:

```text
methods/
├── CSDI/
│   ├── csdi_adapter.py
│   └── ...
├── SAITS/
│   └── ...
└── MatrixFormer/
    └── ...
```

All methods should use the same dataset splits, artificial missingness masks, and evaluation metrics.

This makes their results more comparable.

---

## 2. Overall Data Preparation Pipeline

The full pipeline is:

```text
Original PhysioNet 2012 dataset
              |
              v
      datasets/physio2012.py
              |
              | Read individual patient files
              | Extract 35 clinical attributes
              | Organize into 48 hourly measurements
              | Preserve naturally missing values
              v
      data/processed/parsed.npz
              |
              v
      preprocessing/splitting.py
              |
              | Separate patients into
              | training, validation, and testing
              v
      preprocessing/normalization.py
              |
              | Calculate training means/stds
              | Standardize all data using them
              v
      preprocessing/missingness.py
              |
              | Artificially hide known values
              | Create reproducible evaluation masks
              v
      prepare_physio2012.py
              |
              | Save processed datasets
              v
      data/processed/physio2012/
              |
              v
       Imputation methods
              |
       +------+------+
       |             |
       v             v
      CSDI        Other methods
       |             |
       +------+------+
              |
              v
       Shared evaluation
```

The path `data/processed/parsed.npz` in this diagram is conceptual; the implementation saves it under `data/processed/physio2012/parsed.npz`.

---

## 3. Dataset Parsing

**File:** `datasets/physio2012.py`

### Purpose

Convert the original PhysioNet 2012 data into numerical arrays usable by different imputation methods.

The dataset contains time-stamped clinical measurements recorded during patients' stays in intensive care.

The original CSDI implementation uses:

- 48 hourly timestamps.
- 35 clinical attributes.
- One matrix per patient.

### Shape

Each patient:

```python
(48, 35)
```

Complete dataset:

```python
(N, 48, 35)
```

Where:

- `N` = number of successfully parsed patients.
- `48` = number of hourly timestamps.
- `35` = number of clinical attributes.

### How timestamps are processed

The original data contains times such as:

```text
00:15
00:45
01:30
02:10
```

The parser extracts only the hour:

```text
00:15 -> 0
00:45 -> 0
01:30 -> 1
02:10 -> 2
```

If multiple measurements of the same attribute occur during one hour, the **last recorded value is retained**.

This behavior is preserved from the original CSDI implementation.

### Natural missingness

Not every clinical attribute is measured every hour.

For example:

```text
Hour    HR     Glucose    Temp
0       80     NaN        37.0
1       85     120        NaN
2       NaN    125        37.2
```

`NaN` indicates a naturally missing measurement.

The parser also creates an `observed_mask`:

```text
Hour    HR    Glucose    Temp
0       1     0          1
1       1     1          0
2       0     1          1
```

Here:

- `1` means the measurement exists.
- `0` means it was naturally missing.

**Difference from original CSDI:**

The parsing module does not immediately replace `NaN` with zero or generate artificial missingness.

These operations are handled separately during preprocessing.

This preserves the original data and makes it reusable.

---

## 4. Patient Splitting

**File:** `preprocessing/splitting.py`

### Purpose

Split the dataset into training, validation, and testing groups.

The split is performed at the **patient level**, not at the individual measurement level.

This is important because measurements belonging to one patient should not be distributed between training and testing.

### Split proportions

The implementation follows the original CSDI approach:

| Split | Percentage | Purpose |
|---|---:|---|
| Training | ~70% | Train imputation models |
| Validation | ~10% | Select model hyperparameters |
| Testing | ~20% | Evaluate final performance |

### Five-fold evaluation

The original CSDI code uses five different test folds.

Each fold selects a different approximately 20% subset of patients for testing.

```text
Fold 0: Patients in group 0 used for testing
Fold 1: Patients in group 1 used for testing
Fold 2: Patients in group 2 used for testing
Fold 3: Patients in group 3 used for testing
Fold 4: Patients in group 4 used for testing
```

Training and validation are generated from the remaining patients.

The same patient splits should be used for every imputation method.

### Reproducibility

A fixed random seed controls the splitting process.

For example:

```python
seed = 1
```

This allows the same splits to be generated again.

The prepared dataset also saves patient IDs and split indices so later experiments do not need to regenerate them.

**Preserved from original CSDI:**

- Patient-level splitting.
- Approximately 70/10/20 proportions.
- Five-fold test strategy.
- Seed-controlled shuffling.

---

## 5. Normalization

**File:** `preprocessing/normalization.py`

### Purpose

Transform clinical measurements onto comparable numerical scales.

Different clinical attributes have very different ranges.

For example:

```text
Heart rate:     80 bpm
Temperature:    37 °C
Glucose:       150 mg/dL
```

Without normalization, these variables can have very different numerical magnitudes.

### Mean

The mean is the average of a feature's observed measurements.

For example:

```text
Heart rate measurements:

60, 70, 80, 90, 100
```

The mean is:

```text
(60 + 70 + 80 + 90 + 100) / 5 = 80
```

### Standard deviation

Standard deviation measures how spread out the measurements are around their mean.

For this example:

```text
Mean = 80
Standard deviation ≈ 14.14
```

### Z-score standardization

CSDI uses:

```text
normalized_value = (original_value - mean) / std
```

For a heart rate of 90:

```text
(90 - 80) / 14.14 ≈ 0.71
```

The normalized value becomes approximately `0.71`.

### Important: Training-only statistics

**Difference from original CSDI:**

The original code calculates normalization statistics before splitting patients into training, validation, and test groups.

This means test measurements influence normalization.

The new pipeline instead:

1. Splits patients first.
2. Calculates means using only observed training measurements.
3. Calculates standard deviations using those same measurements.
4. Applies the training statistics to training, validation, and test data.

This prevents test-set information from leaking into preprocessing.

**This change may affect the final results compared with original CSDI experiments.**

### Missing values during normalization

Naturally missing values are ignored when calculating means and standard deviations.

After normalization, naturally missing positions are represented using zero placeholders.

Their masks still identify them as missing.

A standardized zero placeholder is not considered a real measurement.

### Reversing normalization

Original units can be recovered using:

```python
original_value = normalized_value * std + mean
```

This is important when interpreting results in clinically meaningful units.

---

## 6. Artificial Missingness

**File:** `preprocessing/missingness.py`

### Purpose

Artificially hide some measurements that are actually known.

This makes it possible to evaluate how accurately a model reconstructs missing values.

### Why artificial missingness is needed

Naturally missing values do not have known ground truth.

For example:

```text
Hour 5: Glucose = NaN
```

We cannot determine the actual reconstruction error because the original value was never measured.

Instead, we deliberately hide a known measurement:

```text
Original measurement: Glucose = 120

Value shown to imputer: NaN

Ground truth: 120
```

After imputation, we compare the prediction against 120.

### MCAR missingness

The implementation generates MCAR-style artificial missingness.

MCAR means **Missing Completely At Random**.

For example, with:

```python
missing_ratio = 0.1
```

The pipeline randomly hides 10% of originally observed measurements per patient, subject to integer rounding.

It does not hide additional values at positions that were already naturally missing.

### Different missingness levels

The pipeline supports experiments such as:

- 10% artificial missingness.
- 30% artificial missingness.
- 50% artificial missingness.

These allow comparison of how models perform as data becomes increasingly incomplete.

### Masks

Each prepared dataset uses three related masks:

**`observed_mask`**

Identifies originally measured values.

```text
1 = Originally observed
0 = Naturally missing
```

**`input_mask`**

Identifies values available to the imputation model.

```text
1 = Available to model
0 = Hidden or naturally missing
```

**`eval_mask`**

Identifies artificially hidden values that should be evaluated.

```text
1 = Artificially hidden known value
0 = Not an evaluation target
```

The relationship is:

```python
eval_mask = observed_mask & ~input_mask
```

### Reproducibility

Missingness generation uses a fixed random seed.

Every method should use identical saved evaluation masks.

**Preserved from original CSDI:**

- Random selection among originally observed measurements.
- Artificial hiding performed per patient.
- Seed-controlled masking.

**Difference from original CSDI:**

Artificial missingness generation is independent of the dataset parser and model.

Fixed evaluation masks are saved for reuse by multiple methods.

### Training missingness versus evaluation missingness

The prepared training dataset does not have a fixed artificial evaluation mask.

CSDI generates additional random masks internally during training.

Other imputation methods may implement their own training procedures.

However, fixed validation/test masks are shared between methods.

---

## 7. Preparing PhysioNet 2012

**File:** `prepare_physio2012.py`

This script executes the complete data preparation process.

### Installation

Run commands from the project root.

Install preprocessing dependencies:

```bash
python -m pip install -r requirements-preprocessing.txt
```

PyTorch is required later for CSDI integration, but it is not required for the independent preprocessing pipeline.

### Download dataset

Download PhysioNet Challenge 2012 Set A:

https://physionet.org/content/challenge-2012/1.0.0/

Place the patient files under:

```text
data/raw/physio2012/set-a/
```

For example:

```text
data/raw/physio2012/set-a/132539.txt
data/raw/physio2012/set-a/132540.txt
```

Alternatively, pass `--raw-dir` to specify an existing dataset location.

### Prepare one fold

```bash
python prepare_physio2012.py \
    --folds 0 \
    --ratios 0.1 0.3 0.5 \
    --seed 1
```

This prepares:

- Test fold 0.
- Three artificial missingness levels.
- Reproducible patient splits and evaluation masks.

### Prepare all five folds

```bash
python prepare_physio2012.py \
    --folds 0 1 2 3 4 \
    --ratios 0.1 0.3 0.5 \
    --seed 1
```

### Reparse the original dataset

If the source files change:

```bash
python prepare_physio2012.py \
    --reparse \
    --folds 0 \
    --ratios 0.1 \
    --seed 1
```

### Running tests

```bash
python -m unittest discover -s tests -v
```

These tests use small synthetic patient files and verify the preprocessing behavior without needing the complete PhysioNet dataset.

---

## 8. Processed Dataset Files

After preparation:

```text
data/processed/physio2012/
│
├── parsed.npz
│
└── fold_0/
    ├── normalizer.npz
    ├── splits.npz
    ├── metadata.json
    ├── train.npz
    │
    ├── mcar_0.1/
    │   ├── valid.npz
    │   └── test.npz
    │
    ├── mcar_0.3/
    │   ├── valid.npz
    │   └── test.npz
    │
    └── mcar_0.5/
        ├── valid.npz
        └── test.npz
```

### `parsed.npz`

Contains parsed patient measurements before normalization.

It preserves natural missing values as `NaN`.

It also stores observation masks, patient IDs, and feature names.

### `normalizer.npz`

Contains training-only normalization statistics:

```text
mean
std
```

These are needed to standardize inputs and restore original measurement units.

### `splits.npz`

Contains patient indices and IDs assigned to training, validation, and testing.

### `metadata.json`

Records experiment preparation settings, such as:

```text
Seed
Missingness ratios
Fold
Feature order
Preprocessing choices
```

### `train.npz`, `valid.npz`, `test.npz`

These contain the actual arrays prepared for the models.

Each archive contains:

| Key | Description |
|---|---|
| `values` | Normalized measurements, including hidden ground-truth values |
| `observed_mask` | Originally observed measurements |
| `input_mask` | Measurements available to the imputer |
| `eval_mask` | Artificially hidden evaluation targets |
| `patient_ids` | Patient identifiers |

### Loading a prepared dataset

```python
import numpy as np

path = "data/processed/physio2012/fold_0/mcar_0.1/test.npz"

with np.load(path) as data:
    values = data["values"]
    observed_mask = data["observed_mask"]
    input_mask = data["input_mask"]
    eval_mask = data["eval_mask"]
    patient_ids = data["patient_ids"]
```

### Important: Prevent evaluation leakage

The `values` array retains the actual values at artificially hidden positions.

They are necessary for calculating evaluation metrics.

However, methods that do not explicitly support masks must not receive those hidden values as visible inputs.

Create the actual input using:

```python
model_input = np.where(input_mask, values, np.nan)
```

This ensures the imputer only sees permitted measurements.

---

## 9. CSDI Integration

**File:** `methods/CSDI/csdi_adapter.py`

The CSDI adapter converts prepared NumPy arrays into PyTorch datasets and data loaders compatible with the original CSDI model.

### Why an adapter is needed

Different methods may expect different data formats.

For example:

```text
CSDI:    Time-series tensors + masks + timestamps
SAITS:   Time-series tensors + masks
Tabular:  Two-dimensional tables with missing values
```

The shared preprocessing pipeline should not depend on these implementation details.

Instead, each method can have its own adapter.

### Using the CSDI adapter

```python
from methods.CSDI.csdi_adapter import get_prepared_dataloaders

train_loader, valid_loader, test_loader = (
    get_prepared_dataloaders(
        output_dir="data/processed/physio2012",
        fold=0,
        ratio=0.1,
        batch_size=16,
        seed=1,
    )
)
```

### What the adapter returns

Each batch contains:

```python
{
    "observed_data": ...,
    "observed_mask": ...,
    "gt_mask": ...,
    "timepoints": ...
}
```

These are the names expected by the original CSDI implementation.

**Important:** CSDI calls the conditioning mask `gt_mask`. In the shared preprocessing pipeline, the equivalent is called `input_mask`.

### What is not implemented yet?

The adapter only loads and converts the data.

The following components still need to be integrated:

- CSDI model architecture.
- Diffusion model.
- Training logic.
- Imputation logic.
- Evaluation logic.
- Model configuration.

The original implementation is available at:

https://github.com/ermongroup/CSDI

---

## 10. Methodological Differences From Original CSDI

Some changes are intended to improve experimental reproducibility and prevent data leakage.

These changes may affect the results compared with the original implementation.

| Change | Potential effect |
|---|---|
| Training-only normalization | Removes leakage; may change model inputs and scores |
| Separation of data parsing and artificial missingness | Improves reuse and comparison |
| Preservation of natural `NaN` values until normalization | Clarifies missingness representation |
| Explicitly saved evaluation masks | Enables identical evaluation targets across methods |
| Strict patient filename validation | May change accepted patient records if files are malformed |
| Handling zero standard deviation | Prevents division by zero |
| Explicit random generators | Improves reproducibility |
| Explicit float32 conversion | May introduce small numerical differences |

### Behaviors retained from original CSDI

- Same 35 physiological attributes.
- Same attribute order.
- Same 48-hour observation windows.
- Last recorded measurement retained for each hour.
- Patient-level five-fold test strategy.
- Approximately 70% training, 10% validation, and 20% testing.
- Artificial missingness sampled from originally observed values.
- Per-patient artificial missingness generation.

The new pipeline is intended to provide a consistent basis for future comparisons, not necessarily reproduce every numerical result from the original CSDI paper.

---

## 11. Planned Evaluation

The `evaluation/` directory will eventually contain model-independent evaluation functions.

### Imputation metrics

**MAE — Mean Absolute Error**

Average absolute difference between predicted and actual values.

```text
MAE = mean(abs(predicted - actual))
```

Lower is better.

**RMSE — Root Mean Squared Error**

Square root of the average squared prediction error.

```text
RMSE = sqrt(mean((predicted - actual)^2))
```

Lower is better.

These metrics should be calculated only on artificially hidden, originally observed positions identified by `eval_mask`.

### Classification metrics

Potential metrics include:

- ROC-AUC.
- PR-AUC.
- F1 score.
- Accuracy.

The objective is to investigate whether better imputation accuracy also leads to better downstream classification.

PhysioNet outcome labels will need to be loaded separately and aligned with patient IDs.

### Efficiency

Potential metrics include:

- Model training time.
- Imputation inference time.
- Peak memory usage.
- GPU memory usage.

All methods should be compared under documented hardware and experimental conditions.

---

## 12. Next Development Steps

1. **Implement shared evaluation metrics**

   Create `evaluation/imputation_metrics.py` for MAE and RMSE.

2. **Integrate the original CSDI model**

   Move or adapt the required CSDI training and imputation components into `methods/CSDI/`.

3. **Verify CSDI on the prepared dataset**

   Confirm the correct masks and patient splits are used and that all evaluation batches are processed.

4. **Add simple imputation baselines**

   For example:
   - Mean imputation.
   - Linear interpolation.
   - Forward filling.

5. **Add additional imputation methods**

   Such as SAITS and selected tabular models.

6. **Investigate time-series representations**

   Compare direct time-series input with approaches such as:
   - Per-timestamp rows.
   - Flattened patient windows.
   - Sliding windows.

7. **Implement downstream classification**

   Compare how different imputations affect classification performance.

8. **Record reproducible experiment results**

   Save predictions, metrics, random seeds, configuration, and execution times.

---

## 13. Troubleshooting

### Dataset directory not found

Verify that patient files are in:

```text
data/raw/physio2012/set-a/
```

Or specify another path:

```bash
python prepare_physio2012.py \
    --raw-dir data/physio/set-a \
    --folds 0 \
    --ratios 0.1 \
    --seed 1
```

### Python cannot import preprocessing modules

Run commands from the project root rather than from inside individual directories.

### Changes to raw files are not reflected

Use:

```bash
python prepare_physio2012.py --reparse --folds 0 --ratios 0.1
```

### Dataset files are not visible on GitHub

The `.gitignore` may intentionally exclude generated data.

This is generally desirable because raw and processed datasets can be large and can be regenerated.

### Missing CSDI model

The preprocessing pipeline can run independently.

The CSDI adapter requires PyTorch, and actual CSDI training requires the original model implementation.

---

## References

**PhysioNet Challenge 2012**

https://physionet.org/content/challenge-2012/1.0.0/

**CSDI — Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation**

Tashiro et al., NeurIPS 2021.

https://github.com/ermongroup/CSDI

**Project repository**

https://github.com/Matas5/imputation_comparison1
