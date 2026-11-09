
"""Load PhysioNet 2012 Set A, without scaling or artificial masking.

Based on the parsing rules in Matas5/CSDI's dataset_physio.py.
Run scripts from the project root or use paths relative to this file's parent.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

# SAME AS ORIGINAL: keep the precise list AND order of the original 35 variables.
# Changing the feature list/order changes the learning problem and CSDI embeddings.
ATTRIBUTES = [
    "DiasABP", "HR", "Na", "Lactate", "NIDiasABP", "PaO2", "WBC", "pH",
    "Albumin", "ALT", "Glucose", "SaO2", "Temp", "AST", "Bilirubin",
    "HCO3", "BUN", "RespRate", "Mg", "HCT", "SysABP", "FiO2", "K",
    "GCS", "Cholesterol", "NISysABP", "TroponinT", "MAP", "TroponinI",
    "PaCO2", "Platelets", "Urine", "NIMAP", "Creatinine", "ALP",
]
HOURS = 48


def extract_hour(time_string: str) -> int:
    """'03:45' -> 3. SAME AS ORIGINAL: minute information is discarded."""
    hour, _ = map(int, time_string.split(":"))
    return hour


def parse_data(hour_rows: pd.DataFrame) -> list[float]:
    """Return 35 measurements for an hour, with NaN for unobserved features."""
    # SAME AS ORIGINAL: turning records into a dictionary keeps the LAST value
    # encountered for a variable in a given hour. Changing to an hourly mean
    # would change the dataset and therefore experimental results.
    by_parameter = hour_rows.set_index("Parameter")["Value"].to_dict()
    return [by_parameter.get(name, np.nan) for name in ATTRIBUTES]


def parse_patient(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return values (48,35) with NaNs and natural-observation mask (48,35)."""
    data = pd.read_csv(path)
    expected = {"Time", "Parameter", "Value"}
    if not expected.issubset(data.columns):
        raise ValueError(f"{path}: expected columns {sorted(expected)}")
    data["Time"] = data["Time"].map(extract_hour)

    values = np.asarray(
        [parse_data(data[data["Time"] == hour]) for hour in range(HOURS)],
        dtype=np.float64,
    )
    # CHANGE FROM ORIGINAL: keep NaN values here, instead of replacing with 0.
    # This makes natural missingness unambiguous for other imputation methods.
    # Zeros will only be inserted as placeholders AFTER normalization.
    observed_mask = ~np.isnan(values)
    if not np.isfinite(values[observed_mask]).all():
        raise ValueError(f"{path}: non-finite observed value")
    return values, observed_mask


def load_physio2012(raw_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read all Set A patient files sorted by ID; fail if data are malformed."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

    # CHANGE: match only real six-digit patient filenames, not arbitrary names
    # containing six consecutive digits as the original regex did.
    files = sorted(
        (p for p in raw_dir.iterdir() if p.is_file() and re.fullmatch(r"\d{6}\.txt", p.name)),
        key=lambda p: p.stem,
    )
    if not files:
        raise FileNotFoundError(f"No patient .txt files in {raw_dir}")

    values, masks, ids = [], [], []
    for path in files:
        # CHANGE: fail loudly instead of 'print and continue' on bad patients.
        # A different set of accepted patients changes all folds and metrics.
        try:
            patient_values, patient_mask = parse_patient(path)
        except Exception as exc:
            raise ValueError(f"Could not parse patient {path.stem}: {exc}") from exc
        values.append(patient_values)
        masks.append(patient_mask)
        ids.append(path.stem)

    return (
        np.stack(values),
        np.stack(masks),
        np.asarray(ids, dtype="U6"),
    )

"""Load PhysioNet 2012 Set A, without scaling or artificial masking.

Based on the parsing rules in Matas5/CSDI's dataset_physio.py.
Run scripts from the project root or use paths relative to this file's parent.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

# SAME AS ORIGINAL: keep the precise list AND order of the original 35 variables.
# Changing the feature list/order changes the learning problem and CSDI embeddings.
ATTRIBUTES = [
    "DiasABP", "HR", "Na", "Lactate", "NIDiasABP", "PaO2", "WBC", "pH",
    "Albumin", "ALT", "Glucose", "SaO2", "Temp", "AST", "Bilirubin",
    "HCO3", "BUN", "RespRate", "Mg", "HCT", "SysABP", "FiO2", "K",
    "GCS", "Cholesterol", "NISysABP", "TroponinT", "MAP", "TroponinI",
    "PaCO2", "Platelets", "Urine", "NIMAP", "Creatinine", "ALP",
]
HOURS = 48


def extract_hour(time_string: str) -> int:
    """'03:45' -> 3. SAME AS ORIGINAL: minute information is discarded."""
    hour, _ = map(int, time_string.split(":"))
    return hour


def parse_data(hour_rows: pd.DataFrame) -> list[float]:
    """Return 35 measurements for an hour, with NaN for unobserved features."""
    # SAME AS ORIGINAL: turning records into a dictionary keeps the LAST value
    # encountered for a variable in a given hour. Changing to an hourly mean
    # would change the dataset and therefore experimental results.
    by_parameter = hour_rows.set_index("Parameter")["Value"].to_dict()
    return [by_parameter.get(name, np.nan) for name in ATTRIBUTES]


def parse_patient(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return values (48,35) with NaNs and natural-observation mask (48,35)."""
    data = pd.read_csv(path)
    expected = {"Time", "Parameter", "Value"}
    if not expected.issubset(data.columns):
        raise ValueError(f"{path}: expected columns {sorted(expected)}")
    data["Time"] = data["Time"].map(extract_hour)

    values = np.asarray(
        [parse_data(data[data["Time"] == hour]) for hour in range(HOURS)],
        dtype=np.float64,
    )
    # CHANGE FROM ORIGINAL: keep NaN values here, instead of replacing with 0.
    # This makes natural missingness unambiguous for other imputation methods.
    # Zeros will only be inserted as placeholders AFTER normalization.
    observed_mask = ~np.isnan(values)
    if not np.isfinite(values[observed_mask]).all():
        raise ValueError(f"{path}: non-finite observed value")
    return values, observed_mask


def load_physio2012(raw_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read all Set A patient files sorted by ID; fail if data are malformed."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")

    # CHANGE: match only real six-digit patient filenames, not arbitrary names
    # containing six consecutive digits as the original regex did.
    files = sorted(
        (p for p in raw_dir.iterdir() if p.is_file() and re.fullmatch(r"\d{6}\.txt", p.name)),
        key=lambda p: p.stem,
    )
    if not files:
        raise FileNotFoundError(f"No patient .txt files in {raw_dir}")

    values, masks, ids = [], [], []
    for path in files:
        # CHANGE: fail loudly instead of 'print and continue' on bad patients.
        # A different set of accepted patients changes all folds and metrics.
        try:
            patient_values, patient_mask = parse_patient(path)
        except Exception as exc:
            raise ValueError(f"Could not parse patient {path.stem}: {exc}") from exc
        values.append(patient_values)
        masks.append(patient_mask)
        ids.append(path.stem)

    return (
        np.stack(values),
        np.stack(masks),
        np.asarray(ids, dtype="U6"),
    )
