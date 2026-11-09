"""Small tests using fabricated patient records; never requires downloading PhysioNet."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.physio2012 import ATTRIBUTES, load_physio2012
from preprocessing.splitting import split_patient_indices
from preprocessing.normalization import fit_normalizer, transform, inverse_transform
from preprocessing.missingness import generate_mcar_input_mask, get_eval_mask
from prepare_physio2012 import prepare


class TestPhysioPipeline(unittest.TestCase):
    def test_parse_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw" / "set-a"
            raw.mkdir(parents=True)
            for patient in range(10):
                entries = [
                    {"Time": "00:00", "Parameter": attr, "Value": 10 + patient + i}
                    for i, attr in enumerate(ATTRIBUTES)
                ]
                # Repeated HR at same hour; expect last record to win.
                entries.append({"Time": "00:55", "Parameter": "HR", "Value": 123 + patient})
                # A single observed value at hour 1, most are naturally missing.
                entries.append({"Time": "01:00", "Parameter": "HR", "Value": 85 + patient})
                pd.DataFrame(entries).to_csv(raw / f"{100000+patient}.txt", index=False)

            vals, obs, ids = load_physio2012(raw)
            self.assertEqual(vals.shape, (10, 48, 35))
            self.assertEqual(ids[0], "100000")
            self.assertEqual(vals[0, 0, ATTRIBUTES.index("HR")], 123)
            self.assertFalse(obs[0, 3, ATTRIBUTES.index("HR")])
            self.assertEqual(obs[0].sum(), 36)

            out = root / "processed"
            prepare(raw, out, folds=[0], ratios=[0.1, 0.5], seed=1)
            prepare(raw, out, folds=[0], ratios=[0.1, 0.5], seed=1)
            with np.load(out / "fold_0" / "normalizer.npz") as norm:
                mean, std = norm["mean"], norm["std"]
                with np.load(out / "fold_0" / "splits.npz") as split:
                    indices = split["train_indices"]
                    self.assertEqual(len(split["test_indices"]), 2)
                    self.assertEqual(len(indices), 7)
                    self.assertEqual(len(split["valid_indices"]), 1)
                    # Means determined only by train patient numbers.
                    expected = vals[indices, 0, ATTRIBUTES.index("DiasABP")].mean()
                    self.assertAlmostEqual(mean[0], expected)
                    self.assertNotAlmostEqual(mean[0], vals[:, 0, 0].mean())
                with np.load(out / "fold_0" / "mcar_0.1" / "test.npz") as test:
                    self.assertTrue(np.all(~test["eval_mask"] | test["observed_mask"]))
                    self.assertTrue(np.all(~test["eval_mask"] | ~test["input_mask"]))
                    self.assertEqual(int(test["eval_mask"].sum()), 6)  # 3/patient
                    # Missing placeholders are zero, non-hidden truth remains stored.
                    self.assertTrue(np.all(test["values"][~test["observed_mask"]] == 0))
                    self.assertTrue(np.any(test["values"][test["eval_mask"]] != 0))
                    # Reversible, with the SAME train-derived statistics.
                    recovered = inverse_transform(test["values"], mean, std)
                    self.assertTrue(np.isfinite(recovered).all())

    def test_mask_per_patient_and_repeatable(self):
        mask = np.ones((3, 4, 5), dtype=bool)
        mask[0, :, 0] = False
        a = generate_mcar_input_mask(mask, 0.25, 123)
        b = generate_mcar_input_mask(mask, 0.25, 123)
        np.testing.assert_array_equal(a, b)
        target = get_eval_mask(mask, a)
        np.testing.assert_array_equal(target.sum(axis=(1, 2)), [4, 5, 5])
        self.assertFalse(np.any(a & ~mask))

    def test_split_matches_legacy(self):
        n, fold, seed = 100, 2, 1
        train, valid, test = split_patient_indices(n, fold, seed)
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        expected_test = idx[40:60]
        rest = np.delete(idx, np.arange(40, 60))
        rng = np.random.RandomState(seed)
        rng.shuffle(rest)
        np.testing.assert_array_equal(train, rest[:70])
        np.testing.assert_array_equal(valid, rest[70:])
        np.testing.assert_array_equal(test, expected_test)

    def test_train_stats_ignore_nan_and_missing(self):
        x = np.array([[[10., np.nan]], [[20., 100.]], [[1000., 999.]]])
        mask = ~np.isnan(x)
        mean, std, count = fit_normalizer(x[:2], mask[:2])
        np.testing.assert_allclose(mean, [15., 100.])
        np.testing.assert_allclose(std, [5., 1.])  # constant observed feature
        np.testing.assert_array_equal(count, [2, 1])
        out = transform(x, mask, mean, std)
        self.assertEqual(out[0, 0, 1], 0)
        self.assertEqual(out[2, 0, 0], 197.)  # test not centered on its own mean


if __name__ == "__main__":
    unittest.main()
