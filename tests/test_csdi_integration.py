"""Fast smoke checks: real model shapes, conditioning, and mask-restricted scoring.

These are NOT a training-quality test or reproduction of the research paper.
"""
import unittest
import numpy as np
import torch

from evaluation.imputation_metrics import compute_metrics
from methods.CSDI.main_model import CSDI_Physio


class CSDIIntegrationTests(unittest.TestCase):
    def test_metrics_only_score_hidden_known_positions(self):
        original = np.array([[[1., 5.], [2., 10.]]], dtype=np.float32)
        predicted = np.array([[[100., 3.], [2., -100.]]], dtype=np.float32)
        eval_mask = np.array([[[False, True], [False, False]]])
        metrics = compute_metrics(predicted, original, eval_mask, mean=np.zeros(2), std=np.array([1., 2.]))
        self.assertEqual(metrics["n_eval_points"], 1)
        self.assertAlmostEqual(metrics["mae_normalized"], 2)
        self.assertAlmostEqual(metrics["rmse_normalized"], 2)
        self.assertAlmostEqual(metrics["mae_original_mixed_units"], 4)

    def test_physio_model_loss_and_sample_shapes(self):
        torch.manual_seed(7)
        np.random.seed(7)
        config = {
            "model": {"timeemb": 8, "featureemb": 4, "is_unconditional": False,
                      "target_strategy": "random"},
            "diffusion": {"layers": 1, "channels": 8, "nheads": 2,
                          "diffusion_embedding_dim": 16, "beta_start": 0.0001,
                          "beta_end": 0.5, "num_steps": 2, "schedule": "quad",
                          "is_linear": False},
        }
        model = CSDI_Physio(config, device=torch.device("cpu"))
        # (B, time, features); no naturally missing entries in this toy batch.
        data = torch.randn(1, 48, 35)
        observed = torch.ones_like(data)
        visible = observed.clone()
        visible[:, 2:5, 0] = 0
        batch = {
            "observed_data": data,
            "observed_mask": observed,
            "gt_mask": visible,
            "timepoints": torch.arange(48).float().unsqueeze(0),
        }
        loss = model(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(next(model.parameters()).grad)
        with torch.no_grad():
            samples, targets, eval_mask, _, _ = model.evaluate(batch, n_samples=1)
        self.assertEqual(tuple(samples.shape), (1, 1, 35, 48))
        self.assertEqual(tuple(targets.shape), (1, 35, 48))
        self.assertEqual(int(eval_mask.sum().item()), 3)


if __name__ == "__main__":
    unittest.main()
