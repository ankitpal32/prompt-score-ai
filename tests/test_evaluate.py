"""
Tests for evaluation metrics, reporting, and score formatting functions.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_scorer.evaluate import (
    evaluate_predictions,
    format_score,
    save_metrics,
    save_predictions,
)


class TestEvaluate(unittest.TestCase):
    def test_evaluate_predictions(self):
        y_true = [5.0, 7.0, 9.0]
        y_pred = [5.5, 6.5, 9.0]
        baseline_pred = [7.0, 7.0, 7.0]

        metrics = evaluate_predictions(y_true, y_pred, baseline_pred=baseline_pred)
        self.assertEqual(metrics["test_rows"], 3)
        self.assertAlmostEqual(metrics["mae"], (0.5 + 0.5 + 0.0) / 3, places=4)
        self.assertIn("r2", metrics)
        self.assertAlmostEqual(metrics["baseline_mae"], (2.0 + 0.0 + 2.0) / 3, places=4)

    def test_format_score(self):
        # Acceptance test: 6.666 -> 6.67/10
        self.assertEqual(format_score(6.666), "6.67/10")
        self.assertEqual(format_score(6.664), "6.66/10")
        self.assertEqual(format_score(10.0), "10.00/10")
        self.assertEqual(format_score(0.0), "0.00/10")

        # Clamping checks for out-of-bounds predictions
        self.assertEqual(format_score(11.5), "10.00/10")
        self.assertEqual(format_score(-1.5), "0.00/10")

    def test_save_metrics_and_predictions(self):
        metrics_data = {"test_rows": 10, "mae": 0.75, "r2": 0.80}
        prompts = ["Prompt A", "Prompt B"]
        y_true = [8.0, 6.0]
        y_pred = [7.8, 6.2]

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            pred_path = Path(tmpdir) / "preds.csv"

            save_metrics(metrics_data, metrics_path)
            self.assertTrue(metrics_path.exists())
            with open(metrics_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["mae"], 0.75)

            save_predictions(prompts, y_true, y_pred, pred_path)
            self.assertTrue(pred_path.exists())
            df = pd.read_csv(pred_path)
            self.assertEqual(len(df), 2)
            self.assertListEqual(list(df.columns), ["prompt", "actual_score", "predicted_score"])


if __name__ == "__main__":
    unittest.main()
