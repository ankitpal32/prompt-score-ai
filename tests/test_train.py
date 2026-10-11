"""
Tests for model building, training, and persistence pipelines.
"""

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_scorer.train import (
    build_baseline_model,
    build_rf_pipeline,
    build_ridge_pipeline,
    load_model,
    save_model,
    train_model,
)


class TestTrainPipelines(unittest.TestCase):
    def setUp(self):
        self.X_toy = pd.Series([
            "Write a python quicksort function with unit tests",
            "Give me a recipe for chocolate chip cookies",
            "Explain quantum computing principles to a child",
            "Draft a professional cold email for marketing outreach",
            "Summarize the French Revolution causes and effects",
        ])
        self.y_toy = pd.Series([9.2, 5.0, 7.5, 6.8, 8.1])

    def test_build_ridge_pipeline(self):
        pipeline = build_ridge_pipeline(alpha=0.5, max_features=100)
        self.assertIn("tfidf", pipeline.named_steps)
        self.assertIn("regressor", pipeline.named_steps)

        train_model(pipeline, self.X_toy, self.y_toy)
        preds = pipeline.predict(["Write quicksort in python"])
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], (float, np.floating))

    def test_build_rf_pipeline(self):
        pipeline = build_rf_pipeline(n_estimators=10, max_features=100)
        self.assertIn("tfidf", pipeline.named_steps)
        self.assertIn("regressor", pipeline.named_steps)

        train_model(pipeline, self.X_toy, self.y_toy)
        preds = pipeline.predict(["Chocolate chip recipe"])
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(preds[0], (float, np.floating))

    def test_build_baseline_model(self):
        baseline = build_baseline_model(strategy="mean")
        baseline.fit(self.X_toy.to_frame(), self.y_toy)
        preds = baseline.predict(pd.Series(["Any prompt"]).to_frame())
        self.assertAlmostEqual(preds[0], float(self.y_toy.mean()), places=4)

    def test_save_and_load_model(self):
        pipeline = build_ridge_pipeline(alpha=1.0)
        train_model(pipeline, self.X_toy, self.y_toy)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "test_model.joblib"
            save_model(pipeline, temp_path)
            self.assertTrue(temp_path.exists())

            loaded = load_model(temp_path)
            preds_orig = pipeline.predict(["Test prompt"])
            preds_loaded = loaded.predict(["Test prompt"])
            self.assertAlmostEqual(preds_orig[0], preds_loaded[0], places=5)


if __name__ == "__main__":
    unittest.main()
