"""
Tests for data processing and cleaning utilities.
"""

import sys
from pathlib import Path
import unittest

import pandas as pd

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompt_scorer.data import (
    clean_dataframe,
    find_column,
    normalize,
    split_data,
)


class TestDataProcessing(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Prompt Text"), "prompttext")
        self.assertEqual(normalize("Final\nScore"), "finalscore")
        self.assertEqual(normalize("Weighted_Score!"), "weightedscore")

    def test_find_column(self):
        cols = ["Prompt Text", "Domain", "Final\nScore"]
        self.assertEqual(
            find_column(cols, ["prompt_text", "prompt"]),
            "Prompt Text",
        )
        self.assertEqual(
            find_column(cols, ["final_score", "score"]),
            "Final\nScore",
        )
        self.assertIsNone(find_column(cols, ["non_existent_column"]))

    def test_clean_dataframe(self):
        raw_df = pd.DataFrame({
            "prompt_raw": [
                "  Explain recursion  ",
                "Explain recursion",  # Duplicate casefold
                "",                     # Empty string
                "   ",                  # Whitespace only
                None,                   # Missing prompt
                "Write a poem",
                "Generate python code",
            ],
            "score_raw": [
                8.5,
                9.0,
                7.0,
                6.0,
                5.0,
                "not_a_number",        # Invalid numeric score
                4.2,
            ],
        })

        cleaned = clean_dataframe(raw_df, "prompt_raw", "score_raw")

        # Only 2 rows should survive: "Explain recursion" (first instance) and "Generate python code"
        self.assertEqual(len(cleaned), 2)
        self.assertListEqual(list(cleaned.columns), ["prompt", "score"])
        self.assertEqual(cleaned.iloc[0]["prompt"], "Explain recursion")
        self.assertEqual(cleaned.iloc[0]["score"], 8.5)
        self.assertEqual(cleaned.iloc[1]["prompt"], "Generate python code")
        self.assertEqual(cleaned.iloc[1]["score"], 4.2)

    def test_split_data(self):
        df = pd.DataFrame({
            "prompt": [f"Prompt sample {i}" for i in range(50)],
            "score": [float(i) for i in range(50)],
        })
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
        self.assertEqual(len(X_train), 40)
        self.assertEqual(len(X_test), 10)
        self.assertEqual(len(y_train), 40)
        self.assertEqual(len(y_test), 10)


if __name__ == "__main__":
    unittest.main()
