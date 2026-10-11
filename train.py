
from pathlib import Path
import json
import re

import joblib
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/PELS_Final_Justification.xlsx")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def normalize(name):
    """Normalize column names for matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns, candidates):
    lookup = {normalize(c): c for c in columns}

    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]

    return None


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Excel file from AIKosh and place it in data/."
        )

    # 1. Load data
    df = pd.read_excel(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nAvailable columns:")
    for col in df.columns:
        print(f"- {col}")

    # 2. Identify the input and target
    prompt_col = find_column(
        df.columns,
        ["prompt", "prompt_text", "input_prompt", "text"]
    )

    target_col = find_column(
        df.columns,
        [
            "final_weighted_score",
            "final weighted score",
            "weighted_score",
            "overall_score",
            "final_score",
        ]
    )

    if prompt_col is None or target_col is None:
        raise ValueError(
            "\nCould not identify the required columns.\n"
            f"Detected prompt column: {prompt_col}\n"
            f"Detected score column: {target_col}\n"
            "Edit the candidate lists in train.py using the "
            "actual column names printed above."
        )

    print(f"\nInput column: {prompt_col}")
    print(f"Target column: {target_col}")

    # 3. Keep only the required columns
    data = df[[prompt_col, target_col]].copy()
    data.columns = ["prompt", "score"]

    data["prompt"] = data["prompt"].astype("string").str.strip()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")

    # Remove missing, blank, or invalid records
    data = data.dropna(subset=["prompt", "score"])
    data = data[data["prompt"].str.len() > 0]

    # Remove exact duplicate prompts before splitting
    data["prompt_key"] = data["prompt"].str.casefold()
    data = data.drop_duplicates(subset=["prompt_key"])
    data = data.drop(columns=["prompt_key"])

    if len(data) < 10:
        raise ValueError(
            f"Only {len(data)} usable unique records found. "
            "Inspect the spreadsheet before training."
        )

    print("\nUsable unique records:", len(data))
    print("\nScore distribution:")
    print(data["score"].describe())

    # 4. Split before fitting any text transformations
    X_train, X_test, y_train, y_test = train_test_split(
        data["prompt"],
        data["score"],
        test_size=0.20,
        random_state=42,
    )

    # 5. Baseline: always predict the training-set mean
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train.to_frame(), y_train)
    baseline_pred = baseline.predict(X_test.to_frame())

    # 6. TF-IDF + Ridge Regression
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            ),
        ),
        ("regressor", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 7. Evaluate on held-out test data
    metrics = {
        "dataset_rows_after_cleaning": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": str(target_col),
        "baseline_mae": float(
            mean_absolute_error(y_test, baseline_pred)
        ),
        "model_mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "model_r2": (
            float(r2_score(y_test, predictions))
            if len(y_test) >= 2 else None
        ),
    }

    print("\nEvaluation results:")
    print(json.dumps(metrics, indent=2))

    # 8. Save the model and evaluation artifacts
    joblib.dump(model, MODEL_DIR / "pels_ridge.joblib")

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    examples = pd.DataFrame({
        "prompt": X_test.reset_index(drop=True),
        "actual_score": y_test.reset_index(drop=True),
        "predicted_score": predictions,
    })
    examples.to_csv(
        REPORT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\nSaved:")
    print("- models/pels_ridge.joblib")
    print("- reports/metrics.json")
    print("- reports/test_predictions.csv")


if __name__ == "__main__":
    main()
