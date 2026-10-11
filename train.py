"""
Training script for the PELS Prompt Scorer (Ridge Regression Pipeline).

Loads and cleans the PELS dataset, fits a TF-IDF + Ridge Regression pipeline,
evaluates on held-out test data, and saves model and evaluation reports.
"""

from pathlib import Path
import json
import sys

# Ensure src directory is on module search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from prompt_scorer.data import DEFAULT_DATA_PATH, load_and_clean_data, split_data
from prompt_scorer.train import (
    build_baseline_model,
    build_ridge_pipeline,
    save_model,
    train_model,
)
from prompt_scorer.evaluate import save_metrics, save_predictions

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Loading & Cleaning Dataset ---")
    data, prompt_col, target_col = load_and_clean_data(DEFAULT_DATA_PATH)
    print(f"Detected input column:  '{prompt_col}'")
    print(f"Detected target column: '{target_col}'")
    print(f"Cleaned unique records: {len(data)}")

    print("\nScore summary statistics:")
    print(data["score"].describe().to_string())

    print("\n--- 2. Splitting Data (80/20 train/test) ---")
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.20, random_state=42)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    print("\n--- 3. Fitting Baseline (Mean Regressor) ---")
    baseline = build_baseline_model(strategy="mean")
    baseline.fit(X_train.to_frame(), y_train)
    baseline_preds = baseline.predict(X_test.to_frame())

    print("\n--- 4. Training TF-IDF + Ridge Regression Pipeline ---")
    model = build_ridge_pipeline(alpha=1.0)
    train_model(model, X_train, y_train)
    predictions = model.predict(X_test)

    print("\n--- 5. Evaluating on Test Split ---")
    from sklearn.metrics import mean_absolute_error, r2_score

    metrics = {
        "dataset_rows_after_cleaning": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": str(target_col),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_preds)),
        "model_mae": float(mean_absolute_error(y_test, predictions)),
        "model_r2": float(r2_score(y_test, predictions)) if len(y_test) >= 2 else None,
    }

    print("\nEvaluation metrics:")
    print(json.dumps(metrics, indent=2))

    print("\n--- 6. Saving Artifacts ---")
    model_path = MODEL_DIR / "pels_ridge.joblib"
    metrics_path = REPORT_DIR / "metrics.json"
    predictions_path = REPORT_DIR / "test_predictions.csv"

    save_model(model, model_path)
    save_metrics(metrics, metrics_path)
    save_predictions(X_test, y_test, predictions, predictions_path)

    print(f"Saved model:       {model_path}")
    print(f"Saved metrics:     {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
