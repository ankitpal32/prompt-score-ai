"""
Model comparison and Random Forest training script for PELS Prompt Scorer.

Trains a TF-IDF + Random Forest Regressor pipeline using the exact same split,
evaluates on held-out test data, and saves models/pels_random_forest.joblib.
"""

from pathlib import Path
import sys

# Ensure src directory is on module search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from prompt_scorer.data import DEFAULT_DATA_PATH, load_and_clean_data, split_data
from prompt_scorer.train import build_rf_pipeline, save_model, train_model
from prompt_scorer.evaluate import evaluate_predictions

MODEL_PATH = Path("models/pels_random_forest.joblib")


def main():
    print("\n--- 1. Loading & Splitting Data ---")
    data, _, _ = load_and_clean_data(DEFAULT_DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(data, test_size=0.20, random_state=42)

    print("\n--- 2. Training TF-IDF + Random Forest Pipeline ---")
    model = build_rf_pipeline(n_estimators=100, random_state=42, n_jobs=-1)
    train_model(model, X_train, y_train)

    print("\n--- 3. Evaluating on Test Split ---")
    predictions = model.predict(X_test)
    results = evaluate_predictions(y_test, predictions)

    print("Random Forest Results:")
    print(f"Test samples: {results['test_rows']}")
    print(f"MAE:          {results['mae']:.4f}")
    print(f"R²:           {results['r2']:.4f}")

    save_model(model, MODEL_PATH)
    print(f"\nRandom Forest model saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    main()