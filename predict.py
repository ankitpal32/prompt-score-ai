"""
CLI inference script for PELS Prompt Scorer.

Provides an interactive terminal prompt to score prompts using the trained model.
"""

from pathlib import Path
import sys

# Ensure src directory is on module search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from prompt_scorer.evaluate import format_score
from prompt_scorer.train import load_model

RF_MODEL_PATH = Path("models/pels_random_forest.joblib")
RIDGE_MODEL_PATH = Path("models/pels_ridge.joblib")


def get_default_model_path() -> Path:
    """Find the best available model artifact."""
    if RF_MODEL_PATH.exists():
        return RF_MODEL_PATH
    if RIDGE_MODEL_PATH.exists():
        return RIDGE_MODEL_PATH
    raise FileNotFoundError(
        "No trained model artifact found.\n"
        "Please train a model first using 'python compare_models.py' or 'python train.py'."
    )


def main():
    model_path = get_default_model_path()
    print(f"Loading model from: {model_path}...")
    model = load_model(model_path)

    print("\n========================================")
    print("       PELS Prompt Quality Scorer       ")
    print("========================================")
    print("Enter your prompt to score it out of 10.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("Enter a prompt: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter a non-empty prompt.\n")
            continue

        try:
            score = float(model.predict([prompt])[0])
            formatted = format_score(score, scale=10.0)
            print(f"-> Predicted Prompt Quality Score: {formatted}\n")
        except Exception as e:
            print(f"Error predicting score: {e}\n")


if __name__ == "__main__":
    main()
