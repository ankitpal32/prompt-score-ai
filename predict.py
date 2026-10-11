
import joblib
from pathlib import Path

MODEL_PATH = Path("models/pels_random_forest.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run compare_models.py first."
    )

model = joblib.load(MODEL_PATH)

print("PELS Prompt Quality Scorer")
print("Type 'exit' to quit.\n")

while True:
    prompt = input("Enter a prompt: ").strip()

    if prompt.lower() == "exit":
        break

    if not prompt:
        print("Please enter a non-empty prompt.\n")
        continue

    score = model.predict([prompt])[0]
    print(f"Predicted quality score: {score:.2f}\n")
