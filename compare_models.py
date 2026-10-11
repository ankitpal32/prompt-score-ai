
from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Reuse the exact cleaned dataset and split used in train.py.
DATA_PATH = Path("data/PELS_Final_Justification.xlsx")

df = pd.read_excel(DATA_PATH)

prompt_col = "Prompt Text"
target_col = "Final\nScore"

# Adjust these two names if your spreadsheet differs.
data = df[[prompt_col, target_col]].copy()
data.columns = ["prompt", "score"]
data["prompt"] = data["prompt"].astype("string").str.strip()
data["score"] = pd.to_numeric(data["score"], errors="coerce")
data = data.dropna(subset=["prompt", "score"])
data = data[data["prompt"].str.len() > 0]
data = data.drop_duplicates(subset=["prompt"], keep="first")

X_train, X_test, y_train, y_test = train_test_split(
    data["prompt"],
    data["score"],
    test_size=0.20,
    random_state=42,
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Random Forest results")
print(f"Test samples: {len(y_test)}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(f"R2:  {r2_score(y_test, predictions):.4f}")

joblib.dump(model, "models/pels_random_forest.joblib")
print("Random Forest model saved successfully!")