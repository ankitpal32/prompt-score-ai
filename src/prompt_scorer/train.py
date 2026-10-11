"""
Model definitions, training pipelines, and persistence utilities for PELS Prompt Scorer.
"""

from pathlib import Path
from typing import Any, Tuple, Union

import joblib
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def build_tfidf_vectorizer(
    lowercase: bool = True,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Create a configured TfidfVectorizer for prompt text."""
    return TfidfVectorizer(
        lowercase=lowercase,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def build_ridge_pipeline(
    alpha: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    sublinear_tf: bool = True,
) -> Pipeline:
    """Construct an end-to-end Pipeline with TF-IDF and Ridge Regression."""
    return Pipeline([
        (
            "tfidf",
            build_tfidf_vectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
            ),
        ),
        ("regressor", Ridge(alpha=alpha)),
    ])


def build_rf_pipeline(
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    sublinear_tf: bool = True,
) -> Pipeline:
    """Construct an end-to-end Pipeline with TF-IDF and Random Forest Regressor."""
    return Pipeline([
        (
            "tfidf",
            build_tfidf_vectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                sublinear_tf=sublinear_tf,
            ),
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            ),
        ),
    ])


def build_baseline_model(strategy: str = "mean") -> DummyRegressor:
    """Construct a DummyRegressor baseline."""
    return DummyRegressor(strategy=strategy)


def train_model(pipeline: Any, X_train: Any, y_train: Any) -> Any:
    """Fit a model or pipeline on training data."""
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model: Any, filepath: Union[str, Path]) -> Path:
    """Serialize model artifact using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath: Union[str, Path]) -> Any:
    """Deserialize model artifact using joblib."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {path.resolve()}\n"
            "Please train or download the model before making predictions."
        )
    return joblib.load(path)
