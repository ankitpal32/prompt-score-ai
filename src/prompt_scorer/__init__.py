"""
Prompt Scorer - Machine learning package for scoring prompt quality using PELS rubrics.
"""

__version__ = "0.1.0"

from .data import find_column, load_and_clean_data, normalize, split_data
from .evaluate import (
    evaluate_predictions,
    format_score,
    save_metrics,
    save_predictions,
)
from .train import (
    build_baseline_model,
    build_rf_pipeline,
    build_ridge_pipeline,
    build_tfidf_vectorizer,
    load_model,
    save_model,
    train_model,
)

__all__ = [
    "normalize",
    "find_column",
    "load_and_clean_data",
    "split_data",
    "build_tfidf_vectorizer",
    "build_ridge_pipeline",
    "build_rf_pipeline",
    "build_baseline_model",
    "train_model",
    "save_model",
    "load_model",
    "evaluate_predictions",
    "format_score",
    "save_metrics",
    "save_predictions",
]
