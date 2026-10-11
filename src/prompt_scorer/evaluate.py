"""
Evaluation metrics computation, comparison, and reporting utilities for PELS Prompt Scorer.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def format_score(score: float, scale: float = 10.0) -> str:
    """
    Format a numerical score out of scale (default 10) with exactly two decimal places.
    Clamps the score within [0.0, scale] to ensure valid range presentation.

    Example:
        format_score(6.666) -> "6.67/10"
        format_score(10.2)  -> "10.00/10"
        format_score(-0.5)  -> "0.00/10"
    """
    clamped = max(0.0, min(float(score), float(scale)))
    return f"{clamped:.2f}/{int(scale)}"


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    baseline_pred: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Calculate regression evaluation metrics (MAE, R2, baseline MAE).
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None

    results: Dict[str, Any] = {
        "test_rows": int(len(y_true)),
        "mae": mae,
        "r2": r2,
    }

    if baseline_pred is not None:
        results["baseline_mae"] = float(mean_absolute_error(y_true, baseline_pred))

    return results


def save_metrics(metrics: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """Save metrics dictionary to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def save_predictions(
    prompts: Any,
    y_true: Any,
    y_pred: Any,
    filepath: Union[str, Path],
) -> Path:
    """Save test prompts, actual scores, and predictions to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "prompt": pd.Series(prompts).reset_index(drop=True),
        "actual_score": pd.Series(y_true).reset_index(drop=True),
        "predicted_score": pd.Series(y_pred).reset_index(drop=True),
    })
    df.to_csv(path, index=False)
    return path
