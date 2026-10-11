"""
Data processing and preparation utilities for the PELS Prompt Scorer.
"""

from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path("data/PELS_Final_Justification.xlsx")

PROMPT_COLUMN_CANDIDATES = [
    "prompt text",
    "prompt_text",
    "prompt",
    "input_prompt",
    "text",
]

TARGET_COLUMN_CANDIDATES = [
    "final score",
    "final_score",
    "final_weighted_score",
    "final weighted score",
    "weighted_score",
    "overall_score",
]


def normalize(name: str) -> str:
    """Normalize string column names for case and special character insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching column name in candidates based on normalized string comparison."""
    lookup = {normalize(c): c for c in columns}
    for candidate in candidates:
        if normalize(candidate) in lookup:
            return lookup[normalize(candidate)]
    return None


def clean_dataframe(
    df: pd.DataFrame,
    prompt_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Extract, clean, and deduplicate prompt and score columns from a DataFrame."""
    data = df[[prompt_col, target_col]].copy()
    data.columns = ["prompt", "score"]

    data["prompt"] = data["prompt"].astype("string").str.strip()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")

    # Remove missing, blank, or invalid records
    data = data.dropna(subset=["prompt", "score"])
    data = data[data["prompt"].str.len() > 0]

    # Remove exact duplicate prompts case-insensitively
    data["prompt_key"] = data["prompt"].str.casefold()
    data = data.drop_duplicates(subset=["prompt_key"], keep="first")
    data = data.drop(columns=["prompt_key"])

    return data.reset_index(drop=True)


def load_and_clean_data(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    prompt_candidates: Optional[List[str]] = None,
    target_candidates: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Load the PELS Excel dataset, identify required columns, and clean the data.

    Returns:
        Tuple containing (cleaned_df, detected_prompt_col, detected_target_col)
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path.resolve()}\n"
            "Please ensure PELS_Final_Justification.xlsx is placed in data/."
        )

    df = pd.read_excel(path)

    prompt_col = find_column(
        list(df.columns),
        prompt_candidates or PROMPT_COLUMN_CANDIDATES,
    )
    target_col = find_column(
        list(df.columns),
        target_candidates or TARGET_COLUMN_CANDIDATES,
    )

    if prompt_col is None or target_col is None:
        raise ValueError(
            "Could not identify the required prompt and score columns.\n"
            f"Detected prompt column: {prompt_col}\n"
            f"Detected target column: {target_col}\n"
            f"Available columns: {list(df.columns)}"
        )

    cleaned_data = clean_dataframe(df, prompt_col, target_col)

    if len(cleaned_data) < 10:
        raise ValueError(
            f"Only {len(cleaned_data)} usable unique records found. "
            "Please inspect the spreadsheet before training."
        )

    return cleaned_data, prompt_col, target_col


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split clean DataFrame into X_train, X_test, y_train, y_test."""
    return train_test_split(
        data["prompt"],
        data["score"],
        test_size=test_size,
        random_state=random_state,
    )
