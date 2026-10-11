"""
Streamlit Web Application for PELS Prompt Quality Scorer.
"""

from pathlib import Path
import sys

# Ensure src directory is on module search path for local and Streamlit Cloud execution
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
from prompt_scorer.evaluate import format_score
from prompt_scorer.train import load_model

BASE_DIR = Path(__file__).resolve().parent
RF_MODEL_PATH = BASE_DIR / "models" / "pels_random_forest.joblib"
RIDGE_MODEL_PATH = BASE_DIR / "models" / "pels_ridge.joblib"


def get_available_model_path() -> Path:
    if RF_MODEL_PATH.exists():
        return RF_MODEL_PATH
    if RIDGE_MODEL_PATH.exists():
        return RIDGE_MODEL_PATH
    return RF_MODEL_PATH


st.set_page_config(
    page_title="PELS Prompt Quality Scorer",
    page_icon="",
    layout="centered",
)

st.title("PELS Prompt Quality Scorer")
st.write(
    "Enter a prompt to estimate its quality score out of 10 using "
    "our trained machine-learning model (Prompt Engineering Literacy Scale rubrics)."
)


@st.cache_resource
def cached_load_model(path: Path):
    return load_model(path)


model_path = get_available_model_path()

if not model_path.exists():
    st.error(
        "Model not found. Please train a model first by running:\n"
        "`python compare_models.py` or `python train.py`"
    )
    st.stop()

try:
    model = cached_load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model from {model_path}: {e}")
    st.stop()

prompt = st.text_area(
    "Your prompt",
    placeholder="Example: Explain machine learning to a beginner using three examples.",
    height=150,
)

if st.button("Score my prompt", type="primary"):
    cleaned_prompt = prompt.strip() if prompt else ""
    if not cleaned_prompt:
        st.warning("Please enter a prompt first.")
    else:
        try:
            raw_score = float(model.predict([cleaned_prompt])[0])
            formatted = format_score(raw_score, scale=10.0)
            st.metric(label="Predicted Prompt Quality Score", value=formatted)
            st.caption(
                "This is an experimental prediction, not an objective "
                "measure of prompt quality. Results may be inaccurate."
            )
        except Exception as e:
            st.error(f"Error predicting score: {e}")

st.divider()
st.caption("Built with Python, scikit-learn, and Streamlit.")
