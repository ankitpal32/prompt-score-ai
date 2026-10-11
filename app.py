
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "pels_random_forest.joblib"

st.set_page_config(
    page_title="PELS Prompt Quality Scorer",
    page_icon="🎯",
    layout="centered",
)

st.title("PELS Prompt Quality Scorer")
st.write(
    "Enter a prompt to estimate its quality score using "
    "our trained machine-learning model."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model not found. Run compare_models.py first.")
    st.stop()

model = load_model()

prompt = st.text_area(
    "Your prompt",
    placeholder="Example: Explain machine learning to a beginner using three examples.",
    height=150,
)

if st.button("Score my prompt", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        score = float(model.predict([prompt.strip()])[0])
        st.metric("Predicted quality score", f"{score:.2f}")
        st.caption(
            "This is an experimental prediction, not an objective "
            "measure of prompt quality. Results may be inaccurate."
        )

st.divider()
st.caption("Built with Python, scikit-learn, and Streamlit.")
