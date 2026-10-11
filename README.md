# Prompt Score AI 🎯

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Dataset](https://img.shields.io/badge/Dataset-AIKosh%20PELS-008080.svg)](https://aikosh.indiaai.gov.in/)

An end-to-end, modular machine learning project that scores and predicts the quality of text prompts based on empirical rubrics from the **Prompt Engineering Literacy Scale (PELS)** dataset.

---

## 📖 Overview

As generative AI becomes integral to technical and creative workflows, formulating precise, structured, and effective prompts is an essential skill. **Prompt Score AI** applies natural language processing and classical regression models to objectively evaluate prompt effectiveness.

Using human-annotated ratings from the **Prompt Engineering Literacy Scale (PELS)** hosted on India's national AI platform (**AIKosh**), the system transforms unstructured prompt text using sublinear n-gram TF-IDF vectorization and predicts continuous quality scores via optimized regression pipelines.

> ⚠️ **Experimental Disclaimer**: Predictions are experimental estimates produced by statistical text models and should not be treated as an absolute or objective benchmark of prompt utility. Outputs may be inaccurate for domain-specific, ambiguous, or out-of-distribution prompts.

---

## ✨ Key Features

- **Modular ML Architecture**: Clean separation of concerns with a dedicated `prompt_scorer` core package (`data.py`, `train.py`, `evaluate.py`).
- **Automated Prompt Scoring**: Predicts numerical quality scores for arbitrary user prompts in real-time.
- **Dual Pipeline Architecture**: Implements and benchmarks both L2-regularized linear modeling (**Ridge Regression**) and non-linear ensemble modeling (**Random Forest Regression**).
- **Interactive Web Interface**: Clean, accessible web dashboard powered by **Streamlit** for quick interactive scoring and experimentation.
- **Terminal CLI Mode**: A lightweight, interactive command-line loop (`predict.py`) for scoring prompts directly from the shell.
- **Reproducible Evaluation**: Automated cleaning, de-duplication, train-test splitting (80/20), dummy baseline benchmarking, and structured reporting (`metrics.json` and `test_predictions.csv`).
- **Comprehensive Unit Tests**: Built-in test suite covering data normalization, cleaning, pipeline construction, persistence, and evaluation logic.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3 | Core programming language |
| **Data Processing** | Pandas, OpenPyXL | Dataset ingestion, deduplication, and manipulation |
| **Feature Extraction** | scikit-learn (`TfidfVectorizer`) | Sublinear TF, word (1, 2)-grams, max 20,000 features |
| **Modeling** | scikit-learn (`Ridge`, `RandomForestRegressor`) | Linear and ensemble regression algorithms |
| **Model Persistence**| Joblib | Serialization and loading of trained pipeline artifacts |
| **Web UI** | Streamlit | Lightweight, interactive web interface |
| **Testing** | unittest | Standardized test discovery and assertions |

---

## 📊 Model Evaluation & Benchmarks

All models were evaluated on the held-out test split (**333 samples**, 20% test size, `random_state=42`) after removing invalid and duplicate entries from the 1,665 unique records in the PELS dataset.

| Model / Pipeline | Target Metric | MAE (Mean Absolute Error) ↓ | R² Score ↑ | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (Dummy Regressor)** | Mean Strategy | `1.7789` | `0.0000` | Baseline |
| **TF-IDF + Ridge Regression** | $\alpha = 1.0$ | `0.8722` | `0.7576` | Trained via `train.py` |
| **TF-IDF + Random Forest** | $n\_estimators = 100$ | **`0.7956`** | **`0.7608`** | **Best Performing Model** |

### Key Takeaways
- The **Random Forest Regressor** achieves the lowest error (**MAE: 0.7956**) and explains **~76.08%** of score variance ($R^2 = 0.7608$).
- Both regression pipelines substantially outperform the naive mean-predicting baseline (**MAE: 1.7789**).

---

## 📁 Project Structure

```text
prompt-score-ai/
├── app.py                              # Streamlit web application
├── train.py                            # Pipeline training & evaluation for Ridge
├── compare_models.py                   # Model comparison & Random Forest training
├── predict.py                          # Interactive CLI scoring interface
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
├── .gitignore                          # Clean Git ignore rules
│
├── src/
│   └── prompt_scorer/                  # Core ML package
│       ├── __init__.py                 # Package exports & versioning
│       ├── data.py                     # Data loading, cleaning & train-test split
│       ├── train.py                    # Model architectures & persistence
│       └── evaluate.py                 # Evaluation metrics & reporting
│
├── data/
│   └── PELS_Final_Justification.xlsx   # PELS dataset from AIKosh
├── models/
│   ├── pels_random_forest.joblib       # Best performing model (Random Forest)
│   └── pels_ridge.joblib               # Linear model (Ridge regression)
├── reports/
│   ├── metrics.json                    # Evaluation metrics
│   └── test_predictions.csv           # Test set predictions & actuals
└── tests/
    ├── __init__.py
    ├── test_data.py                    # Unit tests for data processing
    ├── test_train.py                   # Unit tests for pipelines & persistence
    └── test_evaluate.py                # Unit tests for evaluation logic
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/ankitpal32/prompt-score-ai.git
cd prompt-score-ai
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📂 Dataset Setup

The model is trained on the **Prompt Engineering Literacy Scale (PELS)** dataset sourced from India's **AIKosh** open repository.

1. Ensure the dataset spreadsheet `PELS_Final_Justification.xlsx` is placed in the `data/` directory:
   ```text
   data/PELS_Final_Justification.xlsx
   ```
2. The dataset includes prompt texts along with evaluative ratings (`Final Score`).

---

## 🏋️ Model Training & Comparison

### Train Ridge Regression Pipeline
Runs dataset cleaning, train/test split, fits the TF-IDF + Ridge model, outputs metrics to `reports/metrics.json`, and saves `models/pels_ridge.joblib`:

```bash
python train.py
```

### Train Random Forest Regressor
Fits the TF-IDF + Random Forest model, logs MAE and R² on the test set, and saves `models/pels_random_forest.joblib`:

```bash
python compare_models.py
```

---

## 💻 Running the Application

### 1. Interactive Web Interface (Streamlit)

Launch the Streamlit interface in your browser:

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501`, enter any prompt text, and click **Score my prompt** to view the estimated score.

### 2. Command-Line Scoring (CLI)

For quick, headless testing directly inside the terminal:

```bash
python predict.py
```

*Type your prompt when prompted, or enter `exit` to quit.*

---

## 🧪 Running Unit Tests

The test suite is built on standard `unittest` and requires no additional dependencies:

```bash
python -m unittest discover tests
```

---

## ☁️ Deployment & GitHub Readiness

- **Streamlit Community Cloud**: Ready for immediate deployment. Point Streamlit to `app.py` in this repository; dependencies in `requirements.txt` will automatically install and run.
- **Model Checkpoints**: Serialized model artifacts (`.joblib`) and reports are located in `models/` and `reports/`.
- **Git Hygiene**: Comprehensive `.gitignore` ignores Python bytecode, virtual environments (`.venv/`), operating system caches, and local editor settings.

---

## ⚠️ Limitations & Disclaimers

- **Statistical Heuristics**: The models rely on TF-IDF n-gram token statistics and cannot fully capture complex semantics, reasoning chains, or prompt intent like modern deep language models.
- **Dataset Domain Specificity**: The scoring rubrics reflect the specific tasks, prompts, and evaluators within the PELS study. Performance may degrade on prompts with syntax or tasks unseen in the training distribution.
- **Subjective Nature of Quality**: Prompt performance varies across different target LLMs (e.g., GPT-4, Claude, Gemini, open weights). A prompt scored moderately here may still perform effectively for a specific niche use case.
- **Experimental Tool**: This software is intended for research, learning, and exploratory purposes.

---

## 🔮 Future Improvements

- [ ] **Dense Embeddings**: Benchmark pretrained sentence encoders (e.g., Sentence-Transformers / MiniLM) alongside TF-IDF.
- [ ] **Cross-Validation**: Incorporate stratified k-fold cross-validation across difficulty tiers.
- [ ] **Feature Engineering**: Incorporate explicit readability indices, token counts, and structural prompt markers (e.g., delimiters, few-shot examples).
- [ ] **Explainability**: Integrate SHAP or LIME to highlight which words or phrases contribute positively or negatively to the score.
- [ ] **Dimension-Level Scoring**: Predict individual sub-dimension scores (e.g., clarity, specificity, context) in addition to the aggregate final score.

---

## 📜 Dataset Attribution & Acknowledgements

- **Dataset**: Prompt Engineering Literacy Scale (PELS)
- **Source**: [AIKosh](https://aikosh.indiaai.gov.in/) — India's National AI Artifact & Dataset Repository
- Developed with [scikit-learn](https://scikit-learn.org/) and [Streamlit](https://streamlit.io/).
