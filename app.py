from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluate import METRIC_COLUMNS, compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "Class"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return models, encoder


models, encoder = load_artifacts()
feature_cols = list(models["Logistic Regression"].feature_names_in_)

st.title("Dry Bean Classifier Lab")
st.caption(
    "Evaluate five pre-trained classifiers on your own test data. "
    "Models are loaded from disk and never retrained on the uploaded file."
)

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    st.markdown(f"Required columns: {len(feature_cols)} features and the target `{TARGET}`.")
    sample = (ROOT / "test_data.csv").read_bytes()
    st.download_button("Download sample test_data.csv", sample, "test_data.csv", "text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin. You can download a sample file there.")
    st.stop()

data = pd.read_csv(uploaded)
missing = [c for c in feature_cols + [TARGET] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

unknown = set(data[TARGET]) - set(encoder.classes_)
if unknown:
    st.error(f"Unknown class labels in `{TARGET}`: {', '.join(map(str, unknown))}")
    st.stop()

X = data[feature_cols]
y_true = encoder.transform(data[TARGET])
n_classes = len(encoder.classes_)

st.subheader("Uploaded data preview")
st.write(f"{len(data):,} rows, {len(feature_cols)} features")
st.dataframe(data.head(), use_container_width=True)

st.subheader(f"Results: {model_name}")
model = models[model_name]
scores = compute_metrics(model, X, y_true, n_classes)

cols = st.columns(len(METRIC_COLUMNS))
for col, metric in zip(cols, METRIC_COLUMNS):
    value = scores[metric]
    col.metric(metric, "n/a" if pd.isna(value) else f"{value:.4f}")
if pd.isna(scores["AUC"]):
    st.caption("AUC needs every class to appear in the uploaded file.")

left, right = st.columns(2)
y_pred = model.predict(X)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu",
        xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y_true, y_pred, labels=range(n_classes),
        target_names=encoder.classes_, output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.subheader("All models on this test data")
comparison = pd.DataFrame(
    [{"ML Model Name": n, **compute_metrics(m, X, y_true, n_classes)} for n, m in models.items()]
).set_index("ML Model Name")
st.dataframe(comparison.round(4), use_container_width=True)
st.bar_chart(comparison["Accuracy"])
