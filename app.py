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

st.set_page_config(page_title="Dry Bean Model Bench", page_icon="DB", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #20352f;
        --muted: #64746e;
        --green: #277568;
        --coral: #c5674e;
        --line: #dce7e1;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 0%, rgba(201, 225, 214, .42), transparent 30rem),
            linear-gradient(180deg, #f8fbf8 0%, #ffffff 35rem);
        color: var(--ink);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f6f2 0%, #f8faf8 100%);
        border-right: 1px solid var(--line);
    }
    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: 0 !important;
    }
    h1 {
        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 2.65rem !important;
        line-height: 1.08 !important;
    }
    .eyebrow {
        color: var(--green);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: .4rem 0 .45rem;
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.06rem;
        margin: 0 0 1.2rem;
    }
    .data-ribbon {
        display: flex;
        flex-wrap: wrap;
        gap: 1.7rem 3.2rem;
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 1rem 0;
        margin: .4rem 0 .55rem;
    }
    .data-ribbon div { display: grid; gap: .12rem; }
    .data-ribbon strong {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.35rem;
        font-weight: 600;
    }
    .data-ribbon span { color: var(--muted); font-size: .78rem; }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, .82);
        border: 1px solid var(--line);
        border-top: 3px solid var(--green);
        padding: .8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    .stDownloadButton button {
        border: 1px solid var(--green);
        color: var(--green);
        border-radius: 4px;
    }
    .stDownloadButton button:hover {
        border-color: var(--coral);
        color: var(--coral);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="eyebrow">UCI DRY BEAN DATASET · MULTI-CLASS CLASSIFICATION</p>', unsafe_allow_html=True)
st.title("Dry Bean Model Bench")
st.markdown(
    '<p class="hero-copy">One held-out dataset, five classifiers, and a clear side-by-side evaluation.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="data-ribbon">
        <div><strong>13,543</strong><span>cleaned observations</span></div>
        <div><strong>16</strong><span>image-derived features</span></div>
        <div><strong>7</strong><span>bean varieties</span></div>
        <div><strong>5</strong><span>pre-trained classifiers</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Source: UCI Dry Bean Dataset. Upload labeled test data to evaluate the saved models; uploaded data is never used for training.")


@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return models, encoder


models, encoder = load_artifacts()
feature_cols = list(models["Logistic Regression"].feature_names_in_)

with st.sidebar:
    st.header("Evaluation setup")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    st.caption(f"Include all {len(feature_cols)} numeric features and the `{TARGET}` target column.")
    sample = (ROOT / "test_data.csv").read_bytes()
    st.download_button("Download sample test_data.csv", sample, "test_data.csv", "text/csv")

    st.header("Results view")
    view_mode = st.radio("Choose an analysis", ["Single model", "Compare models"], key="view_mode")
    if view_mode == "Single model":
        model_name = st.selectbox(
            "Choose a model",
            list(MODEL_FILES),
            index=None,
            placeholder="Select a classifier",
            key="single_model",
        )
    else:
        selected_models = st.multiselect(
            "Choose 2 to 5 models",
            list(MODEL_FILES),
            default=["Logistic Regression", "Support Vector Machine"],
            max_selections=len(MODEL_FILES),
            key="comparison_models",
        )

if uploaded is None:
    st.info("Upload a labeled test CSV to begin. The sample test file is available in the sidebar.")
    st.subheader("About this dataset")
    st.write(
        "Dry Bean contains shape measurements extracted from bean images. "
        "The seven labels are BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA. "
        "The provided sample is the held-out test split used for evaluation."
    )
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
if data.empty:
    st.error("The uploaded CSV has no rows to evaluate.")
    st.stop()

X = data[feature_cols]
y_true = encoder.transform(data[TARGET])
n_classes = len(encoder.classes_)

st.subheader("Uploaded test data")
summary_cols = st.columns(4)
summary_cols[0].metric("Rows", f"{len(data):,}")
summary_cols[1].metric("Features", len(feature_cols))
summary_cols[2].metric("Varieties present", data[TARGET].nunique())
summary_cols[3].metric("File", uploaded.name)

preview_col, distribution_col = st.columns([1.5, 1])
with preview_col:
    st.markdown("**Data preview**")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)
with distribution_col:
    st.markdown("**Variety distribution**")
    class_counts = (
        data[TARGET].value_counts().reindex(encoder.classes_, fill_value=0)
        .rename_axis("Variety").to_frame("Rows")
    )
    st.bar_chart(class_counts)
    most_common = class_counts["Rows"].idxmax()
    st.caption(f"Most represented in this upload: {most_common} ({class_counts.loc[most_common, 'Rows']:,} rows).")

if view_mode == "Single model":
    if model_name is None:
        st.info("Test data is ready. Choose a classifier in the sidebar to start the evaluation.")
        st.caption("Select one classifier in the sidebar to see its metrics and diagnostics.")
        st.stop()

    model = models[model_name]
    st.subheader(f"Single-model results · {model_name}")
    scores = compute_metrics(model, X, y_true, n_classes)

    cols = st.columns(len(METRIC_COLUMNS))
    for col, metric in zip(cols, METRIC_COLUMNS):
        value = scores[metric]
        col.metric(metric, "n/a" if pd.isna(value) else f"{value:.4f}")
    if pd.isna(scores["AUC"]):
        st.caption("AUC needs every class to appear in the uploaded file.")

    y_pred = model.predict(X)
    left, right = st.columns(2)

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
        confusion_data = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
        confusion_data.index.name = "Actual / predicted"
        st.download_button(
            "Download confusion matrix",
            confusion_data.to_csv().encode("utf-8"),
            "dry_bean_confusion_matrix.csv",
            "text/csv",
        )

    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, y_pred, labels=range(n_classes),
            target_names=encoder.classes_, output_dict=True, zero_division=0,
        )
        report_data = pd.DataFrame(report).T.round(3)
        st.dataframe(report_data, use_container_width=True)
        st.download_button(
            "Download classification report",
            report_data.to_csv(index_label="Class").encode("utf-8"),
            "dry_bean_classification_report.csv",
            "text/csv",
        )
else:
    if len(selected_models) < 2:
        st.warning("Select at least two classifiers in the sidebar to compare their results.")
        st.stop()

    comparison = pd.DataFrame(
        [
            {"ML Model Name": name, **compute_metrics(models[name], X, y_true, n_classes)}
            for name in selected_models
        ]
    ).set_index("ML Model Name")

    st.subheader("Model comparison")
    st.caption(f"Comparing {len(selected_models)} classifiers on the same {len(data):,} uploaded test rows.")

    metric_winners = {}
    for metric in METRIC_COLUMNS:
        available_scores = comparison[metric].dropna()
        if not available_scores.empty:
            highest_score = available_scores.max()
            metric_winners[metric] = (
                available_scores[available_scores == highest_score].index.tolist(),
                highest_score,
            )

    lead_counts = {name: 0 for name in selected_models}
    for winners, _ in metric_winners.values():
        for winner in winners:
            lead_counts[winner] += 1
    most_metric_leads = max(lead_counts.values())
    overall_leaders = [name for name, count in lead_counts.items() if count == most_metric_leads]
    st.markdown("### Comparison findings")
    if len(overall_leaders) == 1:
        st.success(f"{overall_leaders[0]} leads on the most metrics ({most_metric_leads} of {len(metric_winners)} available metrics).")
    else:
        st.success(f"Overall metric lead is shared by {', '.join(overall_leaders)} ({most_metric_leads} metric leads each).")

    findings = []
    for metric in METRIC_COLUMNS:
        if metric in metric_winners:
            winners, highest_score = metric_winners[metric]
            findings.append(f"**{metric}:** {', '.join(winners)} ({highest_score:.4f})")
        else:
            findings.append(f"**{metric}:** unavailable; the upload must include every bean variety")
    st.markdown("  \n".join(findings))

    st.dataframe(comparison.round(4), use_container_width=True)
    comparison_col, download_col = st.columns([1, 1])
    with comparison_col:
        comparison_metric = st.selectbox("Chart metric", METRIC_COLUMNS, key="comparison_metric")
        st.bar_chart(comparison[[comparison_metric]].sort_values(comparison_metric))
    with download_col:
        st.markdown("**Export comparison report**")
        st.write("Download the selected models and their six evaluation metrics as a CSV.")
        st.download_button(
            "Download comparison report (CSV)",
            comparison.to_csv(index_label="ML Model Name").encode("utf-8"),
            "dry_bean_model_comparison.csv",
            "text/csv",
            type="primary",
        )
