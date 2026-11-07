import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def compute_metrics(model, X, y_true, n_classes):
    """Return the six assignment metrics; multi-class scores use weighted averaging, AUC is one-vs-rest."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # One-vs-rest AUC is undefined if an uploaded file lacks some classes.
    present = np.unique(y_true)
    if len(present) < n_classes:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
