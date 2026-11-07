from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from evaluate import METRIC_COLUMNS, compute_metrics

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "raw" / "DryBeanDataset" / "Dry_Bean_Dataset.xlsx"
MODEL_DIR = ROOT / "model"
TARGET = "Class"
SEED = 42

MODELS = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="rbf", C=10), ensemble=False, cv=5),
    ),
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_excel(RAW_FILE).drop_duplicates().reset_index(drop=True)
    print(f"Rows after dropping duplicates: {len(df)}, features: {df.shape[1] - 1}")

    encoder = LabelEncoder().fit(df[TARGET])
    X = df.drop(columns=TARGET)
    y = encoder.transform(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # test_data.csv keeps the readable class names so users can inspect or upload it directly.
    test_df = X_test.copy()
    test_df[TARGET] = encoder.inverse_transform(y_test)
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

    rows = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        scores = compute_metrics(model, X_test, y_test, len(encoder.classes_))
        rows.append({"ML Model Name": name, **scores})
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    results = pd.DataFrame(rows, columns=["ML Model Name", *METRIC_COLUMNS]).round(4)
    results.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print("\n" + results.to_string(index=False))


if __name__ == "__main__":
    main()
