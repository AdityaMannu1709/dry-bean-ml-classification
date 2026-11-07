# Dry Bean Classification: ML Assignment 2 (M.Tech AIML)

## a. Problem statement

Dry beans of seven varieties look alike to the eye, and sorting them by hand is slow and error-prone. This project builds a multi-class classifier that predicts the bean variety from 16 shape measurements taken from images of each grain. Five classification algorithms are trained on the same data, compared on six evaluation metrics, and exposed through an interactive Streamlit app.

## b. Dataset description

- **Source:** Dry Bean Dataset, UCI Machine Learning Repository (Koklu and Ozkan, 2020): https://archive.ics.uci.edu/dataset/602/dry+bean+dataset.
- **Task:** Multi-class classification with 7 classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA.
- **Instances:** 13,611 raw rows. 68 exact duplicates were removed, leaving 13,543.
- **Features:** 16, all numeric and with no missing values: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.
- **Target column:** `Class`.
- **Class balance:** Imbalanced. DERMASON is the largest class (about 26%) and BOMBAY the smallest (about 4%).
- **Split:** Stratified 80/20 train/test split with `random_state=42`. The 2,709 test rows are saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` is applied inside the pipelines for Logistic Regression, KNN and SVM, because their features sit on very different scales. Decision Tree and Naive Bayes use the raw features. Class names are encoded with a saved `LabelEncoder`.

## c. GitHub repository link

Repository: https://github.com/AdityaMannu1709/dry-bean-ml-classification/tree/main

Live Streamlit app: https://dry-bean-ml-classification-apwtblb4d2ckjcbhtqtcap.streamlit.app/

### Repository structure

```
├── app.py                  Streamlit app
├── train.py                Trains and saves the five models, writes test_data.csv
├── evaluate.py             Shared metric computation
├── requirements.txt
├── README.md
├── test_data.csv           Held-out test data (features + Class)
├── data/raw/DryBeanDataset/
│   ├── Dry_Bean_Dataset.xlsx
│   ├── Dry_Bean_Dataset.arff
│   └── Dry_Bean_Dataset.txt
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── label_encoder.joblib
    └── metrics.csv
```

### Run locally

```bash
pip install -r requirements.txt
python train.py          # optional, models are already saved in model/
streamlit run app.py
```

`train.py` reads `data/raw/DryBeanDataset/Dry_Bean_Dataset.xlsx` to reproduce the training split and saved artifacts. The raw dataset is included in the repository.

## d. Models used

All five models were trained on the same training split and evaluated on the same held-out test split. Precision, Recall and F1 use weighted averaging, and AUC is one-vs-rest weighted.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.9011 | 0.9543 | 0.9013 | 0.9011 | 0.9012 | 0.8804 |
| KNN | 0.9118 | 0.9832 | 0.9123 | 0.9118 | 0.9119 | 0.8933 |
| Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| Support Vector Machine | 0.9240 | 0.9881 | 0.9241 | 0.9240 | 0.9240 | 0.9080 |

Hyperparameters: Logistic Regression `max_iter=2000`; Decision Tree `max_depth=12`; KNN `n_neighbors=7`; Gaussian Naive Bayes defaults; SVM RBF kernel with `C=10`, wrapped in `CalibratedClassifierCV` (5-fold) to produce probabilities for AUC.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy of 0.919 and the highest AUC (0.993). The classes are close to linearly separable once the features are scaled, and the model is fast and compact. Its weakest class is SIRA (F1 0.871); its most frequent error is DERMASON predicted as SIRA (53 cases). |
| Decision Tree | The lowest accuracy among the well-performing models (0.901) and a clearly lower AUC (0.954). Hard splits give coarse probabilities, which limits ranking quality. It misclassifies DERMASON as SIRA most often (63 cases) and has the lowest SIRA F1 of the competitive models (0.834). |
| KNN | Accuracy of 0.912 and a strong AUC (0.983) after scaling. It is competitive but slightly behind SVM and Logistic Regression, and it stores the full training set, so predictions are slower. Apart from BOMBAY, it does best on HOROZ (F1 0.948) and worst on SIRA (F1 0.857), where it most often confuses SIRA with DERMASON (55 cases). |
| Naive Bayes | The weakest model by a wide margin (accuracy 0.763, MCC 0.714). The shape features are strongly correlated (for example Area, ConvexArea and EquivDiameter), which breaks the independence assumption. It struggles most on BARBUNYA (F1 0.548), which it often labels as CALI (108 cases). Its AUC (0.964) is much better than its accuracy suggests, so its ranking is reasonable but its hard decisions are poor. |
| Support Vector Machine | The best accuracy (0.924), F1 (0.924) and MCC (0.908), and the highest F1 on most classes. The RBF kernel captures the non-linear boundary between the similar classes SIRA, DERMASON and BARBUNYA. Its AUC (0.988) is slightly below Logistic Regression. |
| **Overall winner for this dataset** | **Support Vector Machine.** It leads on Accuracy, Precision, Recall, F1 and MCC. Logistic Regression is a very close second and has the best AUC with a much simpler model. All models classify BOMBAY perfectly (F1 1.0); the remaining errors are mostly between the visually similar DERMASON and SIRA beans. |

## Streamlit app features

- Upload a labeled test CSV containing the 16 features and the `Class` target; the app validates required columns and bean labels before evaluating.
- Review the uploaded file name, row and feature counts, first 10 rows, and class distribution.
- Download the provided sample test data from the sidebar.
- Choose **Single model** to select one classifier and view its Accuracy, AUC, Precision, Recall, F1, MCC, confusion matrix, and classification report.
- Choose **Compare models** to select between two and five classifiers and compare their metrics on the same uploaded rows.
- Review automatically generated comparison findings, including the leaders for each available metric.
- Download the selected-model classification report, confusion matrix, and model-comparison metrics as CSV files.

The app loads the saved models from `model/` and does not retrain on uploaded data. AUC is shown as unavailable when the uploaded test data does not contain every bean variety.
