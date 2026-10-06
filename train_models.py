import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

DATA_PATH = "professional_banking_fraud_risk_dataset.csv"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("=" * 70)

print(f"Dataset shape: {df.shape}")

print(
    f"Fraud transactions: "
    f"{df['fraud_flag'].sum():,}"
)

print(
    f"Fraud rate: "
    f"{df['fraud_flag'].mean() * 100:.2f}%"
)

print()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["transaction_timestamp"] = pd.to_datetime(
    df["transaction_timestamp"],
    errors="coerce"
)

df["transaction_hour"] = (
    df["transaction_timestamp"].dt.hour
)

df["transaction_day_of_week"] = (
    df["transaction_timestamp"].dt.dayofweek
)

df["is_weekend"] = (
    df["transaction_day_of_week"] >= 5
).astype(int)


# ============================================================
# REMOVE IDENTIFIERS AND POTENTIAL DATA LEAKAGE
# ============================================================

DROP_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "transaction_timestamp",
    "fraud_flag",
    "risk_score",
    "risk_level",
    "risk_action",
    "manual_review_flag"
]


X = df.drop(columns=DROP_COLUMNS)

y = df["fraud_flag"].astype(int)


# ============================================================
# IDENTIFY DATA TYPES
# ============================================================

categorical_cols = (
    X.select_dtypes(include=["object"])
    .columns
    .tolist()
)

numeric_cols = (
    X.select_dtypes(exclude=["object"])
    .columns
    .tolist()
)


print("Numeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print()


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_cols
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_cols
    )
])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print()


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# XGBOOST
# ============================================================

if XGBOOST_AVAILABLE:

    models["XGBoost"] = XGBClassifier(

        n_estimators=250,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.9,

        colsample_bytree=0.9,

        eval_metric="logloss",

        random_state=42,

        n_jobs=4
    )


# ============================================================
# TRAIN SUPERVISED MODELS
# ============================================================

results = []

trained_models = {}


for name, estimator in models.items():

    print(
        f"Training {name}..."
    )

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            estimator
        )

    ])


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]


    metrics = {

        "Model": name,

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_test,
                probabilities
            )
    }


    results.append(
        metrics
    )

    trained_models[name] = pipeline


# ============================================================
# ISOLATION FOREST
# ============================================================

print(
    "Training Isolation Forest..."
)


iso_preprocessor = ColumnTransformer([

    (
        "num",

        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",
                StandardScaler()
            )
        ]),

        numeric_cols
    ),

    (
        "cat",

        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]),

        categorical_cols
    )
])


X_train_iso = (
    iso_preprocessor.fit_transform(
        X_train
    )
)


X_test_iso = (
    iso_preprocessor.transform(
        X_test
    )
)


isolation_model = IsolationForest(

    n_estimators=300,

    contamination=float(
        y_train.mean()
    ),

    random_state=42,

    n_jobs=-1
)


isolation_model.fit(
    X_train_iso
)


iso_predictions_raw = (
    isolation_model.predict(
        X_test_iso
    )
)


iso_predictions = (
    iso_predictions_raw == -1
).astype(int)


iso_anomaly_score = (
    -isolation_model.decision_function(
        X_test_iso
    )
)


iso_result = {

    "Model":
        "Isolation Forest",

    "Accuracy":
        accuracy_score(
            y_test,
            iso_predictions
        ),

    "Precision":
        precision_score(
            y_test,
            iso_predictions,
            zero_division=0
        ),

    "Recall":
        recall_score(
            y_test,
            iso_predictions,
            zero_division=0
        ),

    "F1":
        f1_score(
            y_test,
            iso_predictions,
            zero_division=0
        ),

    "ROC_AUC":
        roc_auc_score(
            y_test,
            iso_anomaly_score
        )
}


results.append(
    iso_result
)


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "ROC_AUC",
    ascending=False
)


results_df = results_df.reset_index(
    drop=True
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# SELECT BEST SUPERVISED MODEL
# ============================================================

supervised_results = results_df[
    results_df["Model"]
    != "Isolation Forest"
]


best_model_name = (
    supervised_results.iloc[0]["Model"]
)


best_model = trained_models[
    best_model_name
]


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(

    {
        "model": best_model,

        "model_name":
            best_model_name,

        "feature_columns":
            X.columns.tolist(),

        "numeric_columns":
            numeric_cols,

        "categorical_columns":
            categorical_cols

    },

    os.path.join(
        MODEL_DIR,
        "best_model.pkl"
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_output = X_test.copy()


test_output["actual_fraud"] = (
    y_test.values
)


test_output["fraud_probability"] = (
    best_model
    .predict_proba(X_test)[:, 1]
)


test_output["predicted_fraud"] = (
    test_output[
        "fraud_probability"
    ] >= 0.50
).astype(int)


test_output.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),

    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "MODEL COMPARISON"
)

print("=" * 70)

print(
    results_df.round(4).to_string(
        index=False
    )
)

print()

print(
    f"BEST SUPERVISED MODEL: "
    f"{best_model_name}"
)

print()

print(
    "Model saved:"
)

print(
    "models/best_model.pkl"
)

print()

print(
    "Model comparison saved:"
)

print(
    "outputs/model_comparison.csv"
)

print()

print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)