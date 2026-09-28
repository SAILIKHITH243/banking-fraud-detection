# ============================================================
# BANKING FRAUD DETECTION & RISK ANALYTICS
# Complete End-to-End Machine Learning Project
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# MATPLOTLIB - NON INTERACTIVE MODE
# IMPORTANT: NO GRAPH WINDOWS WILL OPEN
# ------------------------------------------------------------

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# DATA SCIENCE LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# MACHINE LEARNING
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# PROJECT HEADER
# ============================================================

print("=" * 70)
print("BANKING FRAUD DETECTION & RISK ANALYTICS")
print("=" * 70)


# ============================================================
# 1. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

print("\nStep 1 - Project folders prepared")


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - LOADING DATASET")
print("=" * 70)

df = pd.read_csv("banking_transactions.csv")

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 3. BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - BASIC DATA ANALYSIS")
print("=" * 70)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nBasic Statistics:")
print(df.describe())

print("\nFraud Distribution:")
print(df["fraud_flag"].value_counts())

print("\nFraud Percentage:")
print(
    df["fraud_flag"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("STEP 4 - DATA CLEANING")
print("=" * 70)

# Remove duplicate rows
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates()

print("Duplicate rows removed:", duplicate_count)

# Remove rows where target is missing
target_missing = df["fraud_flag"].isnull().sum()

if target_missing > 0:
    df = df.dropna(subset=["fraud_flag"])

print("Missing target rows removed:", target_missing)

# Convert fraud flag to integer
df["fraud_flag"] = df["fraud_flag"].astype(int)

print("\nData cleaning completed.")
print("Final dataset shape:", df.shape)


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# 5.1 Fraud Distribution
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="fraud_flag"
)

plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Fraud Flag")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/fraud_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.2 Transaction Amount Distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="transaction_amount",
    hue="fraud_flag",
    bins=30,
    kde=True
)

plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/transaction_amount_distribution.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.3 Transaction Amount by Fraud
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="transaction_amount"
)

plt.title("Transaction Amount by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Transaction Amount")

plt.tight_layout()

plt.savefig(
    "outputs/plots/transaction_amount_by_fraud.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.4 Payment Channel
# ------------------------------------------------------------

payment_fraud = pd.crosstab(
    df["payment_channel"],
    df["fraud_flag"]
)

print("\nPayment Channel vs Fraud:")
print(payment_fraud)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="payment_channel",
    hue="fraud_flag"
)

plt.title("Fraud by Payment Channel")
plt.xlabel("Payment Channel")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "outputs/plots/payment_channel_fraud.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.5 Authentication Type
# ------------------------------------------------------------

authentication_fraud = pd.crosstab(
    df["authentication_type"],
    df["fraud_flag"]
)

print("\nAuthentication Type vs Fraud:")
print(authentication_fraud)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="authentication_type",
    hue="fraud_flag"
)

plt.title("Fraud by Authentication Type")
plt.xlabel("Authentication Type")
plt.ylabel("Number of Transactions")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "outputs/plots/authentication_type_fraud.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.6 Suspicious IP
# ------------------------------------------------------------

suspicious_ip = pd.crosstab(
    df["suspicious_ip_flag"],
    df["fraud_flag"],
    normalize="index"
) * 100

print("\nSuspicious IP Fraud Percentage:")
print(suspicious_ip)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="suspicious_ip_flag",
    hue="fraud_flag"
)

plt.title("Fraud vs Suspicious IP")
plt.xlabel("Suspicious IP Flag")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/suspicious_ip_fraud.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.7 International Transactions
# ------------------------------------------------------------

international = pd.crosstab(
    df["international_transaction_flag"],
    df["fraud_flag"],
    normalize="index"
) * 100

print("\nInternational Transaction Fraud Percentage:")
print(international)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="international_transaction_flag",
    hue="fraud_flag"
)

plt.title("Fraud vs International Transactions")
plt.xlabel("International Transaction Flag")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/international_transaction_fraud.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.8 Device Risk Score
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="device_risk_score"
)

plt.title("Device Risk Score by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Device Risk Score")

plt.tight_layout()

plt.savefig(
    "outputs/plots/device_risk_score.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.9 Anomaly Score
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="anomaly_score"
)

plt.title("Anomaly Score by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Anomaly Score")

plt.tight_layout()

plt.savefig(
    "outputs/plots/anomaly_score.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.10 Transaction Velocity
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="transaction_velocity_score"
)

plt.title("Transaction Velocity Score by Fraud Status")
plt.xlabel("Fraud Flag")
plt.ylabel("Velocity Score")

plt.tight_layout()

plt.savefig(
    "outputs/plots/transaction_velocity_score.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 5.11 Failed Transactions
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraud_flag",
    y="failed_transactions_last_30d"
)

plt.title("Failed Transactions in Last 30 Days")
plt.xlabel("Fraud Flag")
plt.ylabel("Failed Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/failed_transactions.png",
    dpi=150
)

plt.close()


# ============================================================
# 6. CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - CORRELATION ANALYSIS")
print("=" * 70)

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_df.corr()

print("\nCorrelation with fraud_flag:")

print(
    correlation["fraud_flag"]
    .sort_values(ascending=False)
)

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/plots/correlation_heatmap.png",
    dpi=150
)

plt.close()


# ============================================================
# 7. PREPARE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 - PREPARING FEATURES AND TARGET")
print("=" * 70)

X = df.drop(
    columns=["fraud_flag"]
)

y = df["fraud_flag"]

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ============================================================
# 8. IDENTIFY COLUMN TYPES
# ============================================================

print("\n" + "=" * 70)
print("STEP 8 - IDENTIFYING COLUMN TYPES")
print("=" * 70)

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nCategorical Features:")
print(categorical_features)

print("\nNumeric Features:")
print(numeric_features)


# ============================================================
# 9. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 - TRAIN TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 10. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("STEP 10 - DATA PREPROCESSING")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

print("Preprocessing pipeline created.")


# ============================================================
# 11. RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 - RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

print("Random Forest model created.")


# ============================================================
# 12. MODEL TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 - MODEL TRAINING")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully!")


# ============================================================
# 13. PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 - MAKING PREDICTIONS")
print("=" * 70)

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

print("\nPredictions completed.")


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 - MODEL EVALUATION")
print("=" * 70)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nModel Performance:")
print("-" * 50)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 - CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Fraud",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("STEP 16 - CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Non-Fraud",
        "Fraud"
    ],
    yticklabels=[
        "Non-Fraud",
        "Fraud"
    ]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/plots/confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# 17. MODEL METRICS CSV
# ============================================================

print("\n" + "=" * 70)
print("STEP 17 - SAVING MODEL METRICS")
print("=" * 70)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_df.to_csv(
    "outputs/model_metrics.csv",
    index=False
)

print(
    "Metrics saved to: outputs/model_metrics.csv"
)


# ============================================================
# 18. RISK SCORING
# ============================================================

print("\n" + "=" * 70)
print("STEP 18 - TRANSACTION RISK SCORING")
print("=" * 70)

risk_results = X_test.copy()

risk_results["actual_fraud"] = y_test.values

risk_results["predicted_fraud"] = y_pred

risk_results["fraud_probability"] = y_probability

risk_results["risk_score"] = (
    y_probability * 100
)


# ------------------------------------------------------------
# Risk Categories
# ------------------------------------------------------------

def assign_risk_category(score):

    if score >= 70:
        return "High Risk"

    elif score >= 40:
        return "Medium Risk"

    else:
        return "Low Risk"


risk_results["risk_category"] = (
    risk_results["risk_score"]
    .apply(assign_risk_category)
)


# ============================================================
# 19. DISPLAY RISK RESULTS
# ============================================================

print("\nSample Risk Results:")

print(
    risk_results[
        [
            "transaction_id",
            "transaction_amount",
            "fraud_probability",
            "risk_score",
            "risk_category"
        ]
    ].head(10)
)


print("\nRisk Category Distribution:")

print(
    risk_results[
        "risk_category"
    ].value_counts()
)


# ============================================================
# 20. RISK DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=risk_results,
    x="risk_category",
    order=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

plt.title("Transaction Risk Category Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/risk_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 21. RISK SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=risk_results,
    x="risk_score",
    bins=30,
    kde=True
)

plt.title("Transaction Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "outputs/plots/risk_score_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# 22. SAVE RISK RESULTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 22 - SAVING RISK RESULTS")
print("=" * 70)

risk_results.to_csv(
    "outputs/transaction_risk_scores.csv",
    index=False
)

print(
    "Risk results saved to: outputs/transaction_risk_scores.csv"
)


# ============================================================
# 23. SAVE MODEL
# ============================================================

print("\n" + "=" * 70)
print("STEP 23 - SAVING MACHINE LEARNING MODEL")
print("=" * 70)

joblib.dump(
    pipeline,
    "models/fraud_detection_model.pkl"
)

print(
    "Model saved to: models/fraud_detection_model.pkl"
)


# ============================================================
# 24. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(
    "\nTotal Transactions:",
    len(df)
)

print(
    "Training Transactions:",
    len(X_train)
)

print(
    "Testing Transactions:",
    len(X_test)
)

print(
    "Fraud Transactions:",
    int(df["fraud_flag"].sum())
)

print(
    "Non-Fraud Transactions:",
    int((df["fraud_flag"] == 0).sum())
)

print(
    f"Model Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Model Precision: {precision * 100:.2f}%"
)

print(
    f"Model Recall: {recall * 100:.2f}%"
)

print(
    f"Model F1 Score: {f1 * 100:.2f}%"
)

print(
    f"Model ROC-AUC: {roc_auc:.4f}"
)


# ============================================================
# 25. OUTPUT FILES
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "\nModel:"
)

print(
    "models/fraud_detection_model.pkl"
)

print(
    "\nMetrics:"
)

print(
    "outputs/model_metrics.csv"
)

print(
    "\nRisk Results:"
)

print(
    "outputs/transaction_risk_scores.csv"
)

print(
    "\nGraphs:"
)

print(
    "outputs/plots/"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "BANKING FRAUD DETECTION PROJECT COMPLETED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nAll analysis, machine learning, evaluation, "
    "risk scoring and model saving steps are completed."
)

print(
    "\nReady for the next stage: STREAMLIT DASHBOARD"
)

print("=" * 70)