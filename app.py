import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Banking Fraud Detection & Risk Analytics")
st.write(
    "Machine Learning based banking transaction fraud detection "
    "and risk analytics dashboard."
)

st.markdown("---")


# ============================================================
# LOAD DATASET
# ============================================================

DATA_FILE = "banking_transactions.csv"

if not os.path.exists(DATA_FILE):

    st.error("❌ banking_transactions.csv not found!")

    st.stop()

df = pd.read_csv(DATA_FILE)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_FILE = "models/fraud_detection_model.pkl"

model = None

if os.path.exists(MODEL_FILE):

    try:

        model = joblib.load(MODEL_FILE)

    except Exception as e:

        st.warning(
            f"⚠️ Model could not be loaded: {e}"
        )

else:

    st.warning(
        "⚠️ Fraud detection model file not found."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Dashboard Menu")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Transaction Analysis",
        "Fraud Analysis",
        "Risk Analysis",
        "Dataset"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Banking Fraud Detection Dashboard")

    total_transactions = len(df)

    fraud_count = int(
        df["fraud_flag"].sum()
    )

    genuine_count = (
        total_transactions - fraud_count
    )

    fraud_percentage = (
        fraud_count / total_transactions
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:

        st.metric(
            "Fraud Transactions",
            f"{fraud_count:,}"
        )

    with col3:

        st.metric(
            "Genuine Transactions",
            f"{genuine_count:,}"
        )

    with col4:

        st.metric(
            "Fraud Percentage",
            f"{fraud_percentage:.2f}%"
        )

    st.markdown("---")

    st.subheader("Fraud vs Non-Fraud")

    fraud_chart = (
        df["fraud_flag"]
        .value_counts()
        .rename(
            {
                False: "Genuine",
                True: "Fraud"
            }
        )
    )

    st.bar_chart(fraud_chart)

    st.markdown("---")

    st.subheader("Key Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            f"Rows: {df.shape[0]:,}"
        )

    with col2:

        st.info(
            f"Columns: {df.shape[1]}"
        )

    with col3:

        st.info(
            f"Missing Values: {df.isnull().sum().sum()}"
        )


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

elif page == "Transaction Analysis":

    st.header("💳 Transaction Analysis")

    st.subheader("Transaction Amount Distribution")

    st.bar_chart(
        df["transaction_amount"]
        .value_counts()
        .sort_index()
        .head(50)
    )

    st.subheader("Transaction Amount Statistics")

    st.dataframe(
        df["transaction_amount"]
        .describe()
        .to_frame()
    )

    st.markdown("---")

    st.subheader("Payment Channel")

    payment_counts = (
        df["payment_channel"]
        .value_counts()
    )

    st.bar_chart(payment_counts)

    st.markdown("---")

    st.subheader("Authentication Type")

    auth_counts = (
        df["authentication_type"]
        .value_counts()
    )

    st.bar_chart(auth_counts)


# ============================================================
# FRAUD ANALYSIS
# ============================================================

elif page == "Fraud Analysis":

    st.header("🚨 Fraud Analysis")

    fraud_df = df[
        df["fraud_flag"] == True
    ]

    genuine_df = df[
        df["fraud_flag"] == False
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Transactions",
            len(fraud_df)
        )

    with col2:

        st.metric(
            "Genuine Transactions",
            len(genuine_df)
        )

    st.markdown("---")

    st.subheader("Fraud by Payment Channel")

    fraud_payment = pd.crosstab(
        df["payment_channel"],
        df["fraud_flag"]
    )

    fraud_payment.columns = [
        "Genuine",
        "Fraud"
    ]

    st.bar_chart(fraud_payment)

    st.markdown("---")

    st.subheader("Fraud by Authentication Type")

    fraud_auth = pd.crosstab(
        df["authentication_type"],
        df["fraud_flag"]
    )

    fraud_auth.columns = [
        "Genuine",
        "Fraud"
    ]

    st.bar_chart(fraud_auth)

    st.markdown("---")

    st.subheader("Suspicious IP Analysis")

    suspicious_ip = pd.crosstab(
        df["suspicious_ip_flag"],
        df["fraud_flag"]
    )

    suspicious_ip.columns = [
        "Genuine",
        "Fraud"
    ]

    st.bar_chart(suspicious_ip)

    st.markdown("---")

    st.subheader("International Transaction Analysis")

    international = pd.crosstab(
        df["international_transaction_flag"],
        df["fraud_flag"]
    )

    international.columns = [
        "Genuine",
        "Fraud"
    ]

    st.bar_chart(international)


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.header("⚠️ Transaction Risk Analysis")

    risk_file = (
        "outputs/transaction_risk_scores.csv"
    )

    if os.path.exists(risk_file):

        risk_df = pd.read_csv(
            risk_file
        )

        st.success(
            "✅ Risk scoring results loaded successfully."
        )

        st.subheader(
            "Risk Category Distribution"
        )

        if "risk_category" in risk_df.columns:

            risk_counts = (
                risk_df["risk_category"]
                .value_counts()
            )

            st.bar_chart(
                risk_counts
            )

            st.markdown("---")

            st.dataframe(
                risk_df,
                use_container_width=True
            )

        else:

            st.warning(
                "Risk category column not found."
            )

    else:

        st.warning(
            "⚠️ Risk results file not found."
        )

        st.info(
            "Run banking.py first to generate risk scoring results."
        )


# ============================================================
# DATASET
# ============================================================

elif page == "Dataset":

    st.header("📁 Banking Transaction Dataset")

    st.write(
        f"Dataset contains **{df.shape[0]:,} transactions** "
        f"and **{df.shape[1]} columns**."
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Dataset Information")

    info_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
            "Missing Values": [
                df[col].isnull().sum()
                for col in df.columns
            ]
        }
    )

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Basic Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Banking Fraud Detection & Risk Analytics | "
    "Machine Learning Project"
)