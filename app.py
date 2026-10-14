
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Banking Fraud & Risk Analytics",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PROFESSIONAL LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8FAFC;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    section[data-testid="stSidebar"] * {
        color: #0F172A;
    }

    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 28px;
    }

    h1 {
        color: #0F172A;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #1E293B;
        font-weight: 650;
    }

    h3 {
        color: #334155;
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 15px 18px;
        min-height: 110px;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-weight: 700;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #CBD5E1;
        border-radius: 8px;
    }

    div[data-testid="stSlider"] {
        padding-top: 5px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #E2E8F0;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        background-color: #FFFFFF;
        color: #0F172A;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #2563EB;
        color: #2563EB;
    }

    div[role="radiogroup"] label {
        padding: 7px 5px;
        border-radius: 7px;
    }

    .stCaption {
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "professional_banking_fraud_risk_dataset.csv"
MODEL_PATH = "models/best_model.pkl"
COMPARISON_PATH = "outputs/model_comparison.csv"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

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

    return df


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Place professional_banking_fraud_risk_dataset.csv "
        "in the same folder as app.py."
    )
    st.stop()

df = load_data()


# ============================================================
# CREATE ANALYTICS RISK LEVEL
# ============================================================

def analytics_risk_level(score):
    if score <= 15:
        return "LOW"
    elif score <= 25:
        return "MEDIUM"
    elif score <= 35:
        return "HIGH"
    else:
        return "CRITICAL"


df["analytics_risk_level"] = (
    df["risk_score"].apply(analytics_risk_level)
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-title">
        🛡️ Banking Fraud Detection & Risk Analytics
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="dashboard-subtitle">
        Machine Learning platform for fraud detection,
        transaction investigation and risk-based decision support.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found.")
    st.info("Run this command first: py train_models.py")
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

model_bundle = load_model()
model = model_bundle["model"]
best_model_name = model_bundle["model_name"]


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk(fraud_probability, transaction):
    security_points = 0

    if "new_device_flag" in transaction.index:
        security_points += (
            int(transaction["new_device_flag"]) * 8
        )

    if "new_beneficiary_flag" in transaction.index:
        security_points += (
            int(transaction["new_beneficiary_flag"]) * 8
        )

    if "international_transaction_flag" in transaction.index:
        security_points += (
            int(transaction["international_transaction_flag"]) * 8
        )

    if "suspicious_ip_flag" in transaction.index:
        security_points += (
            int(transaction["suspicious_ip_flag"]) * 8
        )

    if "chargeback_history" in transaction.index:
        security_points += (
            int(transaction["chargeback_history"]) * 8
        )

    if "device_risk_score" in transaction.index:
        if float(transaction["device_risk_score"]) >= 70:
            security_points += 10

    if "merchant_risk_score" in transaction.index:
        if float(transaction["merchant_risk_score"]) >= 70:
            security_points += 8

    risk_score = min(
        100,
        (fraud_probability * 80) + security_points
    )

    if risk_score <= 30:
        risk_level = "LOW"
        action = "APPROVE"
    elif risk_score <= 60:
        risk_level = "MEDIUM"
        action = "MONITOR"
    elif risk_score <= 80:
        risk_level = "HIGH"
        action = "STEP-UP AUTHENTICATION"
    else:
        risk_level = "CRITICAL"
        action = "BLOCK + INVESTIGATE"

    return round(risk_score, 2), risk_level, action


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🏦 Banking Analytics")

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Dashboard",
        "Transaction Investigation",
        "Fraud Analysis",
        "Risk Intelligence",
        "Model Lab",
        "Fraud Monitoring",
        "Statement Analyzer"
    ]
)


# ============================================================
# 1. EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header("📊 Executive Dashboard")

    total_transactions = len(df)
    fraud_transactions = int(df["fraud_flag"].sum())

    fraud_rate = (
        fraud_transactions / total_transactions * 100
        if total_transactions else 0
    )

    high_critical = int(
        df["analytics_risk_level"].isin(
            ["HIGH", "CRITICAL"]
        ).sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Fraud Detected",
        f"{fraud_transactions:,}"
    )

    col3.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    col4.metric(
        "High / Critical Risk",
        f"{high_critical:,}"
    )

    st.divider()

    st.subheader("📈 Daily Transaction Trend")

    trend = (
        df.dropna(subset=["transaction_timestamp"])
        .set_index("transaction_timestamp")
        .resample("D")
        .agg(
            transactions=("transaction_id", "count"),
            fraud=("fraud_flag", "sum")
        )
        .reset_index()
    )

    fig = px.line(
        trend,
        x="transaction_timestamp",
        y=["transactions", "fraud"],
        markers=True,
        title="Daily Transactions vs Fraud"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        risk_counts = (
            df["analytics_risk_level"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = ["risk_level", "count"]

        fig = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            title="Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        channel = (
            df.groupby("payment_channel")["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )

        channel.columns = ["payment_channel", "fraud_rate"]

        fig = px.bar(
            channel,
            x="payment_channel",
            y="fraud_rate",
            title="Fraud Rate by Payment Channel"
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. TRANSACTION INVESTIGATION
# ============================================================

elif page == "Transaction Investigation":

    st.header("🔎 Transaction Investigation")

    st.caption(
        "Investigate an individual transaction using ML probability "
        "and behavioral/security signals."
    )

    selected_transaction = st.selectbox(
        "Select Transaction",
        df["transaction_id"].astype(str).tolist()
    )

    transaction = df[
        df["transaction_id"].astype(str) == selected_transaction
    ].iloc[0]

    excluded_columns = [
        "transaction_id",
        "customer_id",
        "merchant_id",
        "device_id",
        "fraud_flag",
        "risk_score",
        "risk_level",
        "risk_action",
        "manual_review_flag",
        "transaction_timestamp",
        "analytics_risk_level"
    ]

    model_transaction = transaction.drop(
        labels=excluded_columns,
        errors="ignore"
    ).copy()

    fraud_probability = float(
        model.predict_proba(
            pd.DataFrame([model_transaction])
        )[0, 1]
    )

    calculated_risk_score, calculated_risk_level, action = (
        calculate_risk(fraud_probability, transaction)
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ML Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

    col2.metric(
        "Risk Score",
        f"{calculated_risk_score}/100"
    )

    col3.metric("Risk Level", calculated_risk_level)
    col4.metric("Recommended Action", action)

    st.divider()
    st.subheader("💳 Transaction Details")

    transaction_columns = [
        "transaction_id",
        "customer_id",
        "transaction_timestamp",
        "merchant_category",
        "customer_segment",
        "payment_channel",
        "authentication_type",
        "transaction_amount"
    ]

    st.dataframe(
        transaction[transaction_columns].to_frame("Value"),
        use_container_width=True
    )

    st.subheader("🔐 Behavior & Security Signals")

    signal_columns = [
        "transaction_velocity_1h",
        "transaction_velocity_24h",
        "failed_transactions_24h",
        "login_attempts_24h",
        "device_risk_score",
        "anomaly_score",
        "geo_distance_km",
        "new_device_flag",
        "new_beneficiary_flag",
        "international_transaction_flag",
        "suspicious_ip_flag",
        "chargeback_history",
        "merchant_risk_score"
    ]

    available_columns = [
        column for column in signal_columns
        if column in transaction.index
    ]

    st.dataframe(
        transaction[available_columns].to_frame("Value"),
        use_container_width=True
    )

    dataset_label = (
        "FRAUD" if transaction["fraud_flag"] == 1
        else "GENUINE"
    )

    if dataset_label == "FRAUD":
        st.error(f"Dataset Label: {dataset_label}")
    else:
        st.success(f"Dataset Label: {dataset_label}")

    st.info(f"Recommended Action: {action}")


# ============================================================
# 3. FRAUD ANALYSIS
# ============================================================

elif page == "Fraud Analysis":

    st.header("📊 Fraud Analysis")

    st.caption(
        "Analyze fraud patterns across transaction and customer dimensions."
    )

    col1, col2 = st.columns(2)

    with col1:
        category = (
            df.groupby("merchant_category")["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )

        category.columns = ["merchant_category", "fraud_rate"]

        fig = px.bar(
            category,
            x="merchant_category",
            y="fraud_rate",
            title="Fraud Rate by Merchant Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        device = (
            df.groupby("device_type")["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )

        device.columns = ["device_type", "fraud_rate"]

        fig = px.bar(
            device,
            x="device_type",
            y="fraud_rate",
            title="Fraud Rate by Device Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        authentication = (
            df.groupby("authentication_type")["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )

        authentication.columns = [
            "authentication_type", "fraud_rate"
        ]

        fig = px.bar(
            authentication,
            x="authentication_type",
            y="fraud_rate",
            title="Fraud Rate by Authentication"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        segment = (
            df.groupby("customer_segment")["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .reset_index()
        )

        segment.columns = ["customer_segment", "fraud_rate"]

        fig = px.bar(
            segment,
            x="customer_segment",
            y="fraud_rate",
            title="Fraud Rate by Customer Segment"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💰 Transaction Amount Analysis")

    amount_data = df.copy()

    amount_data["Transaction Type"] = (
        amount_data["fraud_flag"].map({
            0: "Genuine",
            1: "Fraud"
        })
    )

    fig = px.box(
        amount_data,
        x="Transaction Type",
        y="transaction_amount",
        points=False,
        title="Transaction Amount: Fraud vs Genuine"
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 4. RISK INTELLIGENCE
# ============================================================

elif page == "Risk Intelligence":

    st.header("⚠️ Risk Intelligence")
    st.caption(
        "Prioritize transactions that require additional review."
    )

    low_count = int((df["analytics_risk_level"] == "LOW").sum())
    medium_count = int((df["analytics_risk_level"] == "MEDIUM").sum())
    high_count = int((df["analytics_risk_level"] == "HIGH").sum())
    critical_count = int(
        (df["analytics_risk_level"] == "CRITICAL").sum()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Low Risk", f"{low_count:,}")
    col2.metric("Medium Risk", f"{medium_count:,}")
    col3.metric("High Risk", f"{high_count:,}")
    col4.metric("Critical Risk", f"{critical_count:,}")

    st.divider()
    st.subheader("📊 Risk Distribution")

    risk_distribution = (
        df["analytics_risk_level"]
        .value_counts()
        .reindex(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            fill_value=0
        )
        .reset_index()
    )

    risk_distribution.columns = ["Risk Level", "Transactions"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Transactions",
            title="Transactions by Risk Level"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(
            risk_distribution,
            names="Risk Level",
            values="Transactions",
            title="Risk Level Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚨 High & Critical Risk Investigation Queue")

    high_risk = df[
        df["analytics_risk_level"].isin(["HIGH", "CRITICAL"])
    ].copy()

    display_columns = [
        "transaction_id",
        "customer_id",
        "transaction_amount",
        "risk_score",
        "analytics_risk_level",
        "risk_action",
        "device_risk_score",
        "merchant_risk_score",
        "suspicious_ip_flag",
        "new_device_flag",
        "international_transaction_flag",
        "chargeback_history",
        "fraud_flag"
    ]

    high_risk = high_risk.sort_values(
        "risk_score",
        ascending=False
    )

    if len(high_risk) > 0:
        available = [
            col for col in display_columns if col in high_risk.columns
        ]
        st.dataframe(
            high_risk[available].head(100),
            use_container_width=True
        )
    else:
        st.success("No high or critical risk transactions found.")

    st.subheader("📋 Risk Summary")

    risk_summary = (
        df.groupby("analytics_risk_level")
        .agg(
            transactions=("transaction_id", "count"),
            fraud_transactions=("fraud_flag", "sum"),
            average_risk_score=("risk_score", "mean"),
            average_transaction_amount=("transaction_amount", "mean")
        )
        .reset_index()
    )

    risk_summary["fraud_rate"] = (
        risk_summary["fraud_transactions"]
        / risk_summary["transactions"].replace(0, np.nan)
        * 100
    )

    risk_summary = risk_summary[
        [
            "analytics_risk_level",
            "transactions",
            "fraud_transactions",
            "fraud_rate",
            "average_risk_score",
            "average_transaction_amount"
        ]
    ]

    risk_summary.columns = [
        "Risk Level",
        "Transactions",
        "Fraud Transactions",
        "Fraud Rate (%)",
        "Average Risk Score",
        "Average Transaction Amount"
    ]

    st.dataframe(risk_summary, use_container_width=True)


# ============================================================
# 5. MODEL LAB
# ============================================================

elif page == "Model Lab":

    st.header("🧪 Model Lab")
    st.caption(
        "Compare machine learning models using standard classification metrics."
    )

    if os.path.exists(COMPARISON_PATH):

        comparison = pd.read_csv(COMPARISON_PATH)

        st.subheader("Machine Learning Model Comparison")

        formatters = {
            column: "{:.2%}"
            for column in [
                "Accuracy", "Precision", "Recall", "F1"
            ]
            if column in comparison.columns
        }

        if "ROC_AUC" in comparison.columns:
            formatters["ROC_AUC"] = "{:.4f}"

        st.dataframe(
            comparison.style.format(formatters),
            use_container_width=True
        )

        metrics = [
            column for column in [
                "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"
            ]
            if column in comparison.columns
        ]

        if metrics and "Model" in comparison.columns:
            metric = st.selectbox("Select Metric", metrics)

            sorted_comparison = comparison.sort_values(
                metric,
                ascending=False
            )

            fig = px.bar(
                sorted_comparison,
                x="Model",
                y=metric,
                title=f"Model Comparison — {metric}"
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("model_comparison.csv not found.")
        st.info("Run py train_models.py first.")

    st.success(f"Best Supervised Model: {best_model_name}")


# ============================================================
# 6. FRAUD MONITORING
# ============================================================

elif page == "Fraud Monitoring":

    st.header("🚨 Fraud Monitoring")

    st.caption(
        "Review recent transactions and quickly filter transactions "
        "based on their risk level."
    )

    col1, col2 = st.columns(2)

    with col1:
        sample_size = st.slider(
            "Transactions to Monitor",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )

    with col2:
        selected_risk = st.selectbox(
            "Filter by Risk Level",
            ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        )

    monitored = (
        df.sort_values("transaction_timestamp")
        .tail(sample_size)
        .copy()
    )

    monitored["monitor_status"] = np.where(
        monitored["analytics_risk_level"].isin(["HIGH", "CRITICAL"]),
        "REVIEW",
        "NORMAL"
    )

    if selected_risk != "ALL":
        monitored = monitored[
            monitored["analytics_risk_level"] == selected_risk
        ].copy()

    monitored_count = len(monitored)

    review_count = int(
        (monitored["monitor_status"] == "REVIEW").sum()
    )

    fraud_count = int(monitored["fraud_flag"].sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Transactions Shown", f"{monitored_count:,}")
    col2.metric("Review Required", f"{review_count:,}")
    col3.metric("Fraud in Selection", f"{fraud_count:,}")

    st.divider()

    if selected_risk == "ALL":
        st.info(f"Showing the latest {sample_size} transactions.")
    else:
        st.info(
            f"Showing {selected_risk} risk transactions "
            f"from the latest {sample_size} transactions."
        )

    st.subheader("📡 Transaction Monitoring Queue")

    monitoring_columns = [
        "transaction_id",
        "transaction_timestamp",
        "transaction_amount",
        "risk_score",
        "analytics_risk_level",
        "risk_action",
        "monitor_status"
    ]

    if len(monitored) > 0:
        available = [
            col for col in monitoring_columns if col in monitored.columns
        ]

        st.dataframe(
            monitored[available].sort_values(
                "risk_score", ascending=False
            ),
            use_container_width=True
        )
    else:
        st.warning("No transactions found for the selected risk level.")

    review_transactions = monitored[
        monitored["monitor_status"] == "REVIEW"
    ].copy()

    if len(review_transactions) > 0:
        st.subheader("🔍 Transactions Requiring Review")

        review_columns = [
            "transaction_id",
            "customer_id",
            "transaction_amount",
            "risk_score",
            "analytics_risk_level",
            "risk_action",
            "suspicious_ip_flag",
            "new_device_flag",
            "international_transaction_flag",
            "chargeback_history"
        ]

        available = [
            col for col in review_columns
            if col in review_transactions.columns
        ]

        st.dataframe(
            review_transactions[available].sort_values(
                "risk_score", ascending=False
            ),
            use_container_width=True
        )
    elif selected_risk in ["HIGH", "CRITICAL"]:
        st.success(
            "No transactions requiring review were found in this selection."
        )



# ============================================================
# 7. STATEMENT ANALYZER
# ============================================================

elif page == "Statement Analyzer":

    st.header("📄 Statement Analyzer")

    st.caption(
        "Analyze supported bank and payment-app statements. "
        "Supports PDF, CSV, TXT, Excel and statement images."
    )

    st.info(
        "This tool does not connect directly to your bank. "
        "OCR or unfamiliar statement layouts can cause extraction errors. "
        "Review transactions against the original statement. "
        "A flagged transaction is not proof of fraud."
    )

    try:
        from statement_analyzer import analyze_statement
    except Exception as import_error:
        st.error(
            "Could not load statement_analyzer.py. "
            "Ensure it is in the same folder as app.py and install "
            "the required packages. Details: "
            f"{import_error}"
        )
        st.stop()

    uploaded_statement = st.file_uploader(
        "Upload your statement",
        type=[
            "pdf", "csv", "txt", "xlsx", "xls",
            "png", "jpg", "jpeg", "webp", "tif", "tiff"
        ],
        key="bank_statement_uploader"
    )

    pdf_password = ""

    if (
        uploaded_statement is not None
        and uploaded_statement.name.lower().endswith(".pdf")
    ):
        pdf_password = st.text_input(
            "PDF password (leave blank if not required)",
            type="password",
            key="bank_statement_pdf_password"
        )

    if uploaded_statement is not None:

        st.caption(
            f"Selected file: {uploaded_statement.name} · "
            f"{uploaded_statement.size / 1024:,.1f} KB"
        )

        if st.button(
            "Analyze Statement",
            type="primary",
            key="analyze_bank_statement"
        ):

            try:
                with st.spinner(
                    "Reading statement and preparing reports..."
                ):
                    # Process the uploaded file in memory.
                    report = analyze_statement(
                        uploaded_statement,
                        password=pdf_password or None
                    )

                st.session_state["statement_analyzer_report"] = report
                st.session_state["statement_analyzer_filename"] = (
                    uploaded_statement.name
                )

                st.success(
                    "Statement analysis finished. "
                    "Please review flagged or unclear rows."
                )

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error(f"Could not analyze this statement: {error}")

    report = st.session_state.get("statement_analyzer_report")
    saved_filename = st.session_state.get(
        "statement_analyzer_filename"
    )

    if (
        report is not None
        and uploaded_statement is not None
        and saved_filename == uploaded_statement.name
    ):

        transactions = report.get(
            "transactions", pd.DataFrame()
        ).copy()

        if not transactions.empty:

            st.divider()
            st.subheader("Statement Overview")

            currency_symbols = {
                "INR": "₹",
                "USD": "$",
                "EUR": "€",
                "GBP": "£"
            }

            def currency_totals(series, currency_series):
                values = pd.to_numeric(
                    series, errors="coerce"
                ).fillna(0)

                currencies = (
                    currency_series.fillna("Unknown")
                    .astype(str)
                    .str.upper()
                )

                grouped = values.groupby(currencies).sum()

                if grouped.empty or float(grouped.abs().sum()) == 0:
                    return "0.00"

                parts = []

                for currency, amount in grouped.items():
                    symbol = currency_symbols.get(
                        currency, f"{currency} "
                    )
                    parts.append(f"{symbol}{amount:,.2f}")

                return " / ".join(parts)

            currency_series = (
                transactions["currency"]
                if "currency" in transactions.columns
                else pd.Series(
                    ["Unknown"] * len(transactions),
                    index=transactions.index
                )
            )

            total_spent = currency_totals(
                transactions.get(
                    "debit",
                    pd.Series(
                        index=transactions.index,
                        dtype=float
                    )
                ),
                currency_series
            )

            total_received = currency_totals(
                transactions.get(
                    "credit",
                    pd.Series(
                        index=transactions.index,
                        dtype=float
                    )
                ),
                currency_series
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Spent", total_spent)
            col2.metric("Total Received", total_received)
            col3.metric(
                "Transactions Extracted",
                f"{len(transactions):,}"
            )

            provider = report.get(
                "provider", "Statement provider unknown"
            )

            st.caption(f"Detected provider: {provider}")

            direction_series = transactions.get(
                "direction",
                pd.Series(
                    ["Unknown"] * len(transactions),
                    index=transactions.index
                )
            )

            unknown_count = int(
                (direction_series == "Unknown").sum()
            )

            if unknown_count:
                st.warning(
                    f"{unknown_count} row(s) have an unclear "
                    "debit/credit direction. Review these manually."
                )

            st.divider()
            st.subheader("Extracted Transactions")

            st.dataframe(
                transactions,
                use_container_width=True,
                hide_index=True
            )

            # ================================================
            # FRAUD DETECTION & RISK ANALYTICS ADD-ON
            # ================================================

            st.divider()
            st.subheader("🛡️ Fraud Detection & Risk Analytics")

            risk_df = transactions.copy()

            # Make sure required columns exist
            for col in ["amount", "date", "description", "currency"]:
                if col not in risk_df.columns:
                    risk_df[col] = None

            risk_df["amount"] = pd.to_numeric(
                risk_df["amount"], errors="coerce"
            ).abs()

            risk_df["date"] = pd.to_datetime(
                risk_df["date"], errors="coerce"
            )

            risk_df["description"] = (
                risk_df["description"].fillna("").astype(str).str.lower().str.strip()
            )

            risk_df["currency"] = (
                risk_df["currency"].fillna("Unknown").astype(str)
            )

            risk_df["risk_score"] = 0
            risk_df["risk_reason"] = ""

            # Find possible duplicate transactions
            duplicate = risk_df.duplicated(
                subset=["date", "amount", "currency", "description"],
                keep=False
            )

            # Find unusually large amounts within each currency
            large_amount = pd.Series(
                False, index=risk_df.index
            )

            for currency, indexes in risk_df.groupby("currency").groups.items():
                values = risk_df.loc[indexes, "amount"].dropna()

                if len(values) >= 5 and values.median() > 0:
                    threshold = values.median() * 5
                    large_amount.loc[indexes] = (
                        risk_df.loc[indexes, "amount"] >= threshold
                    )

            # Score the transactions using transparent rules
            for idx in risk_df.index:
                score = 0
                reasons = []

                if duplicate.loc[idx]:
                    score += 40
                    reasons.append("Possible duplicate")

                if large_amount.loc[idx]:
                    score += 35
                    reasons.append("Unusually large amount")

                old_reason = str(
                    risk_df.at[idx, "review_reason"]
                    if "review_reason" in risk_df.columns else ""
                ).lower()

                if any(word in old_reason for word in [
                    "unclear", "uncertain", "failed",
                    "pending", "declined", "multiple monetary values"
                ]):
                    score += 25
                    reasons.append("Transaction needs verification")

                risk_df.at[idx, "risk_score"] = min(score, 100)
                risk_df.at[idx, "risk_reason"] = (
                    "; ".join(reasons) if reasons else "No rule triggered"
                )

            risk_df["risk_level"] = risk_df["risk_score"].apply(
                lambda score:
                    "CRITICAL" if score >= 75 else
                    "HIGH" if score >= 50 else
                    "MEDIUM" if score >= 25 else "LOW"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Transactions for Review",
                int((risk_df["risk_score"] >= 25).sum())
            )

            col2.metric(
                "High Risk",
                int((risk_df["risk_level"] == "HIGH").sum())
            )

            col3.metric(
                "Critical Risk",
                int((risk_df["risk_level"] == "CRITICAL").sum())
            )

            st.subheader("Risk Distribution")

            risk_chart_data = (
                risk_df["risk_level"].value_counts()
                .reindex(
                    ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    fill_value=0
                )
                .rename_axis("Risk Level")
                .reset_index(name="Transactions")
            )

            fig = px.bar(
                risk_chart_data,
                x="Risk Level",
                y="Transactions",
                title="Transactions by Risk Level"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Transactions Requiring Review")

            review_df = risk_df[
                risk_df["risk_score"] >= 25
            ]

            if not review_df.empty:
                st.dataframe(
                    review_df[
                        [
                            "date", "description", "amount",
                            "currency", "risk_score",
                            "risk_level", "risk_reason"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No transactions matched the current review rules.")

            st.download_button(
                "Download Risk Report CSV",
                data=risk_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="statement_risk_report.csv",
                mime="text/csv"
            )

            csv_data = transactions.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "⬇️ Download Transactions CSV",
                data=csv_data,
                file_name="statement_transactions.csv",
                mime="text/csv",
                key="download_statement_transactions"
            )

            st.divider()
            st.subheader("Spending by Category")

            category_summary = report.get(
                "category_summary", pd.DataFrame()
            )

            if (
                category_summary is not None
                and not category_summary.empty
            ):

                st.dataframe(
                    category_summary,
                    use_container_width=True,
                    hide_index=True
                )

                if {
                    "category", "total_spent"
                }.issubset(category_summary.columns):

                    category_fig = px.bar(
                        category_summary,
                        x="category",
                        y="total_spent",
                        title="Expense Category Totals",
                        labels={
                            "category": "Category",
                            "total_spent": "Amount spent"
                        }
                    )

                    st.plotly_chart(
                        category_fig,
                        use_container_width=True
                    )

                st.caption(
                    "For statements with multiple currencies, "
                    "verify totals separately by currency."
                )

            else:
                st.info(
                    "No confidently identified debit categories are available."
                )

            st.subheader("Weekly Report")

            weekly = report.get(
                "weekly_summary", pd.DataFrame()
            )

            if weekly is not None and not weekly.empty:
                st.dataframe(
                    weekly,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "No weekly report could be prepared from the extracted dates."
                )

            st.subheader("Monthly Report")

            monthly = report.get(
                "monthly_summary", pd.DataFrame()
            )

            if monthly is not None and not monthly.empty:
                st.dataframe(
                    monthly,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "No monthly report could be prepared from the extracted dates."
                )

            st.subheader("Transactions Requiring Review")

            flagged = report.get(
                "flagged_transactions", pd.DataFrame()
            )

            if flagged is not None and not flagged.empty:

                st.warning(
                    f"{len(flagged)} transaction(s) matched the review rules. "
                    "These are not confirmed fraudulent transactions."
                )

                st.dataframe(
                    flagged,
                    use_container_width=True,
                    hide_index=True
                )

                flagged_csv = flagged.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    "⬇️ Download Review Queue CSV",
                    data=flagged_csv,
                    file_name="statement_review_queue.csv",
                    mime="text/csv",
                    key="download_statement_review_queue"
                )

            else:
                st.success(
                    "No transaction matched the current review rules."
                )

            if st.button(
                "Clear Statement Results",
                key="clear_statement_results"
            ):
                st.session_state.pop(
                    "statement_analyzer_report", None
                )
                st.session_state.pop(
                    "statement_analyzer_filename", None
                )
                st.session_state.pop(
                    "bank_statement_pdf_password", None
                )
                st.rerun()

        else:
            st.warning(
                "No transaction rows were returned by the parser."
            )


# ============================================================
# SIDEBAR FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Banking Fraud Detection & Risk Analytics"
)

st.sidebar.caption(
    f"Best Model: {best_model_name}"
)
