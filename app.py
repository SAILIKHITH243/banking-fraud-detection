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

    /* Main application background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Main content */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #0F172A;
    }

    /* Dashboard title */
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    /* Dashboard subtitle */
    .dashboard-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 28px;
    }

    /* Headings */
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

    /* KPI metric cards */
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

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF;
        border-color: #CBD5E1;
        border-radius: 8px;
    }

    /* Slider */
    div[data-testid="stSlider"] {
        padding-top: 5px;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border-color: #E2E8F0;
    }

    /* Buttons */
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

    /* Sidebar navigation */
    div[role="radiogroup"] label {
        padding: 7px 5px;
        border-radius: 7px;
    }

    /* Caption */
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

    # Feature engineering
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
    df["risk_score"]
    .apply(analytics_risk_level)
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

    st.error(
        "Trained model not found."
    )

    st.info(
        "Run this command first: py train_models.py"
    )

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

def calculate_risk(
    fraud_probability,
    transaction
):

    security_points = 0

    # New device
    if "new_device_flag" in transaction.index:

        security_points += (
            int(
                transaction["new_device_flag"]
            ) * 8
        )

    # New beneficiary
    if "new_beneficiary_flag" in transaction.index:

        security_points += (
            int(
                transaction["new_beneficiary_flag"]
            ) * 8
        )

    # International transaction
    if "international_transaction_flag" in transaction.index:

        security_points += (
            int(
                transaction[
                    "international_transaction_flag"
                ]
            ) * 8
        )

    # Suspicious IP
    if "suspicious_ip_flag" in transaction.index:

        security_points += (
            int(
                transaction[
                    "suspicious_ip_flag"
                ]
            ) * 8
        )

    # Chargeback history
    if "chargeback_history" in transaction.index:

        security_points += (
            int(
                transaction[
                    "chargeback_history"
                ]
            ) * 8
        )

    # Device risk
    if "device_risk_score" in transaction.index:

        if float(
            transaction[
                "device_risk_score"
            ]
        ) >= 70:

            security_points += 10

    # Merchant risk
    if "merchant_risk_score" in transaction.index:

        if float(
            transaction[
                "merchant_risk_score"
            ]
        ) >= 70:

            security_points += 8

    # Final risk score
    risk_score = min(
        100,
        (fraud_probability * 80)
        + security_points
    )

    # Risk level and action
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

    return (
        round(risk_score, 2),
        risk_level,
        action
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title(
    "🏦 Banking Analytics"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "Executive Dashboard",
        "Transaction Investigation",
        "Fraud Analysis",
        "Risk Intelligence",
        "Model Lab",
        "Fraud Monitoring"
    ]
)


# ============================================================
# 1. EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.header(
        "📊 Executive Dashboard"
    )

    # ========================================================
    # KPI CALCULATIONS
    # ========================================================

    total_transactions = len(df)

    fraud_transactions = int(
        df["fraud_flag"].sum()
    )

    fraud_rate = (
        fraud_transactions
        / total_transactions
        * 100
    )

    high_critical = int(
        df["analytics_risk_level"].isin(
            [
                "HIGH",
                "CRITICAL"
            ]
        ).sum()
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

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


    # ========================================================
    # DAILY TRANSACTION TREND
    # ========================================================

    st.subheader(
        "📈 Daily Transaction Trend"
    )

    trend = (
        df.set_index(
            "transaction_timestamp"
        )
        .resample("D")
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            fraud=(
                "fraud_flag",
                "sum"
            )
        )
        .reset_index()
    )

    fig = px.line(
        trend,
        x="transaction_timestamp",
        y=[
            "transactions",
            "fraud"
        ],
        markers=True,
        title="Daily Transactions vs Fraud"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # RISK DISTRIBUTION + PAYMENT CHANNEL
    # ========================================================

    col1, col2 = st.columns(2)


    # Risk distribution
    with col1:

        risk_counts = (
            df["analytics_risk_level"]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MEDIUM",
                    "HIGH",
                    "CRITICAL"
                ],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = [
            "risk_level",
            "count"
        ]

        fig = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            title="Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Payment channel
    with col2:

        channel = (
            df.groupby(
                "payment_channel"
            )["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        channel.columns = [
            "payment_channel",
            "fraud_rate"
        ]

        fig = px.bar(
            channel,
            x="payment_channel",
            y="fraud_rate",
            title="Fraud Rate by Payment Channel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# 2. TRANSACTION INVESTIGATION
# ============================================================

elif page == "Transaction Investigation":

    st.header(
        "🔎 Transaction Investigation"
    )

    st.caption(
        "Investigate an individual transaction using ML probability "
        "and behavioral/security signals."
    )


    # ========================================================
    # TRANSACTION SELECTION
    # ========================================================

    selected_transaction = st.selectbox(
        "Select Transaction",
        df["transaction_id"].astype(str).tolist()
    )


    transaction = df[
        df["transaction_id"].astype(str)
        == selected_transaction
    ].iloc[0]


    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    model_transaction = transaction.drop(
        labels=[
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
        ],
        errors="ignore"
    ).copy()


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    fraud_probability = float(
        model.predict_proba(
            pd.DataFrame(
                [model_transaction]
            )
        )[0, 1]
    )


    # ========================================================
    # CALCULATE RISK
    # ========================================================

    calculated_risk_score, calculated_risk_level, action = (
        calculate_risk(
            fraud_probability,
            transaction
        )
    )


    # ========================================================
    # KPI RESULTS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ML Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

    col2.metric(
        "Risk Score",
        f"{calculated_risk_score}/100"
    )

    col3.metric(
        "Risk Level",
        calculated_risk_level
    )

    col4.metric(
        "Recommended Action",
        action
    )


    st.divider()


    # ========================================================
    # TRANSACTION DETAILS
    # ========================================================

    st.subheader(
        "💳 Transaction Details"
    )

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
        transaction[
            transaction_columns
        ].to_frame("Value"),
        use_container_width=True
    )


    # ========================================================
    # BEHAVIOR + SECURITY SIGNALS
    # ========================================================

    st.subheader(
        "🔐 Behavior & Security Signals"
    )

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
        column
        for column in signal_columns
        if column in transaction.index
    ]

    st.dataframe(
        transaction[
            available_columns
        ].to_frame("Value"),
        use_container_width=True
    )


    # ========================================================
    # DATASET LABEL
    # ========================================================

    dataset_label = (
        "FRAUD"
        if transaction["fraud_flag"] == 1
        else "GENUINE"
    )


    if dataset_label == "FRAUD":

        st.error(
            f"Dataset Label: {dataset_label}"
        )

    else:

        st.success(
            f"Dataset Label: {dataset_label}"
        )


    st.info(
        f"Recommended Action: {action}"
    )


# ============================================================
# 3. FRAUD ANALYSIS
# ============================================================

elif page == "Fraud Analysis":

    st.header(
        "📊 Fraud Analysis"
    )

    st.caption(
        "Analyze fraud patterns across transaction and customer dimensions."
    )


    # ========================================================
    # MERCHANT CATEGORY + DEVICE
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        category = (
            df.groupby(
                "merchant_category"
            )["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        category.columns = [
            "merchant_category",
            "fraud_rate"
        ]

        fig = px.bar(
            category,
            x="merchant_category",
            y="fraud_rate",
            title="Fraud Rate by Merchant Category"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        device = (
            df.groupby(
                "device_type"
            )["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        device.columns = [
            "device_type",
            "fraud_rate"
        ]

        fig = px.bar(
            device,
            x="device_type",
            y="fraud_rate",
            title="Fraud Rate by Device Type"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # AUTHENTICATION + CUSTOMER SEGMENT
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        authentication = (
            df.groupby(
                "authentication_type"
            )["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        authentication.columns = [
            "authentication_type",
            "fraud_rate"
        ]

        fig = px.bar(
            authentication,
            x="authentication_type",
            y="fraud_rate",
            title="Fraud Rate by Authentication"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        segment = (
            df.groupby(
                "customer_segment"
            )["fraud_flag"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
            .reset_index()
        )

        segment.columns = [
            "customer_segment",
            "fraud_rate"
        ]

        fig = px.bar(
            segment,
            x="customer_segment",
            y="fraud_rate",
            title="Fraud Rate by Customer Segment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # TRANSACTION AMOUNT
    # ========================================================

    st.subheader(
        "💰 Transaction Amount Analysis"
    )


    amount_data = df.copy()

    amount_data["Transaction Type"] = (
        amount_data["fraud_flag"]
        .map(
            {
                0: "Genuine",
                1: "Fraud"
            }
        )
    )


    fig = px.box(
        amount_data,
        x="Transaction Type",
        y="transaction_amount",
        points=False,
        title="Transaction Amount: Fraud vs Genuine"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 4. RISK INTELLIGENCE
# ============================================================

elif page == "Risk Intelligence":

    st.header(
        "⚠️ Risk Intelligence"
    )

    st.caption(
        "Prioritize transactions that require additional review."
    )


    # ========================================================
    # RISK COUNTS
    # ========================================================

    low_count = int(
        (
            df["analytics_risk_level"]
            == "LOW"
        ).sum()
    )


    medium_count = int(
        (
            df["analytics_risk_level"]
            == "MEDIUM"
        ).sum()
    )


    high_count = int(
        (
            df["analytics_risk_level"]
            == "HIGH"
        ).sum()
    )


    critical_count = int(
        (
            df["analytics_risk_level"]
            == "CRITICAL"
        ).sum()
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Low Risk",
        f"{low_count:,}"
    )


    col2.metric(
        "Medium Risk",
        f"{medium_count:,}"
    )


    col3.metric(
        "High Risk",
        f"{high_count:,}"
    )


    col4.metric(
        "Critical Risk",
        f"{critical_count:,}"
    )


    st.divider()


    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Risk Distribution"
    )


    risk_distribution = (
        df["analytics_risk_level"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            ],
            fill_value=0
        )
        .reset_index()
    )


    risk_distribution.columns = [
        "Risk Level",
        "Transactions"
    ]


    col1, col2 = st.columns(2)


    with col1:

        fig = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Transactions",
            title="Transactions by Risk Level"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        fig = px.pie(
            risk_distribution,
            names="Risk Level",
            values="Transactions",
            title="Risk Level Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # ========================================================
    # HIGH + CRITICAL QUEUE
    # ========================================================

    st.subheader(
        "🚨 High & Critical Risk Investigation Queue"
    )


    high_risk = df[
        df["analytics_risk_level"].isin(
            [
                "HIGH",
                "CRITICAL"
            ]
        )
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

        st.dataframe(
            high_risk[
                display_columns
            ].head(100),
            use_container_width=True
        )

    else:

        st.success(
            "No high or critical risk transactions found."
        )


    # ========================================================
    # RISK SUMMARY
    # ========================================================

    st.subheader(
        "📋 Risk Summary"
    )


    risk_summary = (
        df.groupby(
            "analytics_risk_level"
        )
        .agg(
            transactions=(
                "transaction_id",
                "count"
            ),
            fraud_transactions=(
                "fraud_flag",
                "sum"
            ),
            average_risk_score=(
                "risk_score",
                "mean"
            ),
            average_transaction_amount=(
                "transaction_amount",
                "mean"
            )
        )
        .reset_index()
    )


    risk_summary["fraud_rate"] = (
        risk_summary["fraud_transactions"]
        / risk_summary["transactions"]
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


    st.dataframe(
        risk_summary,
        use_container_width=True
    )


# ============================================================
# 5. MODEL LAB
# ============================================================

elif page == "Model Lab":

    st.header(
        "🧪 Model Lab"
    )

    st.caption(
        "Compare machine learning models using standard classification metrics."
    )


    if os.path.exists(
        COMPARISON_PATH
    ):

        comparison = pd.read_csv(
            COMPARISON_PATH
        )


        # ====================================================
        # MODEL COMPARISON TABLE
        # ====================================================

        st.subheader(
            "Machine Learning Model Comparison"
        )


        st.dataframe(
            comparison.style.format(
                {
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}",
                    "ROC_AUC": "{:.4f}"
                }
            ),
            use_container_width=True
        )


        # ====================================================
        # METRIC SELECTOR
        # ====================================================

        metric = st.selectbox(
            "Select Metric",
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC_AUC"
            ]
        )


        sorted_comparison = (
            comparison.sort_values(
                metric,
                ascending=False
            )
        )


        fig = px.bar(
            sorted_comparison,
            x="Model",
            y=metric,
            title=f"Model Comparison — {metric}"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    else:

        st.warning(
            "model_comparison.csv not found."
        )

        st.info(
            "Run py train_models.py first."
        )


    st.success(
        f"Best Supervised Model: "
        f"{best_model_name}"
    )


# ============================================================
# 6. FRAUD MONITORING
# ============================================================

elif page == "Fraud Monitoring":

    st.header(
        "🚨 Fraud Monitoring"
    )

    st.caption(
        "Review recent transactions and quickly filter transactions "
        "based on their risk level."
    )


    # ========================================================
    # FILTERS
    # ========================================================

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
            [
                "ALL",
                "LOW",
                "MEDIUM",
                "HIGH",
                "CRITICAL"
            ]
        )


    # ========================================================
    # GET RECENT TRANSACTIONS
    # ========================================================

    monitored = (
        df.sort_values(
            "transaction_timestamp"
        )
        .tail(sample_size)
        .copy()
    )


    # ========================================================
    # MONITORING STATUS
    # ========================================================

    monitored["monitor_status"] = np.where(
        monitored[
            "analytics_risk_level"
        ].isin(
            [
                "HIGH",
                "CRITICAL"
            ]
        ),
        "REVIEW",
        "NORMAL"
    )


    # ========================================================
    # APPLY RISK FILTER
    # ========================================================

    if selected_risk != "ALL":

        monitored = monitored[
            monitored[
                "analytics_risk_level"
            ] == selected_risk
        ].copy()


    # ========================================================
    # KPI CALCULATIONS
    # ========================================================

    monitored_count = len(
        monitored
    )


    review_count = int(
        (
            monitored[
                "monitor_status"
            ] == "REVIEW"
        ).sum()
    )


    fraud_count = int(
        monitored[
            "fraud_flag"
        ].sum()
    )


    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Transactions Shown",
        f"{monitored_count:,}"
    )


    col2.metric(
        "Review Required",
        f"{review_count:,}"
    )


    col3.metric(
        "Fraud in Selection",
        f"{fraud_count:,}"
    )


    st.divider()


    # ========================================================
    # FILTER MESSAGE
    # ========================================================

    if selected_risk == "ALL":

        st.info(
            f"Showing the latest {sample_size} transactions."
        )

    else:

        st.info(
            f"Showing {selected_risk} risk transactions "
            f"from the latest {sample_size} transactions."
        )


    # ========================================================
    # MONITORING TABLE
    # ========================================================

    st.subheader(
        "📡 Transaction Monitoring Queue"
    )


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

        st.dataframe(
            monitored[
                monitoring_columns
            ]
            .sort_values(
                "risk_score",
                ascending=False
            ),
            use_container_width=True
        )

    else:

        st.warning(
            "No transactions found for the selected risk level."
        )


    # ========================================================
    # REVIEW QUEUE
    # ========================================================

    review_transactions = monitored[
        monitored[
            "monitor_status"
        ] == "REVIEW"
    ].copy()


    if len(review_transactions) > 0:

        st.subheader(
            "🔍 Transactions Requiring Review"
        )


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


        st.dataframe(
            review_transactions[
                review_columns
            ]
            .sort_values(
                "risk_score",
                ascending=False
            ),
            use_container_width=True
        )


    else:

        if selected_risk in [
            "HIGH",
            "CRITICAL"
        ]:

            st.success(
                "No transactions requiring review "
                "were found in this selection."
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