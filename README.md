# 🏦 Banking Fraud Detection & Risk Analytics

A Machine Learning based banking transaction fraud detection and risk analytics dashboard built using Python, Scikit-learn and Streamlit.

## 🚀 Live Demo

https://sailikhith243-banking-fraud-detection-app-d1jbpa.streamlit.app/

## 📌 Project Overview

This project analyzes banking transactions and identifies potentially fraudulent transactions using Machine Learning.

The system also assigns risk scores to transactions and provides an interactive Streamlit dashboard for analyzing fraud patterns, transaction behavior and risk levels.

## 🎯 Objectives

- Detect potentially fraudulent banking transactions
- Analyze transaction behavior and fraud patterns
- Calculate transaction risk scores
- Visualize important transaction features
- Provide an interactive fraud analytics dashboard
- Deploy the application publicly using Streamlit Community Cloud

## 📊 Dataset

The dataset contains **10,000 banking transactions** with 20 transaction-related features.

### Dataset Summary

- Total Transactions: 10,000
- Fraud Transactions: 1,251
- Genuine Transactions: 8,749
- Fraud Percentage: 12.51%

### Important Features

- Transaction Amount
- Login Attempts
- Device Risk Score
- Transfer Frequency
- Anomaly Score
- Account Age
- Failed Transactions
- Average Monthly Balance
- Transaction Velocity
- Geographic Distance
- Payment Channel
- Authentication Type
- International Transaction Flag
- Suspicious IP Flag

## 🤖 Machine Learning

The project uses a Machine Learning classification approach to identify fraudulent transactions.

### Model

- Random Forest Classifier
- Feature preprocessing
- Train/Test evaluation
- Fraud classification
- Risk scoring

## 📈 Dashboard Features

The Streamlit dashboard contains:

### 1. Dashboard
Provides an overview of total transactions, fraud transactions and genuine transactions.

### 2. Transaction Analysis
Analyzes transaction amounts, channels and transaction behavior.

### 3. Fraud Analysis
Explores fraud patterns and important fraud-related features.

### 4. Risk Analysis
Displays transaction risk scores and risk-level distributions.

### 5. Dataset
Provides an interactive view of the transaction dataset.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Git
- GitHub

## 📁 Project Structure

```text
banking-fraud-detection/
│
├── app.py
├── banking.py
├── banking_transactions.csv
├── requirements.txt
│
├── models/
│   └── fraud_detection_model.pkl
│
└── outputs/
    ├── plots/
    └── transaction_risk_scores.csv
📊 Dashboard

The Streamlit dashboard contains:

🏠 Dashboard
💳 Transaction Analysis
🚨 Fraud Analysis
⚠️ Risk Analysis
📋 Dataset Preview
🛠️ Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Streamlit
Git & GitHub
📁 Project Structure
banking-fraud-detection/
│
├── app.py
├── banking.py
├── banking_transactions.csv
├── requirements.txt
│
├── models/
│   └── fraud_detection_model.pkl
│
└── outputs/
    └── transaction_risk_scores.csv
🌐 Deployment

The application is deployed using Streamlit Community Cloud.

🔗 Live Banking Fraud Detection Dashboard

🔮 Future Scope
Real-time fraud detection
Real-time fraud alerts
Explainable AI
Advanced anomaly detection
AWS cloud integration
Real-time transaction monitoring
⚠️ Disclaimer

This project is developed for educational and demonstration purposes. Fraud predictions should not be treated as definitive proof of fraudulent activity.

👨‍💻 Author

Sai Likhith

B.Tech – Computer Science & Engineering (IoT)

🔗 GitHub
