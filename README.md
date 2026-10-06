# Banking Fraud Detection & Risk Analytics

A machine-learning based platform for detecting suspicious banking transactions, analyzing risk signals, and supporting transaction review decisions.

## Project Overview

Financial institutions process a large number of transactions every day. Identifying potentially fraudulent transactions requires analyzing transaction behavior, customer activity, device information, authentication methods, and security-related signals.

This project combines machine learning with transaction risk analysis to identify suspicious transactions and support investigation decisions.

The system provides:

- Fraud detection using multiple machine learning models
- Transaction-level fraud probability
- Behavioral and security risk analysis
- Risk scoring and risk classification
- Recommended actions for suspicious transactions
- Fraud and risk analytics dashboards
- Transaction investigation
- Model performance comparison
- High-risk transaction monitoring

## Problem Statement

The objective of this project is to answer:

> Is this transaction normal, suspicious, or likely fraudulent, and what action should be taken?

The system analyzes transaction and behavioral signals to estimate fraud probability and determine the level of risk associated with a transaction.

## Project Architecture

```text
Banking Transaction
        |
        v
Transaction & Behavioral Features
        |
        v
Machine Learning Models
        |
        v
Fraud Probability
        |
        v
Risk Scoring
        |
        v
Risk Level
        |
        v
Recommended Action
        |
        v
Transaction Investigation / Monitoring
## Key Features

### 1. Executive Dashboard

Provides an overall view of transaction activity and fraud risk.

Key metrics include:

- Total Transactions
- Fraud Detected
- Fraud Rate
- High / Critical Risk Transactions

The dashboard also provides visual analysis of:

- Transaction and fraud trends
- Risk distribution
- Fraud rate by payment channel

### 2. Transaction Investigation

Allows individual transactions to be investigated.

For a selected transaction, the system displays:

- Fraud probability
- Risk score
- Risk level
- Recommended action
- Transaction details
- Behavioral signals
- Security signals
- Dataset fraud label

This module helps understand why a transaction may require additional review.

### 3. Fraud Analysis

Provides fraud analysis across different transaction characteristics.

The analysis includes:

- Merchant category
- Device type
- Authentication type
- Customer segment
- Transaction amount

### 4. Risk Intelligence

The system classifies transactions into four risk levels:

- LOW
- MEDIUM
- HIGH
- CRITICAL

High and critical risk transactions are highlighted for further investigation.

### 5. Model Lab

Multiple machine learning models are evaluated and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Isolation Forest

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### 6. Fraud Monitoring

The monitoring module provides a transaction review queue based on recent transactions.

Users can filter transactions by:

- LOW
- MEDIUM
- HIGH
- CRITICAL

The module highlights transactions that may require manual review.