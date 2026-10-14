
# Banking Fraud Detection & Risk Analytics

An interactive application that helps users explore banking transactions, identify potential fraud patterns, understand transaction risk, and review spending from uploaded financial statements.

**Live Demo:** https://sailikhith243-banking-fraud-detection-app-d1jbpa.streamlit.app/

**GitHub Repository:** https://github.com/SAILIKHITH243/banking-fraud-detection

## Project Overview

Banks process thousands of transactions every day. Reviewing every transaction manually can be difficult.

This project combines machine learning, data analytics, and rule-based risk screening in one interactive dashboard.

Users can explore transaction patterns, investigate individual transactions, compare model results, monitor high-risk activity, and analyze supported bank statement files.

The application is built with Python and Streamlit.

## Key Features

### 1. Executive Dashboard
- View transaction and fraud statistics.
- Explore daily transaction trends.
- Understand the distribution of transaction risk.
- Compare fraud rates across payment channels.

### 2. Transaction Investigation
- Select an individual transaction.
- View the model's fraud probability.
- Review the calculated risk score and risk level.
- Inspect available behavioral and security indicators.

### 3. Fraud Analysis
- Explore fraud patterns by merchant category.
- Compare fraud rates across device types.
- Analyze authentication methods and customer segments.
- Compare transaction amounts for fraud-labelled and genuine transactions.

### 4. Risk Intelligence
- Review transactions by risk level.
- Explore high-risk and critical-risk transaction queues.
- Examine risk indicators and summary statistics.

### 5. Model Lab
- Compare available machine learning models.
- Review evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC when provided by the training results.

### 6. Fraud Monitoring
- Review a selection of recent transactions.
- Filter transactions by risk level.
- Identify transactions that may require further investigation.

### 7. Statement Analyzer
- Upload supported PDF, CSV, TXT, Excel, and image files.
- Enter a password when a PDF is password-protected.
- Extract transaction details from supported statement layouts.
- Review spending, money received, and expense categories.
- View weekly and monthly summaries.
- Identify possible duplicate transactions, unusually large amounts, and unclear records.
- Export extracted transactions and risk screening results as CSV files.

**Important:** Statement screening uses transparent review rules. A flagged transaction is not confirmed fraud. Statement layouts and OCR quality vary, so extracted information should be checked against the original statement.

## How It Works

```text
Banking Transaction Dataset
             |
             v
Data Preparation and Analysis
             |
             v
Machine Learning Model
             |
             v
Fraud Probability and Risk Analysis
             |
             v
Interactive Streamlit Dashboard
             |
             v
Transaction Investigation and Monitoring
```

For uploaded statements, the application follows a separate workflow:

```text
Upload Statement
       |
       v
Read File / Extract Text
       |
       v
Identify Transaction Details
       |
       v
Calculate Spending and Income
       |
       v
Apply Review Rules
       |
       v
Reports and CSV Export
```

## Technology Stack

- **Python** — application logic and data processing
- **Pandas** — transaction analysis and reporting
- **NumPy** — numerical operations
- **Scikit-learn** — machine learning
- **Joblib** — loading saved models
- **Streamlit** — interactive web application
- **Plotly** — interactive charts
- **pdfplumber and PyMuPDF** — PDF processing
- **Pytesseract and Pillow** — image-based text extraction
- **OpenPyXL and xlrd** — spreadsheet support
- **Git and GitHub** — version control and source code
- **Streamlit Community Cloud** — application hosting

## Project Structure

```text
banking-fraud-detection/
|
|-- app.py
|-- statement_analyzer.py
|-- train_models.py
|-- professional_banking_fraud_risk_dataset.csv
|-- requirements.txt
|-- packages.txt
|-- README.md
|-- .gitignore
|
|-- models/
|   |-- best_model.pkl
|
|-- outputs/
|   |-- model_comparison.csv
|   |-- test_predictions.csv
```

Some files may differ depending on the current version of the repository.

## Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/SAILIKHITH243/banking-fraud-detection.git
cd banking-fraud-detection
```

### 2. Install the dependencies

Windows:

```powershell
py -m pip install -r requirements.txt
```

### 3. Run the application

```powershell
py -m streamlit run app.py
```

Open the local URL displayed in the terminal.

The dataset, saved model, and other required project files must be present in their expected locations.

## Deployment

This project is deployed using Streamlit Community Cloud.

1. Push the latest code to the GitHub `main` branch.
2. Open Streamlit Community Cloud.
3. Select the existing application.
4. Confirm that the repository is `SAILIKHITH243/banking-fraud-detection`.
5. Set the main application file to `app.py`.
6. Save the configuration and wait for deployment.
7. Open the live application and test the dashboard and statement analyzer.

Live application:

https://sailikhith243-banking-fraud-detection-app-d1jbpa.streamlit.app/

## Responsible Use

This project is intended for educational, analytical, and demonstration purposes.

- Machine learning predictions are estimates, not proof of fraud.
- Risk scores help prioritize reviews and should not be treated as definitive decisions.
- Statement extraction may contain errors, especially for scanned or unfamiliar layouts.
- Do not upload real financial statements to a public demo unless you understand the privacy and security implications.
- Never commit passwords, API keys, account credentials, or other private information to GitHub.

## Future Improvements

- Add more statement-specific parsers.
- Improve OCR and extraction validation.
- Add transaction search and date-range filters.
- Expand model evaluation and explainability.
- Improve automated testing and deployment checks.

## Author

**Sai Likhith**

GitHub: https://github.com/SAILIKHITH243

Built as a hands-on project to explore machine learning, financial analytics, risk screening, and interactive Python applications.
