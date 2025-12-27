# Credit Risk Scoring System (Legacy-Style)

End-to-end risk scoring pipeline for credit default prediction:
**ETL → SQLite → Training + MLflow → Batch scoring → Monitoring (drift + fairness) → Streamlit dashboard**.

---

## What it does

- Ingests raw credit application data from CSV
- Cleans and normalizes fields (consistent ETL + inference transforms)
- Stores cleaned data in SQLite for reproducibility
- Trains multiple models and tracks runs in MLflow
- Selects and saves the best model as a single sklearn Pipeline (`best_model.pkl`)
- Scores new application batches and writes risk scores + risk buckets
- Produces basic governance reports:
  - data shift (simple drift checks)
  - basic fairness sanity checks (score parity / high-risk-rate parity)
- Provides a Streamlit dashboard for exploration and scoring uploads

---

## Architecture

```text
RAW CSV
  │
  ▼
ETL (clean + normalize)
  │
  ├──> data/processed/credit_default_clean.csv
  │
  └──> SQLite (credit_risk.sqlite)
           │
           ▼
Training (models/train_model.py)
  │  ├── MLflow runs (metrics + artifacts + model)
  │  └── best model saved: models/artifacts/best_model.pkl
  │
  ▼
Batch scoring (scoring/batch_score.py)
  │
  └──> data/processed/new_applications_scored.csv
           │
           ▼
Monitoring (monitoring/*.py)  → reports/*.csv / *.json
           │
           ▼
Dashboard (dashboard/app.py)
```

---

## Tech stack

    Python 3.11
    pandas / numpy
    scikit-learn (Pipeline + preprocessing + models)
    MLflow (experiment tracking + model logging)
    SQLite (simple warehouse-like storage)
    Streamlit (dashboard)
    matplotlib (plots)

---

## Repo structure

risk-scoring-system/
  config/
    settings_example.yaml
  data/
    raw/
      credit_default_raw.csv
      new_applications.csv
    processed/
      credit_default_clean.csv
      new_applications_scored.csv
  dashboard/
    app.py
  etl/
    etl_credit_data.py
    transformers.py
  models/
    model_config.yaml
    train_model.py
    artifacts/
      best_model.pkl
      best_model_info.json
  monitoring/
    monitor_data_shift.py
    fairness_check.py
  reports/
    data_shift_report.csv
    fairness_report.csv
    fairness_report.json
    risk_scoring_case_study.md
    loom_script.md
  credit_risk.sqlite
  requirements.txt
  README.md

---

## Dataset notes

This project uses the **UCI “Default of Credit Card Clients”** dataset (Taiwan credit card clients).

- Size: **30,000 rows** (clients) with **24 input features** + an ID column.
- Target: `default_payment_next_month`  
  - **1 = defaulted next month**, **0 = did not default**.
- Feature groups:
  - Demographics: `sex`, `education`, `marriage`, `age`
  - Credit limit: `limit_bal`
  - Repayment history codes: `pay_0`, `pay_2` … `pay_6`
  - Bill statement amounts: `bill_amt1` … `bill_amt6`
  - Past payment amounts: `pay_amt1` … `pay_amt6`
- Notes:
  - `pay_*` columns are **ordinal repayment status codes** (delinquency level). 
  - This dataset is a **historical snapshot** and is used here to demonstrate an end-to-end risk scoring pipeline (ETL, training, scoring, monitoring), not to claim real-world lending decisions.

---

## Setup

Create and activate a virtual environment (Windows PowerShell or equivalent):
cd C:\dev\risk-scoring-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

---

## How to run (end-to-end):

1 - ETL: raw CSV → cleaned CSV + SQLite
Input:
data/raw/credit_default_raw.csv
Run:
python etl\etl_credit_data.py
Outputs:
data/processed/credit_default_clean.csv
credit_risk.sqlite (table: credit_applications)

2 - Train models + track with MLflow
Run training:
python models\train_model.py
Start MLflow UI:
mlflow ui
Open:
http://127.0.0.1:5000
Outputs:
MLflow runs logged in mlruns/
Best model saved:
models/artifacts/best_model.pkl
models/artifacts/best_model_info.json

3 - Create a fake “new applications” batch (for demo/testing)
python -c "import pandas as pd; df=pd.read_csv('data/processed/credit_default_clean.csv'); new=df.sample(200, random_state=42).drop(columns=['default_payment_next_month']); new.to_csv('data/raw/new_applications.csv', index=False); print(new.shape)"

4 - Batch scoring: new applications → risk scores + buckets
Run:
python scoring\batch_score.py
Output:
data/processed/new_applications_scored.csv

5 - Monitoring: drift + fairness sanity checks
Run:
python monitoring\monitor_data_shift.py
python monitoring\fairness_check.py
Outputs:
reports/data_shift_report.csv
reports/fairness_report.csv
reports/fairness_report.json

6 - Streamlit dashboard
Run:
streamlit run dashboard\app.py
Open:
http://localhost:8501
Dashboard features:
dataset summary + class balance
model performance (AUC/PR + curves)
risk distribution + thresholding
governance tables (drift + fairness)
upload CSV to score on the fly

---

## Notes on governance outputs
Drift check is intentionally simple and explainable:
    it flags large mean shifts relative to training std dev
Fairness check is a sanity check on scores:
    no ground truth outcomes for new batches, so it is not a compliance-grade fairness audit
    large gaps often occur when some groups have very small sample sizes