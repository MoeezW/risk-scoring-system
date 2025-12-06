# Credit Risk Scoring System

This project implements a small, end-to-end credit risk scoring system similar to what a bank or insurer might use.

**Pipeline overview:**

1. **ETL (`etl/`)** – Load raw credit/default data, clean it, and write it to SQLite and `data/processed/`.
2. **Model training (`models/`)** – Train and evaluate ML models to predict default risk, log experiments with MLflow, and save the best model.
3. **Scoring (`scoring/`)** – Score new applications with the best model, producing risk scores and risk buckets.
4. **Monitoring (`monitoring/`)** – Basic data drift and fairness checks between training data and new scored batches.
5. **Dashboard (`dashboard/`)** – A Streamlit app to explore data, model performance, and risk score distributions.
6. **Reports (`reports/`)** – Generated reports and a written case study describing the system.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows PowerShell
pip install -r requirements.txt
