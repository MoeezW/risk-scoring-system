from __future__ import annotations

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------
# Path setup so imports work even when running Streamlit
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl.transformers import load_raw_credit_csv, clean_credit_applications  # noqa: E402


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def bucket_risk(score: float, low: float, high: float) -> str:
    if score < low:
        return "low"
    if score < high:
        return "medium"
    return "high"


def extract_expected_columns_from_model(model) -> set[str]:
    preprocess = model.named_steps.get("preprocess")
    if preprocess is None:
        raise ValueError("Model pipeline does not contain a 'preprocess' step.")
    expected = set()
    for _, _, cols in preprocess.transformers_:
        expected.update(cols)
    return expected


def load_csv_cached(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "credit_default_clean.csv"
    return load_csv_cached(path)


@st.cache_data(show_spinner=False)
def load_scored_batch() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "new_applications_scored.csv"
    return load_csv_cached(path)


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = PROJECT_ROOT / "models" / "artifacts" / "best_model.pkl"
    return joblib.load(model_path)


def compute_model_metrics_on_holdout(model, df: pd.DataFrame) -> dict:
    """
    Recompute metrics using the same split approach as training:
      - stratify on target
      - same test_size/random_state

    Note: the model is already fitted; this just evaluates it on the holdout.
    """
    target_col = "default_payment_next_month"
    id_col = "id"

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "avg_precision": float(average_precision_score(y_test, proba)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }

    # curves for plotting
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    return {
        "metrics": metrics,
        "roc": (fpr, tpr),
        "pr": (rec, prec),  # x=recall, y=precision
    }


def plot_roc(fpr, tpr):
    fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    return fig


def plot_pr(recall, precision):
    fig = plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    return fig


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    bins = [18, 25, 35, 45, 55, 65, 200]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    out = df.copy()
    out["age_group"] = pd.cut(out["age"], bins=bins, labels=labels, right=False)
    return out


def fairness_table(df: pd.DataFrame, group_col: str, threshold: float) -> pd.DataFrame:
    tmp = df.dropna(subset=[group_col, "risk_score"]).copy()
    tmp["is_high_risk"] = (tmp["risk_score"] >= threshold).astype(int)

    # observed=False avoids pandas future warning changes around categorical groupby
    summary = (
        tmp.groupby(group_col, observed=False)
        .agg(
            count=("risk_score", "size"),
            avg_risk=("risk_score", "mean"),
            high_risk_rate=("is_high_risk", "mean"),
        )
        .reset_index()
        .sort_values(by="high_risk_rate", ascending=False)
    )
    summary["high_risk_rate"] = summary["high_risk_rate"].astype(float)
    summary["avg_risk"] = summary["avg_risk"].astype(float)
    return summary


# ---------------------------------------------------------
# App UI
# ---------------------------------------------------------
st.set_page_config(page_title="Risk Scoring System", layout="wide")

st.title("Credit Risk Scoring System")
st.write(
    """
This dashboard demonstrates an end-to-end legacy-style risk scoring system:
data ingestion → model training → batch scoring → monitoring (drift + basic fairness).
"""
)

# Sidebar controls
st.sidebar.header("Controls")
risk_low = st.sidebar.slider("Low risk threshold", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
risk_high = st.sidebar.slider("High risk threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
if risk_low >= risk_high:
    st.sidebar.error("Low threshold must be < High threshold.")

decision_threshold = st.sidebar.slider("Decision threshold for 'High Risk' checks", 0.0, 1.0, 0.50, 0.01)

st.divider()

# ---------------------------------------------------------
# Section 2 — Data summary
# ---------------------------------------------------------
st.header("Data summary")

train_df = load_training_data()

colA, colB, colC = st.columns(3)
colA.metric("Rows (training dataset)", f"{len(train_df):,}")
colB.metric("Columns", f"{train_df.shape[1]}")
default_rate = float(train_df["default_payment_next_month"].mean())
colC.metric("Default rate", f"{default_rate:.3f}")

# Class balance chart
counts = train_df["default_payment_next_month"].value_counts().sort_index()
chart_df = pd.DataFrame(
    {"label": ["No default (0)", "Default (1)"], "count": [int(counts.get(0, 0)), int(counts.get(1, 0))]}
).set_index("label")
st.bar_chart(chart_df)

st.caption("Class balance matters: imbalanced data makes accuracy misleading; prefer AUC / PR metrics.")

st.divider()

# ---------------------------------------------------------
# Section 3 — Model performance
# ---------------------------------------------------------
st.header("Model performance (best model)")

model = load_model()
perf = compute_model_metrics_on_holdout(model, train_df)
m = perf["metrics"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ROC AUC", f"{m['roc_auc']:.4f}")
m2.metric("Avg Precision", f"{m['avg_precision']:.4f}")
m3.metric("Precision@0.5", f"{m['precision']:.4f}")
m4.metric("Recall@0.5", f"{m['recall']:.4f}")
m5.metric("F1@0.5", f"{m['f1']:.4f}")

fpr, tpr = perf["roc"]
rec, prec = perf["pr"]

pcol1, pcol2 = st.columns(2)
with pcol1:
    st.pyplot(plot_roc(fpr, tpr))
with pcol2:
    st.pyplot(plot_pr(rec, prec))

st.caption("These metrics are recomputed on a holdout split for transparency.")

st.divider()

# ---------------------------------------------------------
# Section 4 — Risk distributions (scored batch)
# ---------------------------------------------------------
st.header("Risk distributions (latest scored batch)")

scored_df = load_scored_batch()

# Re-bucket using current slider thresholds (so the dashboard is interactive)
scored_view = scored_df.copy()
scored_view["risk_bucket"] = scored_view["risk_score"].apply(lambda s: bucket_risk(float(s), risk_low, risk_high))

c1, c2, c3 = st.columns(3)
c1.metric("Scored rows", f"{len(scored_view):,}")
c2.metric("High risk (>= high threshold)", f"{int((scored_view['risk_score'] >= risk_high).sum()):,}")
c3.metric("Low risk (< low threshold)", f"{int((scored_view['risk_score'] < risk_low).sum()):,}")

# Histogram
hist_counts, bin_edges = np.histogram(scored_view["risk_score"], bins=30, range=(0.0, 1.0))
hist_df = pd.DataFrame({"bin_left": bin_edges[:-1], "count": hist_counts}).set_index("bin_left")
st.line_chart(hist_df)

st.write("Bucket counts (based on current thresholds):")
st.dataframe(scored_view["risk_bucket"].value_counts().rename_axis("bucket").reset_index(name="count"))

st.divider()

# ---------------------------------------------------------
# Governance / Monitoring (drift + fairness)
# ---------------------------------------------------------
st.header("Governance checks (drift + basic fairness)")

drift_path = PROJECT_ROOT / "reports" / "data_shift_report.csv"
if drift_path.exists():
    drift = pd.read_csv(drift_path)
    flagged = drift[drift["drift_flag"] == True]
    st.subheader("Data drift summary")
    st.write(f"Monitored features: {len(drift)} | Flagged: {len(flagged)}")
    st.dataframe(drift.head(15))
else:
    st.info("No drift report found yet. Run: python monitoring/monitor_data_shift.py")

st.subheader("Basic fairness sanity check (score parity)")
st.caption(
    "This is a score-based parity check (no ground truth outcomes). Large gaps can be real or due to small group sizes."
)

# fairness tables computed directly from scored batch for interactivity
fair_df = add_age_group(scored_view)

tab1, tab2, tab3 = st.tabs(["By sex", "By age group", "By education"])
with tab1:
    if "sex" in fair_df.columns:
        t = fairness_table(fair_df, "sex", decision_threshold)
        st.dataframe(t)
        st.write("High-risk rate gap (max-min):", float(t["high_risk_rate"].max() - t["high_risk_rate"].min()))
    else:
        st.info("No 'sex' column available in scored batch.")
with tab2:
    t = fairness_table(fair_df, "age_group", decision_threshold)
    st.dataframe(t)
    st.write("High-risk rate gap (max-min):", float(t["high_risk_rate"].max() - t["high_risk_rate"].min()))
with tab3:
    if "education" in fair_df.columns:
        t = fairness_table(fair_df, "education", decision_threshold)
        st.dataframe(t)
        st.write("High-risk rate gap (max-min):", float(t["high_risk_rate"].max() - t["high_risk_rate"].min()))
    else:
        st.info("No 'education' column available in scored batch.")

st.divider()

# ---------------------------------------------------------
# Section 5 — Individual scoring (upload CSV)
# ---------------------------------------------------------
st.header("Score your own CSV (optional)")

st.write(
    "Upload a CSV with the same schema as applications (e.g., columns like limit_bal, sex, education, ...). "
    "We’ll clean it using the shared ETL transforms and score with the saved model pipeline."
)

uploaded = st.file_uploader("Upload new applications CSV", type=["csv"])

if uploaded is not None:
    upload_df = pd.read_csv(uploaded)

    cleaned = clean_credit_applications(upload_df, require_id=False)

    # Build X for scoring
    X = cleaned.copy()
    if "default_payment_next_month" in X.columns:
        X = X.drop(columns=["default_payment_next_month"])
    if "id" in X.columns:
        ids = X["id"].astype(int)
        X = X.drop(columns=["id"])
    else:
        ids = pd.Series(range(1, len(X) + 1))

    expected = extract_expected_columns_from_model(model)
    missing = expected - set(X.columns)
    if missing:
        st.error("Uploaded CSV is missing required columns expected by the model:")
        st.write(sorted(missing))
    else:
        proba = model.predict_proba(X)[:, 1]
        out = cleaned.copy()
        out["risk_score"] = proba
        out["risk_bucket"] = out["risk_score"].apply(lambda s: bucket_risk(float(s), risk_low, risk_high))

        st.success(f"Scored {len(out)} rows.")
        st.dataframe(out.head(50))

        # Download
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download scored CSV",
            data=csv_bytes,
            file_name="uploaded_applications_scored.csv",
            mime="text/csv",
        )
