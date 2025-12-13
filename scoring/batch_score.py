from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import joblib

from etl.transformers import load_raw_credit_csv, clean_credit_applications


def extract_expected_columns_from_model(model) -> set[str]:
    """
    Our saved model is a sklearn Pipeline:
      pipeline.named_steps['preprocess'] is a ColumnTransformer
    ColumnTransformer stores which columns it expects for each transformer.
    We'll extract those so we can validate the input schema before scoring.
    """
    preprocess = model.named_steps.get("preprocess")
    if preprocess is None:
        raise ValueError("Model pipeline does not contain a 'preprocess' step.")

    expected = set()
    # preprocess.transformers_ contains tuples: (name, transformer, columns)
    for _, _, cols in preprocess.transformers_:
        # cols is a list of column names
        expected.update(cols)
    return expected


def bucket_risk(score: float, low: float, high: float) -> str:
    """
    Convert probability score into a risk bucket.
      - low risk: score < low
      - medium: low <= score < high
      - high: score >= high
    """
    if score < low:
        return "low"
    if score < high:
        return "medium"
    return "high"


def main():
    parser = argparse.ArgumentParser(description="Batch score new credit applications.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/new_applications.csv",
        help="Path to raw new applications CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/new_applications_scored.csv",
        help="Path to write scored CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/artifacts/best_model.pkl",
        help="Path to saved model pipeline.",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.20,
        help="Risk bucket threshold: below this is 'low'.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.50,
        help="Risk bucket threshold: at/above this is 'high'.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / args.input
    output_path = project_root / args.output
    model_path = project_root / args.model

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # 1) Load model (Pipeline = preprocess + estimator)
    model = joblib.load(model_path)

    # 2) Load + clean new applications using shared logic
    raw_df = load_raw_credit_csv(input_path)
    clean_df = clean_credit_applications(raw_df, require_id=False)

    # 3) Prepare feature matrix X (drop ID and target if present)
    id_col = "id"
    target_col = "default_payment_next_month"

    X = clean_df.copy()
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    if id_col in X.columns:
        X = X.drop(columns=[id_col])

    # 4) Validate schema: ensure X has all columns the model expects
    expected_cols = extract_expected_columns_from_model(model)
    missing = expected_cols - set(X.columns)
    if missing:
        raise ValueError(
            "Input data is missing required feature columns expected by the model:\n"
            f"{sorted(missing)}\n\n"
            "Make sure new_applications.csv has the same schema as training data."
        )

    # 5) Predict probabilities
    proba = model.predict_proba(X)[:, 1]

    scored = clean_df.copy()
    scored["risk_score"] = proba

    # 6) Risk buckets
    low_t = args.low_threshold
    high_t = args.high_threshold
    if not (0.0 <= low_t <= 1.0 and 0.0 <= high_t <= 1.0 and low_t < high_t):
        raise ValueError("Thresholds must satisfy 0<=low<high<=1")

    scored["risk_bucket"] = scored["risk_score"].apply(lambda s: bucket_risk(float(s), low_t, high_t))

    # 7) Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print("Scored rows:", len(scored))
    print("Wrote:", output_path)
    print("Risk bucket counts:")
    print(scored["risk_bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
