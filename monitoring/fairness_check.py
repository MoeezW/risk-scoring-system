from __future__ import annotations

from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    # Reasonable bins for adult populations
    bins = [18, 25, 35, 45, 55, 65, 200]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    return df


def group_summary(df: pd.DataFrame, group_col: str, score_col: str, threshold: float) -> pd.DataFrame:
    """
    Summarize predicted risk by group:
      - avg risk
      - high-risk rate at threshold (risk >= threshold)
      - count
    """
    tmp = df.dropna(subset=[group_col, score_col]).copy()
    tmp["is_high_risk"] = (tmp[score_col] >= threshold).astype(int)

    summary = (
        tmp.groupby(group_col)
        .agg(
            count=(score_col, "size"),
            avg_risk=(score_col, "mean"),
            high_risk_rate=("is_high_risk", "mean"),
        )
        .reset_index()
        .sort_values(by="high_risk_rate", ascending=False)
    )
    return summary


def disparity(stats: pd.Series) -> float:
    if stats.empty:
        return 0.0
    return float(stats.max() - stats.min())


def main():
    parser = argparse.ArgumentParser(description="Basic fairness sanity check on predicted risk scores.")
    parser.add_argument("--scored", type=str, default="data/processed/new_applications_scored.csv")
    parser.add_argument("--score-col", type=str, default="risk_score")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for 'high risk'")
    parser.add_argument("--warn-gap", type=float, default=0.10, help="Warn if max-min high-risk rate gap exceeds this")
    parser.add_argument("--out-csv", type=str, default="reports/fairness_report.csv")
    parser.add_argument("--out-json", type=str, default="reports/fairness_report.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    scored_path = project_root / args.scored
    out_csv = project_root / args.out_csv
    out_json = project_root / args.out_json

    if not scored_path.exists():
        raise FileNotFoundError(f"Scored batch not found: {scored_path}")

    df = pd.read_csv(scored_path)

    required = {"sex", "age", args.score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for fairness check: {missing}")

    df = add_age_group(df)

    # Group summaries
    sex_summary = group_summary(df, "sex", args.score_col, args.threshold)
    age_summary = group_summary(df, "age_group", args.score_col, args.threshold)

    # Optional: education group if present
    education_summary = None
    if "education" in df.columns:
        education_summary = group_summary(df, "education", args.score_col, args.threshold)

    # Compute disparity on high-risk rates
    sex_gap = disparity(sex_summary["high_risk_rate"])
    age_gap = disparity(age_summary["high_risk_rate"])
    edu_gap = disparity(education_summary["high_risk_rate"]) if education_summary is not None else None

    warnings = []
    if sex_gap > args.warn_gap:
        warnings.append(f"High-risk rate gap across sex groups is {sex_gap:.3f} (> {args.warn_gap}).")
    if age_gap > args.warn_gap:
        warnings.append(f"High-risk rate gap across age groups is {age_gap:.3f} (> {args.warn_gap}).")
    if edu_gap is not None and edu_gap > args.warn_gap:
        warnings.append(f"High-risk rate gap across education groups is {edu_gap:.3f} (> {args.warn_gap}).")

    # Save combined CSV (stacked sections)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write a single CSV with section headers for readability
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(f"# threshold={args.threshold}\n")
        f.write("# --- sex ---\n")
        sex_summary.to_csv(f, index=False)
        f.write("\n# --- age_group ---\n")
        age_summary.to_csv(f, index=False)
        if education_summary is not None:
            f.write("\n# --- education ---\n")
            education_summary.to_csv(f, index=False)

    payload = {
        "threshold": args.threshold,
        "warn_gap": args.warn_gap,
        "sex_gap": sex_gap,
        "age_gap": age_gap,
        "education_gap": edu_gap,
        "warnings": warnings,
        "notes": [
            "This is a basic score-parity sanity check, not a compliance-grade fairness audit.",
            "No ground-truth outcomes are available for the new batch, so we cannot compute equalized odds / error-rate parity.",
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("-", w)
    else:
        print("\nNo disparity warnings triggered.")


if __name__ == "__main__":
    main()
