from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Simple drift monitoring: compare train vs new batch distributions.")
    parser.add_argument("--train", type=str, default="data/processed/credit_default_clean.csv")
    parser.add_argument("--new", type=str, default="data/processed/new_applications_scored.csv")
    parser.add_argument("--out", type=str, default="reports/data_shift_report.csv")
    parser.add_argument("--k", type=float, default=0.5, help="Drift threshold: flag if abs(mean_new-mean_train) > k * std_train")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / args.train
    new_path = project_root / args.new
    out_path = project_root / args.out

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not new_path.exists():
        raise FileNotFoundError(f"New batch not found: {new_path}")

    train_df = pd.read_csv(train_path)
    new_df = pd.read_csv(new_path)

    # Columns to exclude from drift stats
    exclude = {
        "id",
        "default_payment_next_month",  # label exists in train, not necessarily in new
        "risk_score",
        "risk_bucket",
    }

    # Categorical-like columns (integer codes): we skip here because mean/std is not meaningful.
    categorical_like = {"sex", "education", "marriage"}

    candidate_cols = [c for c in train_df.columns if c not in exclude]
    numeric_cols = [c for c in candidate_cols if c not in categorical_like]

    # Make sure the new batch has the same columns we plan to monitor
    missing_in_new = [c for c in numeric_cols if c not in new_df.columns]
    if missing_in_new:
        raise ValueError(f"New batch is missing columns required for drift monitoring: {missing_in_new}")

    rows = []
    eps = 1e-9
    k = args.k

    for c in numeric_cols:
        train_mean = float(train_df[c].mean())
        train_std = float(train_df[c].std(ddof=1))
        new_mean = float(new_df[c].mean())
        new_std = float(new_df[c].std(ddof=1))

        abs_diff = abs(new_mean - train_mean)
        z_like = abs_diff / (train_std + eps)

        drift_flag = abs_diff > k * train_std if train_std > 0 else abs_diff > 0

        rows.append({
            "feature": c,
            "mean_train": train_mean,
            "std_train": train_std,
            "mean_new": new_mean,
            "std_new": new_std,
            "abs_mean_diff": abs_diff,
            "z_like_std_units": z_like,
            "drift_flag": bool(drift_flag),
        })

    report = pd.DataFrame(rows).sort_values(by=["drift_flag", "z_like_std_units"], ascending=[False, False])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    n_flagged = int(report["drift_flag"].sum())
    print("Wrote:", out_path)
    print(f"Monitored {len(report)} numeric features. Flagged drift on {n_flagged} features (k={k}).")

    # Print top 10 most shifted for quick inspection
    print("\nTop shifts:")
    print(report.head(10)[["feature", "mean_train", "mean_new", "abs_mean_diff", "z_like_std_units", "drift_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()
