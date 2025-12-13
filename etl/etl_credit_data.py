from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import sys
import pandas as pd
import numpy as np
import yaml


# ----------------------------
# Configuration (from YAML)
# ----------------------------

@dataclass(frozen=True)
class Settings:
    raw_csv_path: Path
    processed_csv_path: Path
    sqlite_path: Path
    table_name: str
    id_column: str
    target_column: str


def load_settings(project_root: Path) -> Settings:
    """
    Load config/settings_example.yaml and convert relative paths to absolute Paths.
    """
    cfg_path = project_root / "config" / "settings_example.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {cfg_path}\n"
            "Create config/settings_example.yaml as described in Block 2."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv_path = project_root / cfg["dataset"]["raw_csv_path"]
    sqlite_path = project_root / cfg["database"]["sqlite_path"]
    table_name = cfg["database"]["table_name"]

    id_column = cfg["columns"]["id_column"]
    target_column = cfg["columns"]["target_column"]

    processed_csv_path = project_root / "data" / "processed" / "credit_default_clean.csv"

    return Settings(
        raw_csv_path=raw_csv_path,
        processed_csv_path=processed_csv_path,
        sqlite_path=sqlite_path,
        table_name=table_name,
        id_column=id_column,
        target_column=target_column,
    )


# ----------------------------
# Helpers: column name cleanup
# ----------------------------

def to_snake_case(name: str) -> str:
    """
    Normalize column names to snake_case:
      - trim
      - lowercase
      - spaces -> underscores
    """
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def fix_double_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some versions of this dataset store a second header row as the first data row.
    Example: first row contains ['ID', 'LIMIT_BAL', 'SEX', ...].

    If detected, we:
      - set df.columns = first_row
      - drop that row
    """
    if df.empty:
        return df

    first_row = df.iloc[0].astype(str).tolist()
    markers = {"ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0"}
    if any(m in first_row for m in markers):
        df2 = df.iloc[1:].copy()
        df2.columns = first_row
        return df2

    return df.copy()


# ----------------------------
# ETL Core
# ----------------------------

def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame and fix known formatting issues
    (e.g., double header row, stray unnamed index column).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")

    df = pd.read_csv(path)
    df = fix_double_header(df)

    # Drop common junk column from CSV exports, if present
    junk_cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)

    return df


def clean_data(df: pd.DataFrame, *, id_column: str, target_column: str) -> pd.DataFrame:
    """
    Clean the dataframe according to our Block 2 ETL rules.

    IMPORTANT DESIGN CHOICE:
    - We do NOT one-hot encode categories in ETL.
      We keep cleaned integer-coded categories and do one-hot encoding inside the
      model training pipeline (scikit-learn ColumnTransformer).
      This is more realistic and avoids "baking in" training-time decisions into the data layer.
    """
    df = df.copy()

    # 1) Normalize column names
    df.columns = [to_snake_case(c) for c in df.columns]

    # 2) Coerce everything to numeric (this dataset should be all numeric after header fix)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Ensure required columns exist
    required = {id_column, target_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after cleaning: {missing}")

    # 4) Enforce target as {0,1} integer
    # (Usually already 0/1, but we ensure it.)
    df[target_column] = df[target_column].astype("Int64")  # allows NA temporarily

    # If any weird target values exist, fail loudly (better than silently corrupting labels).
    bad_targets = df.loc[~df[target_column].isin([0, 1]) & df[target_column].notna(), target_column].unique()
    if len(bad_targets) > 0:
        raise ValueError(f"Found non-binary target values: {bad_targets}")

    # 5) Category normalization based on what we saw in EDA:
    # education: documented main categories 1..4, but data often includes 0,5,6.
    # We'll map {0,5,6} -> 4 (other/unknown) to keep a clean compact set.
    if "education" in df.columns:
        df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})

    # marriage: documented 1..3, but data includes 0.
    # We'll map 0 -> 3 (other/unknown).
    if "marriage" in df.columns:
        df["marriage"] = df["marriage"].replace({0: 3})

    # 6) Basic row-level sanity checks (optional but helpful)
    # - Age should be positive and not absurd.
    # - Sex should be in {1,2}.
    # Instead of dropping aggressively, we can just flag & drop extreme nonsense.
    if "age" in df.columns:
        df = df[(df["age"].isna()) | ((df["age"] >= 18) & (df["age"] <= 100))]

    if "sex" in df.columns:
        df = df[df["sex"].isin([1, 2]) | df["sex"].isna()]

    # 7) Handle missing values (robustness)
    # The README suggests no missing values, but after coercion we might get NaNs if
    # something was malformed. We'll fill:
    # - numeric columns -> median
    # - categorical-like integer-coded columns -> mode
    cat_like = [c for c in ["sex", "education", "marriage"] if c in df.columns]

    for c in df.columns:
        if df[c].isna().any():
            if c in cat_like:
                # mode might return multiple values; take the first
                mode_val = df[c].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else 0
                df[c] = df[c].fillna(fill_val)
            else:
                median_val = df[c].median()
                df[c] = df[c].fillna(median_val)

    # 8) Make sure ID is integer-like (safe for DB keys)
    df[id_column] = df[id_column].astype(int)

    # 9) Final: ensure target is int
    df[target_column] = df[target_column].astype(int)

    # Optional: sort by ID for stability (nice for reproducibility)
    df = df.sort_values(by=id_column).reset_index(drop=True)

    return df


def write_to_sqlite(df: pd.DataFrame, db_path: str | Path, table_name: str) -> None:
    """
    Write dataframe to SQLite.
    - if_exists='replace' means rerunning ETL replaces the table cleanly.
    """
    db_path = Path(db_path)
    con = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, con, if_exists="replace", index=False)
    finally:
        con.close()


def main() -> None:
    """
    Orchestrate ETL:
      raw CSV -> cleaned dataframe -> processed CSV -> SQLite table
    """
    project_root = Path(__file__).resolve().parents[1]  # .../risk-scoring-system
    settings = load_settings(project_root)

    print("=== Risk Scoring ETL ===")
    print("Project root:", project_root)
    print("Raw CSV:", settings.raw_csv_path)
    print("Processed CSV:", settings.processed_csv_path)
    print("SQLite DB:", settings.sqlite_path)
    print("SQLite table:", settings.table_name)
    print()

    # Ensure output dirs exist
    settings.processed_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load + clean
    raw_df = load_raw(settings.raw_csv_path)
    print("Loaded raw shape:", raw_df.shape)

    clean_df = clean_data(raw_df, id_column=settings.id_column, target_column=settings.target_column)
    print("Cleaned shape:", clean_df.shape)

    # Save processed CSV
    clean_df.to_csv(settings.processed_csv_path, index=False)
    print("Wrote processed CSV.")

    # Save to SQLite
    write_to_sqlite(clean_df, settings.sqlite_path, settings.table_name)
    print("Wrote SQLite table.")

    # Quick summary
    target_counts = clean_df[settings.target_column].value_counts().to_dict()
    print("\nTarget distribution:", target_counts)
    print("ETL complete ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nETL failed ❌")
        print(type(e).__name__ + ":", e)
        sys.exit(1)