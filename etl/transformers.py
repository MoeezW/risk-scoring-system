from __future__ import annotations

from pathlib import Path
import pandas as pd


def to_snake_case(name: str) -> str:
    name = str(name).strip().lower()
    name = name.replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def fix_double_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some dataset copies store a second header row as the first data row.
    Detect that and convert it into real column headers.
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


def load_raw_credit_csv(path: str | Path) -> pd.DataFrame:
    """
    Load raw credit CSV robustly:
      - handles double header row
      - removes stray 'Unnamed: 0' columns from exports
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    df = fix_double_header(df)

    # Drop common junk "Unnamed: 0" export columns, if present
    junk_cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    if junk_cols:
        df = df.drop(columns=junk_cols)

    return df


def clean_credit_applications(
    df: pd.DataFrame,
    *,
    id_column: str = "id",
    target_column: str = "default_payment_next_month",
    require_id: bool = False,
) -> pd.DataFrame:
    """
    Clean credit application data using the same rules as ETL.

    Key design:
    - Do NOT one-hot encode here.
      The saved model is a Pipeline that contains OneHotEncoder + scaler already.
      Our job is to ensure columns exist, are numeric, and category codes are normalized.

    Parameters:
      require_id:
        - if True, raise if id_column missing
        - if False, we'll still score, but you'll lose stable identifiers
    """
    df = df.copy()

    # 1) Standardize column names
    df.columns = [to_snake_case(c) for c in df.columns]

    # 2) Convert everything numeric (coerce errors -> NaN)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) Category normalization seen in EDA
    if "education" in df.columns:
        df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    if "marriage" in df.columns:
        df["marriage"] = df["marriage"].replace({0: 3})

    # 4) Simple sanity filters (same as ETL)
    if "age" in df.columns:
        df = df[(df["age"].isna()) | ((df["age"] >= 18) & (df["age"] <= 100))]

    if "sex" in df.columns:
        df = df[df["sex"].isin([1, 2]) | df["sex"].isna()]

    # 5) Fill missing values robustly
    cat_like = [c for c in ["sex", "education", "marriage"] if c in df.columns]
    for c in df.columns:
        if df[c].isna().any():
            if c in cat_like:
                mode_val = df[c].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else 0
                df[c] = df[c].fillna(fill_val)
            else:
                df[c] = df[c].fillna(df[c].median())

    # 6) ID handling (optional for scoring)
    if id_column not in df.columns:
        if require_id:
            raise ValueError(f"Missing required id column: '{id_column}'")
        # If no ID, create a simple surrogate ID for output tracking
        df[id_column] = range(1, len(df) + 1)
    df[id_column] = df[id_column].astype(int)

    # 7) If target is present, enforce binary (but scoring doesn’t require it)
    if target_column in df.columns:
        bad = df.loc[~df[target_column].isin([0, 1]), target_column].unique()
        if len(bad) > 0:
            raise ValueError(f"Found non-binary target values in '{target_column}': {bad}")
        df[target_column] = df[target_column].astype(int)

    # stable ordering
    df = df.sort_values(by=id_column).reset_index(drop=True)
    return df
