from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd


def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV and fix known formatting issues (like an extra header row).
    Return a dataframe with correct columns but not fully cleaned yet.
    """
    raise NotImplementedError


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning rules:
    - snake_case columns
    - numeric conversion
    - normalize category codes (education, marriage)
    - fill missing values (robustness)
    """
    raise NotImplementedError


def write_to_sqlite(df: pd.DataFrame, db_path: str | Path, table_name: str) -> None:
    """
    Write cleaned dataframe to SQLite (replace table if exists).
    """
    raise NotImplementedError


def main() -> None:
    """
    Orchestrate the ETL:
    raw -> clean -> save processed csv -> write sqlite
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()