"""Data Loader Module
Loads, validates, and partitions the multi-store retail dataset.
Supervised by: Dr. P. Sudhakar
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import config
from data.generate_synthetic_data import generate_retail_sales_dataset

def load_retail_data(data_path: Path = config.DATA_PATH, auto_generate: bool = True) -> pd.DataFrame:
    """Loads retail sales data. Automatically generates if file does not exist."""
    if not data_path.exists():
        if auto_generate:
            print(f"Data file not found at {data_path}. Generating realistic synthetic dataset...")
            return generate_retail_sales_dataset()
        else:
            raise FileNotFoundError(f"Data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    return df

def temporal_train_val_test_split(
    df: pd.DataFrame,
    test_days: int = config.TEST_DAYS,
    val_days: int = config.VALIDATION_DAYS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits time series data temporally to preserve chronological integrity.
    Prevents look-ahead bias and data leakage.
    """
    max_date = df["date"].max()
    test_cutoff = max_date - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train_df = df[df["date"] <= val_cutoff].copy()
    val_df = df[(df["date"] > val_cutoff) & (df["date"] <= test_cutoff)].copy()
    test_df = df[df["date"] > test_cutoff].copy()

    print("=" * 60)
    print("CHRONOLOGICAL TEMPORAL SPLIT SUMMARY:")
    print(f"Train Set : {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df):,} records)")
    print(f"Val Set   : {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df):,} records)")
    print(f"Test Set  : {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df):,} records)")
    print("=" * 60)

    return train_df, val_df, test_df
