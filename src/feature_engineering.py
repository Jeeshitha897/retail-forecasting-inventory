"""Feature Engineering Module for Retail Time-Series Demand Forecasting.
Extracts calendar seasonality, cyclical harmonics, historical lags, and rolling aggregations.
Guarantees zero data leakage by shifting historical aggregations strictly prior to forecast step.
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import LabelEncoder
import config

class RetailFeatureEngineer:
    def __init__(self, lags: List[int] = config.LAGS, rolling_windows: List[int] = config.ROLLING_WINDOWS):
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.categorical_cols = ["store_id", "store_type", "city", "sku_id", "category"]
        self.label_encoders = {}
        self.feature_names = []

    def create_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts temporal calendar and cyclical trigonometric seasonality features."""
        df = df.copy()
        dt = pd.to_datetime(df["date"])

        # Discrete calendar features
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["day_of_year"] = dt.dt.dayofyear
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["year"] = dt.dt.year
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical trigonometric transforms (preserves circular continuity)
        # 1. Day of week (period = 7)
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

        # 2. Month (period = 12)
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # 3. Day of Year (period = 365.25)
        df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

        # Pricing features
        df["price_ratio"] = df["selling_price"] / df["base_price"]

        return df

    def create_lag_and_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates historical demand lags and rolling window aggregates.
        All rolling windows are strictly computed on shift(1) to eliminate look-ahead leakage.
        """
        df = df.copy()
        df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

        grouped = df.groupby(["store_id", "sku_id"])["units_sold"]

        # Lag features
        for lag in self.lags:
            col_name = f"lag_{lag}"
            df[col_name] = grouped.shift(lag)

        # Rolling window features (computed on lag_1 to prevent current day leakage)
        lag1_grouped = df.groupby(["store_id", "sku_id"])["lag_1"]
        for window in self.rolling_windows:
            df[f"roll_mean_{window}"] = lag1_grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
            df[f"roll_std_{window}"] = lag1_grouped.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0)
            df[f"roll_min_{window}"] = lag1_grouped.transform(lambda s: s.rolling(window, min_periods=1).min())
            df[f"roll_max_{window}"] = lag1_grouped.transform(lambda s: s.rolling(window, min_periods=1).max())

        # Demand momentum / ratio: 7-day average vs 28-day baseline
        df["demand_momentum_7_28"] = (df["roll_mean_7"] / (df["roll_mean_28"] + 1e-5)).clip(0.1, 5.0)

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Fits label encoders on training categories and returns enriched dataframe."""
        df = self.create_calendar_features(df)
        df = self.create_lag_and_rolling_features(df)

        # Drop initial rows where maximum lag is NaN
        max_lag = max(self.lags)
        df = df.dropna(subset=[f"lag_{max_lag}"]).reset_index(drop=True)

        # Fit Label Encoders for categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        self.feature_names = self._get_feature_columns(df)
        return df, self.feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation/test datasets using fitted feature logic."""
        df = self.create_calendar_features(df)
        df = self.create_lag_and_rolling_features(df)

        # Apply fitted label encoders
        for col, le in self.label_encoders.items():
            if col in df.columns:
                mapping = dict(zip(le.classes_, range(len(le.classes_))))
                df[f"{col}_encoded"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        return df

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identifies engineered predictor columns excluding targets and raw identifiers."""
        exclude_cols = {
            "date", "units_sold", "revenue", "gross_profit", "lead_time_days",
            "lead_time_std", "order_cost", "cogs_per_unit"
        }
        # Also exclude raw string categorical columns (encoded versions are used)
        exclude_cols.update(self.categorical_cols)

        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols
