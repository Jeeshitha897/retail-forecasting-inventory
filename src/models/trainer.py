"""Model Trainer and Benchmarking Engine.
Executes head-to-head evaluation across XGBoost, LightGBM, and CatBoost.
Supervised by: Dr. P. Sudhakar
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.models.xgboost_model import XGBoostForecaster
from src.models.lightgbm_model import LightGBMForecaster
from src.models.catboost_model import CatBoostForecaster
from src.evaluation import evaluate_forecasts

class ModelBenchmarkTrainer:
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.models = {}
        self.quantile_models = {}  # Upper bound (P90) models for safety stock
        self.results = {}

    def train_and_benchmark(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "units_sold"
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Trains XGBoost, LightGBM, and CatBoost.
        Returns:
            - metrics_df: Comparative metrics summary (WAPE, RMSE, MAE, R2, Latency)
            - point_predictions: Dict of model_name -> test point predictions
            - p90_predictions: Dict of model_name -> test 90th percentile predictions
        """
        X_train = train_df[self.feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[self.feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[target_col]

        candidate_frameworks = {
            "LightGBM": LightGBMForecaster,
            "XGBoost": XGBoostForecaster,
            "CatBoost": CatBoostForecaster,
        }

        metrics_list = []
        point_preds = {}
        p90_preds = {}

        print("\n" + "=" * 70)
        print("GRADIENT BOOSTING BENCHMARK EXECUTION (XGBoost vs LightGBM vs CatBoost)")
        print("=" * 70)

        for name, model_class in candidate_frameworks.items():
            print(f"\n--> Training {name} Point Regressor...")
            start_time = time.time()
            point_model = model_class()
            point_model.fit(X_train, y_train, X_val, y_val)
            train_duration = time.time() - start_time

            # Point Predictions
            pred_start = time.time()
            y_pred = point_model.predict(X_test)
            inference_duration = time.time() - pred_start

            self.models[name] = point_model
            point_preds[name] = y_pred

            # Train P90 Quantile Regressor for Demand Risk
            print(f"--> Training {name} Quantile (tau=0.90) Regressor for Safety Stock...")
            p90_model = model_class(quantile=0.90)
            p90_model.fit(X_train, y_train, X_val, y_val)
            y_p90 = p90_model.predict(X_test)

            self.quantile_models[name] = p90_model
            p90_preds[name] = y_p90

            # Evaluate Metrics
            metrics = evaluate_forecasts(y_test.values, y_pred, y_p90=y_p90)
            metrics["Framework"] = name
            metrics["Train Time (s)"] = round(train_duration, 3)
            metrics["Inference Time (ms)"] = round(inference_duration * 1000, 2)
            metrics_list.append(metrics)

            print(f"    [Result] {name} | WAPE: {metrics['WAPE (%)']}% | RMSE: {metrics['RMSE']} | MAE: {metrics['MAE']} | R2: {metrics['R2']}")

        metrics_df = pd.DataFrame(metrics_list)
        # Reorder columns
        col_order = ["Framework", "WAPE (%)", "RMSE", "MAE", "SMAPE (%)", "R2", "Pinball Loss (P90)", "Train Time (s)", "Inference Time (ms)"]
        metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]].sort_values(by="WAPE (%)").reset_index(drop=True)

        return metrics_df, point_preds, p90_preds
