"""CatBoost Demand Forecasting Wrapper.
Employs ordered boosting and symmetric oblivious decision trees to prevent target leakage.
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    from sklearn.ensemble import HistGradientBoostingRegressor

import config

class CatBoostForecaster:
    def __init__(self, params: Optional[Dict[str, Any]] = None, quantile: Optional[float] = None):
        self.quantile = quantile
        self.params = params or config.CATBOOST_PARAMS.copy()
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        if HAS_CATBOOST:
            cb_kwargs = self.params.copy()
            if self.quantile is not None:
                cb_kwargs["loss_function"] = f"Quantile:alpha={self.quantile}"
                cb_kwargs["eval_metric"] = f"Quantile:alpha={self.quantile}"
            else:
                cb_kwargs["loss_function"] = "RMSE"
                cb_kwargs["eval_metric"] = "RMSE"

            self.model = cb.CatBoostRegressor(**cb_kwargs)
            eval_set = (X_val, y_val) if X_val is not None and y_val is not None else None

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            loss = "quantile" if self.quantile is not None else "squared_error"
            quantile_val = self.quantile if self.quantile is not None else 0.5
            self.model = HistGradientBoostingRegressor(
                loss=loss,
                quantile=quantile_val,
                max_iter=self.params.get("iterations", 200),
                learning_rate=self.params.get("learning_rate", 0.05),
                random_state=config.RANDOM_SEED
            )
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(0, preds)
