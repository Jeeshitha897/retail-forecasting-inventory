"""LightGBM Demand Forecasting Wrapper.
Employs leaf-wise tree growth, Gradient-based One-Side Sampling (GOSS), and native categorical features.
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor

import config

class LightGBMForecaster:
    def __init__(self, params: Optional[Dict[str, Any]] = None, quantile: Optional[float] = None):
        self.quantile = quantile
        self.params = params or config.LGBM_PARAMS.copy()
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        if HAS_LGBM:
            lgb_kwargs = self.params.copy()
            if self.quantile is not None:
                lgb_kwargs["objective"] = "quantile"
                lgb_kwargs["alpha"] = self.quantile
            else:
                lgb_kwargs["objective"] = "regression"

            self.model = lgb.LGBMRegressor(**lgb_kwargs)
            callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)] if X_val is not None else None
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )
        else:
            loss = "quantile" if self.quantile is not None else "squared_error"
            quantile_val = self.quantile if self.quantile is not None else 0.5
            self.model = HistGradientBoostingRegressor(
                loss=loss,
                quantile=quantile_val,
                max_iter=self.params.get("n_estimators", 200),
                learning_rate=self.params.get("learning_rate", 0.05),
                random_state=config.RANDOM_SEED
            )
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.maximum(0, preds)
