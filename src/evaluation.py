"""Forecast Evaluation Module.
Computes retail-specific supply chain accuracy metrics: WAPE, RMSE, MAE, SMAPE, R2, and Pinball Loss.
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE).
    Gold standard in retail inventory planning (immune to zero-sales distortions).
    Formula: sum(|y - y_hat|) / sum(y) * 100%
    """
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    # Handle zero denominator safely
    valid = denominator > 1e-6
    if not np.any(valid):
        return 0.0
    return float(np.mean(diff[valid] / denominator[valid]) * 100.0)

def compute_pinball_loss(y_true: np.ndarray, y_pred_quantile: np.ndarray, tau: float = 0.90) -> float:
    """Pinball (Quantile) loss for asymmetric demand uncertainty."""
    err = y_true - y_pred_quantile
    loss = np.maximum(tau * err, (tau - 1.0) * err)
    return float(np.mean(loss))

def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_p90: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Generates comprehensive accuracy score dictionary."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    wape = compute_wape(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = compute_smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "WAPE (%)": round(wape, 2),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "SMAPE (%)": round(smape, 2),
        "R2": round(float(r2), 4),
    }

    if y_p90 is not None:
        p90_loss = compute_pinball_loss(y_true, y_p90, tau=0.90)
        metrics["Pinball Loss (P90)"] = round(p90_loss, 3)

    return metrics
