"""Configuration Module for Sales & Revenue Forecasting and Inventory Optimization.
Supervised by: Dr. P. Sudhakar
Domain: Predictive Modeling & Business Analytics
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "retail_sales_multiskustore.csv"
RESULTS_PATH = OUTPUT_DIR / "model_evaluation_metrics.csv"
INVENTORY_REPORT_PATH = OUTPUT_DIR / "inventory_optimization_report.csv"

# Dataset Generation Settings
RANDOM_SEED = 42
NUM_STORES = 3
NUM_SKUS = 5
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"  # 3 full years of daily sales

# Time Series Forecasting Horizon
FORECAST_HORIZON_DAYS = 30  # Multi-step ahead demand planning
VALIDATION_DAYS = 60
TEST_DAYS = 90

# Feature Engineering Parameters
LAGS = [1, 2, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Inventory Optimization Parameters
SERVICE_LEVELS = {
    "Target_90": 0.90,
    "Target_95": 0.95,
    "Target_99": 0.99,
}
DEFAULT_SERVICE_LEVEL = 0.95

# Default Costs and Lead Times (Configurable per SKU)
DEFAULT_SUPPLIER_LEAD_TIME_DAYS = 5
DEFAULT_SUPPLIER_LEAD_TIME_STD = 1.2
DEFAULT_ORDERING_COST = 50.0       # $ per purchase order (S)
ANNUAL_HOLDING_COST_RATE = 0.20    # 20% of unit cost per year (h)
STOCKOUT_PENALTY_MULTIPLIER = 2.5  # Stockout penalty = 2.5x lost gross margin

# Gradient Boosting Hyperparameter Grids
XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LGBM_PARAMS = {
    "n_estimators": 250,
    "max_depth": 7,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

CATBOOST_PARAMS = {
    "iterations": 250,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
}
