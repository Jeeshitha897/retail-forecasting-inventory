"""Main Orchestration Script: End-to-End Retail Sales Forecasting & Inventory Optimization.
Supervised by: Dr. P. Sudhakar
Domain: Predictive Modeling & Business Analytics

Execution Pipeline:
1. Synthetic Retail Dataset Ingestion / Generation
2. Seasonality & Lag Feature Engineering
3. Temporal Chronological Train / Val / Test Partition
4. Tri-Framework Gradient Boosting Benchmark (XGBoost, LightGBM, CatBoost)
5. Quantile Forecasting & Uncertainty Estimation
6. Prescriptive Inventory Optimization (Dynamic SS, ROP, EOQ)
7. Operational Inventory Simulation & Financial Impact Analysis
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Setup local imports
sys.path.append(str(Path(__file__).resolve().parent))

import config
from src.data_loader import load_retail_data, temporal_train_val_test_split
from src.feature_engineering import RetailFeatureEngineer
from src.models.trainer import ModelBenchmarkTrainer
from src.inventory.optimizer import InventoryOptimizer
from src.inventory.simulation import InventorySimulator

def run_pipeline():
    print("=" * 80)
    print("RETAIL SALES AND REVENUE FORECASTING WITH SEASONALITY & INVENTORY OPTIMIZATION")
    print("Supervisor: Dr. P. Sudhakar | Domain: Predictive Modeling & Business Analytics")
    print("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Data Ingestion
    # -------------------------------------------------------------
    print("\n[STEP 1/6] Ingesting Retail Sales Dataset...")
    raw_df = load_retail_data()
    print(f"Total Transactions Loaded: {len(raw_df):,} records.")
    print(f"Unique Stores: {raw_df['store_id'].nunique()} | Unique SKUs: {raw_df['sku_id'].nunique()}")

    # -------------------------------------------------------------
    # Step 2: Temporal Split
    # -------------------------------------------------------------
    print("\n[STEP 2/6] Executing Chronological Train / Validation / Test Partition...")
    train_raw, val_raw, test_raw = temporal_train_val_test_split(raw_df)

    # -------------------------------------------------------------
    # Step 3: Feature Engineering
    # -------------------------------------------------------------
    print("\n[STEP 3/6] Engineering Seasonality, Harmonics, Historical Lags & Rolling Stats...")
    fe = RetailFeatureEngineer()
    train_df, feature_cols = fe.fit_transform(train_raw)
    val_df = fe.transform(val_raw).dropna(subset=[f"lag_{max(config.LAGS)}"]).reset_index(drop=True)
    test_df = fe.transform(test_raw).dropna(subset=[f"lag_{max(config.LAGS)}"]).reset_index(drop=True)

    print(f"Engineered Predictor Count: {len(feature_cols)} features.")
    print(f"Feature Sample: {feature_cols[:8]}...")

    # -------------------------------------------------------------
    # Step 4: Tri-GBDT Model Benchmarking
    # -------------------------------------------------------------
    print("\n[STEP 4/6] Benchmarking XGBoost vs LightGBM vs CatBoost (Point & P90 Quantiles)...")
    trainer = ModelBenchmarkTrainer(feature_cols=feature_cols)
    metrics_df, point_preds, p90_preds = trainer.train_and_benchmark(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_col="units_sold"
    )

    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE LEADERBOARD (Sorted by WAPE):")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    # Champion model selection
    champion_name = metrics_df.iloc[0]["Framework"]
    print(f"\n--> Champion Forecasting Model Selected: {champion_name} (Lowest WAPE: {metrics_df.iloc[0]['WAPE (%)']}%)")

    # Save metrics
    metrics_df.to_csv(config.RESULTS_PATH, index=False)
    print(f"Saved benchmark results to: {config.RESULTS_PATH}")

    # Attach champion predictions to test set
    test_results_df = test_df.copy()
    test_results_df["forecast_units"] = point_preds[champion_name]
    test_results_df["forecast_units_p90"] = p90_preds[champion_name]

    # -------------------------------------------------------------
    # Step 5: Prescriptive Inventory Optimization
    # -------------------------------------------------------------
    print("\n[STEP 5/6] Prescriptive Inventory Optimization (Safety Stock, ROP, EOQ)...")
    optimizer = InventoryOptimizer(service_level=config.DEFAULT_SERVICE_LEVEL)
    policy_df = optimizer.optimize_fleet(test_results_df)

    print("\n" + "=" * 80)
    print(f"OPTIMAL INVENTORY POLICY (Target Service Level: {config.DEFAULT_SERVICE_LEVEL * 100}%):")
    print("=" * 80)
    display_cols = ["store_id", "sku_id", "daily_mean_demand", "daily_std_error", "safety_stock_units", "reorder_point_units", "eoq_units", "working_capital_locked ($)"]
    print(policy_df[display_cols].to_string(index=False))

    policy_df.to_csv(config.INVENTORY_REPORT_PATH, index=False)
    print(f"\nSaved inventory policy report to: {config.INVENTORY_REPORT_PATH}")

    # -------------------------------------------------------------
    # Step 6: Discrete-Event Operational Simulation & Financial Impact
    # -------------------------------------------------------------
    print("\n[STEP 6/6] Running Discrete-Event Inventory Simulation (GBDT vs Baseline)...")
    simulator = InventorySimulator()
    sim_summary_df, fleet_financials = simulator.run_comparative_simulation(
        test_df=test_results_df,
        dynamic_policies_df=policy_df
    )

    print("\n" + "=" * 80)
    print("OPERATIONAL INVENTORY SIMULATION RESULTS:")
    print("=" * 80)
    print(sim_summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("EXECUTIVE SUPPLY CHAIN FINANCIAL IMPACT:")
    print("=" * 80)
    for k, v in fleet_financials.items():
        print(f"  {k:35s}: {v}")
    print("=" * 80)
    print("\nEnd-to-End Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    run_pipeline()
