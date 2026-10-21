"""Dynamic Inventory Optimization Engine.
Translates gradient boosting demand forecasts into actionable operational inventory policies:
Safety Stock (SS), Reorder Point (ROP), and Economic Order Quantity (EOQ).
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any
import config

class InventoryOptimizer:
    def __init__(
        self,
        service_level: float = config.DEFAULT_SERVICE_LEVEL,
        holding_cost_rate: float = config.ANNUAL_HOLDING_COST_RATE,
        ordering_cost: float = config.DEFAULT_ORDERING_COST,
        stockout_penalty_mult: float = config.STOCKOUT_PENALTY_MULTIPLIER
    ):
        self.service_level = service_level
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost
        self.stockout_penalty_mult = stockout_penalty_mult
        self.z_score = norm.ppf(service_level)

    def calculate_sku_policy(
        self,
        sku_id: str,
        store_id: str,
        forecast_daily_mean: float,
        forecast_daily_std: float,
        unit_cogs: float,
        selling_price: float,
        lead_time_days: float = config.DEFAULT_SUPPLIER_LEAD_TIME_DAYS,
        lead_time_std: float = config.DEFAULT_SUPPLIER_LEAD_TIME_STD,
        order_cost: float = config.DEFAULT_ORDERING_COST
    ) -> Dict[str, Any]:
        """Calculates operational inventory policy for a single Store-SKU combination."""
        # 1. Annual demand & holding cost
        annual_demand = forecast_daily_mean * 365.0
        unit_holding_cost_annual = unit_cogs * self.holding_cost_rate

        # 2. Safety Stock (accounting for both demand variance and supplier lead time variance)
        # Variance during lead time: Var(LT_Demand) = L * sigma_D^2 + D^2 * sigma_L^2
        lt_variance = (lead_time_days * (forecast_daily_std ** 2)) + ((forecast_daily_mean ** 2) * (lead_time_std ** 2))
        lt_sigma = np.sqrt(np.maximum(1e-4, lt_variance))
        safety_stock = int(np.ceil(self.z_score * lt_sigma))

        # 3. Dynamic Reorder Point (ROP) = Lead Time Demand + Safety Stock
        lead_time_demand = forecast_daily_mean * lead_time_days
        reorder_point = int(np.ceil(lead_time_demand + safety_stock))

        # 4. Economic Order Quantity (EOQ)
        if unit_holding_cost_annual > 0 and annual_demand > 0:
            eoq = int(np.ceil(np.sqrt((2.0 * annual_demand * order_cost) / unit_holding_cost_annual)))
        else:
            eoq = int(np.ceil(forecast_daily_mean * 14))

        # 5. Expected Financials
        annual_order_cost = (annual_demand / max(1, eoq)) * order_cost
        average_inventory = (eoq / 2.0) + safety_stock
        annual_holding_cost = average_inventory * unit_holding_cost_annual
        total_annual_inventory_cost = annual_order_cost + annual_holding_cost

        return {
            "store_id": store_id,
            "sku_id": sku_id,
            "service_level": self.service_level,
            "z_score": round(float(self.z_score), 3),
            "daily_mean_demand": round(float(forecast_daily_mean), 2),
            "daily_std_error": round(float(forecast_daily_std), 2),
            "lead_time_days": lead_time_days,
            "safety_stock_units": safety_stock,
            "reorder_point_units": reorder_point,
            "eoq_units": eoq,
            "average_inventory_units": round(float(average_inventory), 1),
            "working_capital_locked ($)": round(float(average_inventory * unit_cogs), 2),
            "annual_order_cost ($)": round(float(annual_order_cost), 2),
            "annual_holding_cost ($)": round(float(annual_holding_cost), 2),
            "total_annual_policy_cost ($)": round(float(total_annual_inventory_cost), 2),
        }

    def optimize_fleet(
        self,
        df_forecasts: pd.DataFrame,
        actual_col: str = "units_sold",
        pred_col: str = "forecast_units"
    ) -> pd.DataFrame:
        """Computes inventory policies across all Store-SKU groups using model residuals."""
        policies = []

        grouped = df_forecasts.groupby(["store_id", "sku_id"])

        for (store_id, sku_id), group in grouped:
            daily_mean = group[pred_col].mean()
            # Empirical forecast error standard deviation (sigma_e = std(actual - pred))
            residuals = group[actual_col] - group[pred_col]
            daily_std = residuals.std()
            if pd.isna(daily_std) or daily_std < 0.1:
                daily_std = group[pred_col].std() or 1.0

            unit_cogs = group["cogs_per_unit"].iloc[0]
            selling_price = group["selling_price"].iloc[0]
            lead_time_days = group["lead_time_days"].iloc[0]
            lead_time_std = group["lead_time_std"].iloc[0]
            order_cost = group["order_cost"].iloc[0]

            pol = self.calculate_sku_policy(
                sku_id=sku_id,
                store_id=store_id,
                forecast_daily_mean=daily_mean,
                forecast_daily_std=daily_std,
                unit_cogs=unit_cogs,
                selling_price=selling_price,
                lead_time_days=lead_time_days,
                lead_time_std=lead_time_std,
                order_cost=order_cost
            )
            policies.append(pol)

        return pd.DataFrame(policies)
