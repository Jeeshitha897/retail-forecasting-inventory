"""Data Generation Module: Multi-Store, Multi-SKU Retail Sales Simulation.
Simulates realistic daily transactions with trend, seasonality, promotions, and pricing.
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from pathlib import Path
import config

def generate_retail_sales_dataset(
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    random_seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    """Generates synthetic multi-store, multi-SKU retail demand time series."""
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(date_range)

    # Store definitions
    stores = [
        {"store_id": "STORE_01", "store_type": "Flagship", "city": "Metro_North", "traffic_factor": 1.45},
        {"store_id": "STORE_02", "store_type": "Supermarket", "city": "Urban_Center", "traffic_factor": 1.15},
        {"store_id": "STORE_03", "store_type": "Express", "city": "Suburban_South", "traffic_factor": 0.85},
    ]

    # SKU / Product definitions
    skus = [
        {
            "sku_id": "SKU_GROC_01",
            "category": "Packaged Goods",
            "base_price": 4.50,
            "cogs": 2.70,
            "base_demand": 120.0,
            "trend_rate": 0.0003,
            "price_elasticity": -1.2,
            "lead_time_days": 3,
            "lead_time_std": 0.8,
            "order_cost": 30.0,
            "q4_lift": 1.15
        },
        {
            "sku_id": "SKU_ELEC_02",
            "category": "Consumer Electronics",
            "base_price": 149.00,
            "cogs": 98.00,
            "base_demand": 18.0,
            "trend_rate": 0.0005,
            "price_elasticity": -2.1,
            "lead_time_days": 10,
            "lead_time_std": 2.5,
            "order_cost": 120.0,
            "q4_lift": 2.20  # Massive Black Friday / Holiday surge
        },
        {
            "sku_id": "SKU_APPR_03",
            "category": "Apparel",
            "base_price": 38.00,
            "cogs": 19.00,
            "base_demand": 45.0,
            "trend_rate": 0.0002,
            "price_elasticity": -1.6,
            "lead_time_days": 7,
            "lead_time_std": 1.5,
            "order_cost": 65.0,
            "q4_lift": 1.65
        },
        {
            "sku_id": "SKU_BEVG_04",
            "category": "Beverages",
            "base_price": 2.80,
            "cogs": 1.40,
            "base_demand": 160.0,
            "trend_rate": 0.0001,
            "price_elasticity": -0.9,
            "lead_time_days": 2,
            "lead_time_std": 0.5,
            "order_cost": 25.0,
            "q4_lift": 1.25  # Summer & Holidays
        },
        {
            "sku_id": "SKU_HOME_05",
            "category": "Home Essentials",
            "base_price": 24.00,
            "cogs": 13.50,
            "base_demand": 32.0,
            "trend_rate": 0.0003,
            "price_elasticity": -1.3,
            "lead_time_days": 6,
            "lead_time_std": 1.2,
            "order_cost": 50.0,
            "q4_lift": 1.40
        },
    ]

    records = []

    for store in stores:
        for sku in skus:
            # Baseline parameters
            base_vol = sku["base_demand"] * store["traffic_factor"]
            trend = 1.0 + (sku["trend_rate"] * np.arange(n_days))

            # Weekly seasonality (Mon=0, Tue=1, ..., Sun=6)
            # Weekend surge: Fri=1.1, Sat=1.45, Sun=1.35, Mon=0.85
            day_of_week = date_range.dayofweek
            weekly_mult = np.array([0.85, 0.90, 0.95, 1.00, 1.15, 1.45, 1.30])[day_of_week]

            # Annual seasonality (Sine wave peaking in Nov/Dec, plus summer peak for beverages)
            day_of_year = date_range.dayofyear
            annual_cycle = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

            # Q4 holiday season uplift (Nov 15 to Dec 31)
            is_q4_holiday = ((date_range.month == 11) & (date_range.day >= 15)) | (date_range.month == 12)
            q4_multiplier = np.where(is_q4_holiday, sku["q4_lift"], 1.0)

            # Random marketing promotional campaigns (~8% of days)
            promo_prob = 0.08
            is_promo = np.random.binomial(1, promo_prob, size=n_days)
            # Continuous promo streak smoothing (promos usually last 3-5 days)
            is_promo = pd.Series(is_promo).rolling(window=3, min_periods=1).max().astype(int).values

            discount_pct = np.where(is_promo == 1, np.random.choice([0.10, 0.15, 0.20, 0.25], size=n_days), 0.0)
            selling_price = np.round(sku["base_price"] * (1.0 - discount_pct), 2)

            # Price elasticity impact: (P / P_0) ^ elasticity
            price_ratio = selling_price / sku["base_price"]
            elasticity_lift = price_ratio ** sku["price_elasticity"]

            # Composite expected demand
            mu_demand = (
                base_vol
                * trend
                * weekly_mult
                * annual_cycle
                * q4_multiplier
                * elasticity_lift
            )

            # Realistic Poisson/Negative Binomial noise
            noise = np.random.normal(loc=0, scale=0.12 * mu_demand, size=n_days)
            units_sold = np.maximum(0, np.round(mu_demand + noise)).astype(int)

            # Revenue & Profit
            revenue = np.round(units_sold * selling_price, 2)
            total_cogs = np.round(units_sold * sku["cogs"], 2)
            gross_profit = np.round(revenue - total_cogs, 2)

            for d_idx, dt in enumerate(date_range):
                records.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "store_id": store["store_id"],
                    "store_type": store["store_type"],
                    "city": store["city"],
                    "sku_id": sku["sku_id"],
                    "category": sku["category"],
                    "selling_price": selling_price[d_idx],
                    "base_price": sku["base_price"],
                    "cogs_per_unit": sku["cogs"],
                    "is_promo": is_promo[d_idx],
                    "discount_pct": discount_pct[d_idx],
                    "units_sold": units_sold[d_idx],
                    "revenue": revenue[d_idx],
                    "gross_profit": gross_profit[d_idx],
                    "lead_time_days": sku["lead_time_days"],
                    "lead_time_std": sku["lead_time_std"],
                    "order_cost": sku["order_cost"]
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store_id", "sku_id", "date"]).reset_index(drop=True)

    # Save to disk
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PATH, index=False)
    print(f"Generated retail sales dataset: {len(df):,} rows spanning {start_date} to {end_date}.")
    print(f"Saved to: {config.DATA_PATH}")
    return df

if __name__ == "__main__":
    generate_retail_sales_dataset()
