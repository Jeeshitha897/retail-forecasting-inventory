"""Interactive Streamlit Web Dashboard.
Retail Demand Forecasting with Seasonality & Dynamic Inventory Optimization.
Supervised by: Dr. P. Sudhakar | Domain: Predictive Modeling & Business Analytics

Run with:
    streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import config
from src.data_loader import load_retail_data
from src.inventory.optimizer import InventoryOptimizer

st.set_page_config(
    page_title="Retail Forecasting & Inventory Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title & Metadata
st.title("Retail Demand Forecasting & Inventory Optimization Dashboard")
st.caption("Predictive Modeling & Business Analytics | Supervised by: Dr. P. Sudhakar")

@st.cache_data
def get_data():
    df = load_retail_data()
    return df

df = get_data()

# Sidebar Controls
st.sidebar.header("Interactive Scenario Parameters")
selected_store = st.sidebar.selectbox("Select Store", df["store_id"].unique())
selected_sku = st.sidebar.selectbox("Select SKU", df["sku_id"].unique())

filtered_df = df[(df["store_id"] == selected_store) & (df["sku_id"] == selected_sku)].sort_values("date").reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Inventory Policy Controls")
target_sl = st.sidebar.slider("Target Service Level (Cycle SLA %)", min_value=80.0, max_value=99.9, value=95.0, step=0.5) / 100.0
lead_time_input = st.sidebar.slider("Supplier Lead Time (Days)", min_value=1, max_value=30, value=int(filtered_df["lead_time_days"].iloc[0]))
lead_time_std_input = st.sidebar.slider("Lead Time Std Dev (Days)", min_value=0.0, max_value=5.0, value=float(filtered_df["lead_time_std"].iloc[0]), step=0.1)
order_cost_input = st.sidebar.number_input("Order Placement Cost ($/order)", min_value=5.0, max_value=500.0, value=float(filtered_df["order_cost"].iloc[0]))

# Key Product Metadata
cogs = filtered_df["cogs_per_unit"].iloc[0]
price = filtered_df["selling_price"].iloc[0]
category = filtered_df["category"].iloc[0]

# Inventory Optimization Calculation
optimizer = InventoryOptimizer(service_level=target_sl)
daily_mean = filtered_df["units_sold"].mean()
daily_std = filtered_df["units_sold"].std()

policy = optimizer.calculate_sku_policy(
    sku_id=selected_sku,
    store_id=selected_store,
    forecast_daily_mean=daily_mean,
    forecast_daily_std=daily_std,
    unit_cogs=cogs,
    selling_price=price,
    lead_time_days=lead_time_input,
    lead_time_std=lead_time_std_input,
    order_cost=order_cost_input
)

# Top KPI row
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("SKU Category", category)
col2.metric("Safety Stock", f"{policy['safety_stock_units']} units")
col3.metric("Reorder Point (ROP)", f"{policy['reorder_point_units']} units")
col4.metric("Economic Order Qty (EOQ)", f"{policy['eoq_units']} units")
col5.metric("Avg Capital Locked", f"${policy['working_capital_locked ($)']:,}")

st.markdown("---")

# Main Visualizations
tab1, tab2, tab3 = st.tabs(["Demand & Seasonality Series", "Safety Stock & ROP Analysis", "Model Leaderboard"])

with tab1:
    st.subheader(f"Historical Demand Trend & Seasonality: {selected_store} - {selected_sku}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(filtered_df["date"], filtered_df["units_sold"], label="Daily Units Sold", color="#1f77b4", alpha=0.7, linewidth=1.2)
    ax.plot(filtered_df["date"], filtered_df["units_sold"].rolling(30).mean(), label="30-Day Moving Trend", color="#d62728", linewidth=2.0)
    ax.set_ylabel("Units")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    st.pyplot(fig)

with tab2:
    st.subheader("Inventory Policy Breakdown")
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.write("### Operational Parameters")
        st.write(f"- **Z-Score for {target_sl*100:.1f}% Service Level:** `{norm.ppf(target_sl):.3f}`")
        st.write(f"- **Expected Lead Time Demand:** `{daily_mean * lead_time_input:.1f}` units")
        st.write(f"- **Buffer Safety Stock ($SS$):** `{policy['safety_stock_units']}` units")
        st.write(f"- **Trigger Reorder Point ($ROP$):** `{policy['reorder_point_units']}` units")
        st.write(f"- **Batch Order Size ($EOQ$):** `{policy['eoq_units']}` units")

    with c_right:
        st.write("### Annual Cost Projections")
        st.write(f"- **Annual Order Cost:** `${policy['annual_order_cost ($)']:,}`")
        st.write(f"- **Annual Carrying/Holding Cost:** `${policy['annual_holding_cost ($)']:,}`")
        st.write(f"- **Total Annual Policy Cost:** `${policy['total_annual_policy_cost ($)']:,}`")

with tab3:
    st.subheader("Gradient Boosting Benchmarking Summary")
    results_path = config.RESULTS_PATH
    if results_path.exists():
        metrics_df = pd.read_csv(results_path)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.info("Run `python main.py` first to generate benchmark metrics across XGBoost, LightGBM, and CatBoost.")
