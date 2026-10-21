# Retail Sales and Revenue Forecasting with Seasonality via Gradient Boosting Frameworks

### Applied to Retail Demand Planning and Inventory Optimization
**Domain:** Predictive Modeling & Business Analytics  
**Faculty Guide / Mentor:** Dr. P. Sudhakar  

---

## 1. Project Overview

Accurate sales and revenue forecasting is fundamental to omni-channel retail operations. Inaccurate forecasts trigger either **stockouts** (resulting in lost sales, customer dissatisfaction, and brand degradation) or **excess inventory** (leading to bloated working capital, high holding costs, and markdowns/spoilage).

This project develops an end-to-end predictive and prescriptive analytics architecture that:
1. **Models Multi-Scale Seasonality & Demand Drivers:** Leverages daily transaction patterns, weekly cycles, annual trends, promotional campaigns, and price elasticity across multi-store, multi-SKU retail environments.
2. **Tri-GBDT Framework Benchmarking:** Systematically compares **XGBoost**, **LightGBM**, and **CatBoost** across both **Point Forecasting** (RMSE, MAE, WAPE) and **Quantile Forecasting** ($\tau = 0.90$ Pinball Loss).
3. **Prescriptive Inventory Optimization:** Translates demand forecasts and residual standard deviations ($\sigma_e$) directly into optimal supply chain control parameters:
   - **Dynamic Safety Stock ($SS$)**
   - **Reorder Point ($ROP$)**
   - **Economic Order Quantity ($EOQ$)**
4. **Operational Simulation:** Runs a discrete-event inventory simulation proving financial savings and service level improvements ($\alpha \ge 95\%$) over standard static rule-of-thumb retail policies.

---

## 2. Directory Structure

```text
retail_forecasting_inventory/
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Dependencies (XGBoost, LightGBM, CatBoost, etc.)
├── config.py                      # Global parameters, paths, and hyperparameters
├── main.py                        # Master pipeline execution script
├── app_streamlit.py               # Interactive Streamlit analytics dashboard
├── data/
│   ├── generate_synthetic_data.py # Realistic multi-store/SKU retail data simulator
│   └── retail_sales_multiskustore.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Temporal train/val/test partitioning
│   ├── feature_engineering.py     # Calendar, harmonics, lags, and rolling stats
│   ├── evaluation.py              # WAPE, RMSE, MAE, SMAPE, Pinball Loss
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py       # Histogram-based XGBoost with quantile support
│   │   ├── lightgbm_model.py      # Leaf-wise LightGBM with GOSS
│   │   ├── catboost_model.py      # CatBoost with oblivious decision trees
│   │   └── trainer.py             # Benchmarking harness & leaderboard generator
│   └── inventory/
│       ├── __init__.py
│       ├── optimizer.py           # Dynamic Safety Stock, ROP, and EOQ calculator
│       └── simulation.py          # Day-by-day inventory replenishment simulator
└── outputs/                       # Exported benchmark CSVs and policy reports
```

---

## 3. Mathematical Foundations

### 3.1 Cyclical Harmonic Transformations
To avoid discontinuities at period boundaries, calendar features are mapped to polar coordinates:
$$\text{sin\_dow} = \sin\left(\frac{2\pi \cdot t_{\text{dow}}}{7}\right), \quad \text{cos\_dow} = \cos\left(\frac{2\pi \cdot t_{\text{dow}}}{7}\right)$$
$$\text{sin\_doy} = \sin\left(\frac{2\pi \cdot t_{\text{day\_of\_year}}}{365.25}\right), \quad \text{cos\_doy} = \cos\left(\frac{2\pi \cdot t_{\text{day\_of\_year}}}{365.25}\right)$$

### 3.2 Supply Chain Error Metric: WAPE
Weighted Absolute Percentage Error avoids zero-demand division instability:
$$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i} \times 100\%$$

### 3.3 Dynamic Safety Stock ($SS$) & Reorder Point ($ROP$)
Accounting for both demand variance ($\sigma_D$) and supplier lead time variance ($\sigma_L$):
$$SS = Z_{\alpha} \cdot \sqrt{\bar{L} \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2}$$
$$ROP = (\bar{D} \cdot \bar{L}) + SS$$
*(Where $Z_{0.95} = 1.645$ for 95% cycle service level)*

### 3.4 Economic Order Quantity ($EOQ$)
Balancing ordering cost ($S$) against unit inventory carrying cost ($H = h \cdot \text{COGS}$):
$$EOQ = \sqrt{\frac{2 \cdot \mathcal{D}_{\text{annual}} \cdot S}{H}}$$

---

## 4. How to Install and Run

### 4.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.2 Run End-to-End Pipeline
```bash
python main.py
```
This single command executes:
1. Data generation (3 stores, 5 product categories, 3 full years of daily sales).
2. Leakage-free feature engineering (lags, rolling averages, cyclical transforms).
3. Chronological train/validation/test split.
4. Tri-model benchmark (XGBoost, LightGBM, CatBoost) reporting WAPE, RMSE, MAE, R², and training latency.
5. Dynamic Inventory Optimization ($SS$, $ROP$, $EOQ$).
6. Discrete-event policy simulation comparing GBDT vs. Static Baseline.

### 4.3 Launch Interactive Streamlit Dashboard
```bash
streamlit run app_streamlit.py
```
Explore live scenario simulations, adjust service levels ($\alpha$), lead times, and visualize stock trajectories in your browser.
