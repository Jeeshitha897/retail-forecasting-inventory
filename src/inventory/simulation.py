"""Discrete-Event Retail Inventory Policy Simulator.
Simulates day-by-day inventory dynamics over the test horizon.
Compares Static Rule-of-Thumb Policy vs. Dynamic Gradient Boosting Policy.
Supervised by: Dr. P. Sudhakar
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import config

class InventorySimulator:
    def __init__(self, holding_cost_rate: float = config.ANNUAL_HOLDING_COST_RATE):
        self.holding_cost_rate = holding_cost_rate
        self.daily_holding_factor = holding_cost_rate / 365.0

    def simulate_single_sku(
        self,
        df_sku_test: pd.DataFrame,
        reorder_point: int,
        order_quantity: int,
        safety_stock: int,
        lead_time_days: int,
        unit_cogs: float,
        selling_price: float,
        order_cost: float,
        initial_stock: int = None
    ) -> Dict[str, Any]:
        """Simulates inventory trajectory over daily time horizon."""
        df_sku_test = df_sku_test.sort_values("date").reset_index(drop=True)
        n_days = len(df_sku_test)

        if initial_stock is None:
            initial_stock = safety_stock + order_quantity

        on_hand = initial_stock
        pipeline_orders = []  # list of tuples: (arrival_day_idx, quantity)

        total_demand = 0
        total_fulfilled = 0
        total_stockout_units = 0
        stockout_days = 0
        total_holding_cost = 0.0
        total_order_cost = 0.0
        orders_placed = 0

        inventory_levels = []

        for day_idx in range(n_days):
            # 1. Process arriving replenishment shipments
            arrived_qty = sum(qty for arr_day, qty in pipeline_orders if arr_day == day_idx)
            pipeline_orders = [(arr_day, qty) for arr_day, qty in pipeline_orders if arr_day > day_idx]
            on_hand += arrived_qty

            inventory_levels.append(on_hand)

            # 2. Demand realization
            demand = int(df_sku_test.loc[day_idx, "units_sold"])
            total_demand += demand

            if on_hand >= demand:
                fulfilled = demand
                stockout = 0
                on_hand -= demand
            else:
                fulfilled = on_hand
                stockout = demand - on_hand
                on_hand = 0
                stockout_days += 1

            total_fulfilled += fulfilled
            total_stockout_units += stockout

            # 3. Holding cost on end-of-day inventory
            total_holding_cost += on_hand * unit_cogs * self.daily_holding_factor

            # 4. Review inventory position and place replenishment order if needed
            # Inventory Position = On-Hand + On-Order
            on_order = sum(qty for _, qty in pipeline_orders)
            inventory_position = on_hand + on_order

            if inventory_position <= reorder_point:
                arrival_day = day_idx + lead_time_days
                pipeline_orders.append((arrival_day, order_quantity))
                orders_placed += 1
                total_order_cost += order_cost

        # Summary KPIs
        fill_rate = (total_fulfilled / total_demand * 100.0) if total_demand > 0 else 100.0
        margin_per_unit = max(0.5, selling_price - unit_cogs)
        stockout_penalty_cost = total_stockout_units * margin_per_unit * config.STOCKOUT_PENALTY_MULTIPLIER
        total_supply_chain_cost = total_holding_cost + total_order_cost + stockout_penalty_cost

        return {
            "total_demand": total_demand,
            "total_fulfilled": total_fulfilled,
            "total_stockout_units": total_stockout_units,
            "stockout_days": stockout_days,
            "fill_rate_pct": round(fill_rate, 2),
            "orders_placed": orders_placed,
            "holding_cost ($)": round(total_holding_cost, 2),
            "ordering_cost ($)": round(total_order_cost, 2),
            "stockout_penalty ($)": round(stockout_penalty_cost, 2),
            "total_supply_chain_cost ($)": round(total_supply_chain_cost, 2),
            "avg_inventory_units": round(float(np.mean(inventory_levels)), 1),
        }

    def run_comparative_simulation(
        self,
        test_df: pd.DataFrame,
        dynamic_policies_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Simulates all Store-SKUs comparing:
        1. Static Baseline Policy (Naive 7-day rule of thumb)
        2. Dynamic GBDT Optimization Policy
        """
        records = []
        static_total_cost = 0.0
        dynamic_total_cost = 0.0

        for _, pol in dynamic_policies_df.iterrows():
            store_id = pol["store_id"]
            sku_id = pol["sku_id"]
            sku_test = test_df[(test_df["store_id"] == store_id) & (test_df["sku_id"] == sku_id)].copy()

            if len(sku_test) == 0:
                continue

            unit_cogs = sku_test["cogs_per_unit"].iloc[0]
            selling_price = sku_test["selling_price"].iloc[0]
            lead_time = int(pol["lead_time_days"])
            order_cost = sku_test["order_cost"].iloc[0]

            # Dynamic GBDT Policy Parameters
            dyn_rop = int(pol["reorder_point_units"])
            dyn_eoq = int(pol["eoq_units"])
            dyn_ss = int(pol["safety_stock_units"])

            # Static Baseline Policy (Rule of thumb: Fixed 7 days safety stock, Fixed 14 days order qty)
            mean_demand = sku_test["units_sold"].mean()
            stat_ss = int(np.ceil(mean_demand * 5.0))
            stat_rop = int(np.ceil((mean_demand * lead_time) + stat_ss))
            stat_eoq = int(np.ceil(mean_demand * 12.0))

            dyn_res = self.simulate_single_sku(
                sku_test, dyn_rop, dyn_eoq, dyn_ss, lead_time, unit_cogs, selling_price, order_cost
            )
            stat_res = self.simulate_single_sku(
                sku_test, stat_rop, stat_eoq, stat_ss, lead_time, unit_cogs, selling_price, order_cost
            )

            cost_savings = stat_res["total_supply_chain_cost ($)"] - dyn_res["total_supply_chain_cost ($)"]
            savings_pct = (cost_savings / stat_res["total_supply_chain_cost ($)"] * 100.0) if stat_res["total_supply_chain_cost ($)"] > 0 else 0.0

            static_total_cost += stat_res["total_supply_chain_cost ($)"]
            dynamic_total_cost += dyn_res["total_supply_chain_cost ($)"]

            records.append({
                "Store": store_id,
                "SKU": sku_id,
                "Static Fill Rate (%)": stat_res["fill_rate_pct"],
                "GBDT Fill Rate (%)": dyn_res["fill_rate_pct"],
                "Static Stockouts (units)": stat_res["total_stockout_units"],
                "GBDT Stockouts (units)": dyn_res["total_stockout_units"],
                "Static Total Cost ($)": stat_res["total_supply_chain_cost ($)"],
                "GBDT Total Cost ($)": dyn_res["total_supply_chain_cost ($)"],
                "Cost Savings ($)": round(cost_savings, 2),
                "Savings (%)": round(savings_pct, 1),
            })

        summary_df = pd.DataFrame(records)
        net_savings = static_total_cost - dynamic_total_cost
        overall_savings_pct = (net_savings / static_total_cost * 100.0) if static_total_cost > 0 else 0.0

        overall_metrics = {
            "Static Fleet Total Cost ($)": round(static_total_cost, 2),
            "GBDT Fleet Total Cost ($)": round(dynamic_total_cost, 2),
            "Net Fleet Savings ($)": round(net_savings, 2),
            "Net Fleet Cost Reduction (%)": round(overall_savings_pct, 2),
        }

        return summary_df, overall_metrics
