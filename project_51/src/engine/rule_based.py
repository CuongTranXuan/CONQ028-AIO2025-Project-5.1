import sys
import time
from pathlib import Path

# Add project root to path for direct execution
if __name__ == "__main__" or "src.engine" not in sys.modules:
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    BASE_TRANSPORT_COST_PER_KM,
    DISTANCE_WEIGHT,
    EXCESS_WEIGHT,
    MAX_TRANSFER_DISTANCE_KM,
    NEEDED_WEIGHT,
)
from src.utils.logger import get_optimization_logger


class RuleBasedOptimizer:
    def __init__(self, distance_matrix=None, transport_cost_matrix=None):
        """
        Initialize the rule-based optimization engine.

        Args:
            distance_matrix: Matrix of distances between stores
            transport_cost_matrix: Matrix of transport costs between stores
        """
        self.distance_matrix = distance_matrix
        self.transport_cost_matrix = transport_cost_matrix
        self.transfer_plan = None
        self.logger_system = get_optimization_logger()

    def load_matrices(self, distance_path, cost_path):
        """
        Load distance and transport cost matrices from CSV files.

        Args:
            distance_path: Path to distance matrix CSV
            cost_path: Path to transport cost matrix CSV
        """
        print("Loading distance and transport cost matrices...")

        # Load matrices with store IDs as integers
        self.distance_matrix = pd.read_csv(distance_path, index_col=0)
        self.transport_cost_matrix = pd.read_csv(cost_path, index_col=0)

        # Ensure indices and columns are integers
        self.distance_matrix.index = self.distance_matrix.index.astype(int)
        self.distance_matrix.columns = self.distance_matrix.columns.astype(int)
        self.transport_cost_matrix.index = self.transport_cost_matrix.index.astype(int)
        self.transport_cost_matrix.columns = self.transport_cost_matrix.columns.astype(
            int
        )

    def optimize(self, excess_inventory, needed_inventory):
        """
        Generate a transfer plan using rule-based approach.

        Args:
            excess_inventory: DataFrame containing excess inventory
            needed_inventory: DataFrame containing needed inventory

        Returns:
            DataFrame containing transfer recommendations
        """
        # Start timing and logging
        start_time = time.time()

        parameters = {
            "excess_items": len(excess_inventory) if not excess_inventory.empty else 0,
            "needed_items": len(needed_inventory) if not needed_inventory.empty else 0,
            "algorithm": "Rule-Based Optimization",
        }

        self.logger_system.log_execution_start("rule_based_optimization", parameters)

        print("Generating rule-based transfer plan...")
        self.logger_system.log_progress(
            "rule_based_optimization", "Starting rule-based transfer plan generation..."
        )

        if excess_inventory.empty or needed_inventory.empty:
            message = "No excess or needed inventory found. No transfers needed."
            print(message)
            self.logger_system.log_progress("rule_based_optimization", message)
            self.transfer_plan = pd.DataFrame()

            # Log completion
            execution_time = time.time() - start_time
            results = {
                "transfers_generated": 0,
                "reason": "No excess or needed inventory",
            }
            self.logger_system.log_execution_end(
                "rule_based_optimization", execution_time, results
            )
            return self.transfer_plan

        # Create a list to store transfer recommendations
        transfers = []

        self.logger_system.log_progress(
            "rule_based_optimization", "Sorting excess and needed inventory..."
        )

        # Sort excess inventory by excess_units (descending)
        excess_sorted = excess_inventory.sort_values("excess_units", ascending=False)

        # Sort needed inventory by needed_units (descending)
        needed_sorted = needed_inventory.sort_values("needed_units", ascending=False)

        self.logger_system.log_progress(
            "rule_based_optimization",
            f"Processing {len(excess_sorted)} excess items and {len(needed_sorted)} needed items",
        )

        # Track how much has been transferred from each excess item
        transferred_from = {}
        for _, row in excess_sorted.iterrows():
            key = (row["store_id"], row["product_id"])
            transferred_from[key] = 0

        # Track how much has been transferred to each needed item
        transferred_to = {}
        for _, row in needed_sorted.iterrows():
            key = (row["store_id"], row["product_id"])
            transferred_to[key] = 0

        # For each product in need, find the closest store with excess
        with tqdm(
            needed_sorted.iterrows(),
            total=len(needed_sorted),
            desc="Processing needed inventory",
            unit="item",
        ) as pbar:
            for _, need_row in pbar:
                need_store_id = need_row["store_id"]
                need_product_id = need_row["product_id"]
                needed_units = need_row["needed_units"]

                # Adjust for already received units
                need_key = (need_store_id, need_product_id)
                if need_key in transferred_to:
                    needed_units -= transferred_to[need_key]

                if needed_units <= 0:
                    continue

                # Find excess inventory for this product
                excess_for_product = excess_sorted[
                    excess_sorted["product_id"] == need_product_id
                ]

                if excess_for_product.empty:
                    continue

                # Sort excess stores by distance to the need store
                excess_for_product = excess_for_product.copy()

                # Add distance to the excess inventory
                excess_for_product["distance"] = excess_for_product["store_id"].apply(
                    lambda x: (
                        float(self.distance_matrix.loc[x, need_store_id])
                        if x != need_store_id
                        and x in self.distance_matrix.index
                        and need_store_id in self.distance_matrix.columns
                        else float("inf")
                    )
                )

                # Sort by distance (closest first)
                excess_for_product = excess_for_product.sort_values("distance")

                # Transfer from closest stores with excess first
                for _, excess_row in excess_for_product.iterrows():
                    excess_store_id = excess_row["store_id"]
                    excess_product_id = excess_row["product_id"]
                    excess_units = excess_row["excess_units"]

                    # Skip self-transfers
                    if excess_store_id == need_store_id:
                        continue

                    # Adjust for already transferred units
                    excess_key = (excess_store_id, excess_product_id)
                    if excess_key in transferred_from:
                        excess_units -= transferred_from[excess_key]

                    if excess_units <= 0:
                        continue

                    # Calculate units to transfer
                    transfer_units = min(needed_units, excess_units)

                    if transfer_units > 0:
                        # Calculate distance from matrix
                        if (
                            excess_store_id in self.distance_matrix.index
                            and need_store_id in self.distance_matrix.columns
                        ):
                            distance = float(
                                self.distance_matrix.loc[excess_store_id, need_store_id]
                            )
                        else:
                            distance = 0  # Default if distance not available

                        # Calculate transport cost from matrix
                        if (
                            self.transport_cost_matrix is not None
                            and excess_store_id in self.transport_cost_matrix.index
                            and need_store_id in self.transport_cost_matrix.columns
                        ):
                            cost_value = self.transport_cost_matrix.loc[
                                excess_store_id, need_store_id
                            ]
                            base_cost = float(cost_value)
                            if np.isnan(base_cost) or base_cost <= 0:
                                # Skip this transfer if cost is invalid
                                self.logger_system.log_progress(
                                    "rule_based_optimization",
                                    f"Skipping transfer {excess_store_id} → {need_store_id}: invalid cost ({base_cost})",
                                )
                                continue
                            transport_cost = base_cost * transfer_units
                        else:
                            # Skip this transfer if stores not found in matrix or matrix is None
                            self.logger_system.log_progress(
                                "rule_based_optimization",
                                f"Skipping transfer {excess_store_id} → {need_store_id}: stores not in cost matrix or matrix unavailable",
                            )
                            continue

                        # Add transfer to recommendations
                        transfers.append(
                            {
                                "from_store_id": excess_store_id,
                                "to_store_id": need_store_id,
                                "product_id": need_product_id,
                                "units": int(transfer_units),
                                "distance_km": distance,
                                "transport_cost": transport_cost,
                            }
                        )

                        # Update tracking dictionaries
                        if excess_key in transferred_from:
                            transferred_from[excess_key] += transfer_units
                        else:
                            transferred_from[excess_key] = transfer_units

                        if need_key in transferred_to:
                            transferred_to[need_key] += transfer_units
                        else:
                            transferred_to[need_key] = transfer_units

                        # Update needed units
                        needed_units -= transfer_units

                        if needed_units <= 0:
                            break

                # Update progress bar with current item info
                pbar.set_postfix({"product": need_product_id, "store": need_store_id})

        # Create DataFrame from transfers
        self.transfer_plan = pd.DataFrame(transfers)

        if not self.transfer_plan.empty:
            # Calculate total cost and summary metrics
            total_units = self.transfer_plan["units"].sum()
            total_cost = self.transfer_plan["transport_cost"].sum()
            avg_cost_per_unit = total_cost / total_units if total_units > 0 else 0

            summary_msg = f"Rule-Based Transfer Plan Summary:"
            print(summary_msg)
            print(f"- Total transfers: {len(self.transfer_plan)}")
            print(f"- Total units to transfer: {total_units}")
            print(f"- Total transport cost: {total_cost:,.0f} VND")
            print(f"- Average cost per unit: {avg_cost_per_unit:,.0f} VND")

            # Log results
            self.logger_system.log_progress("rule_based_optimization", summary_msg)
            self.logger_system.log_progress(
                "rule_based_optimization", f"Total transfers: {len(self.transfer_plan)}"
            )
            self.logger_system.log_progress(
                "rule_based_optimization", f"Total units to transfer: {total_units}"
            )
            self.logger_system.log_progress(
                "rule_based_optimization",
                f"Total transport cost: {total_cost:,.0f} VND",
            )
            self.logger_system.log_progress(
                "rule_based_optimization",
                f"Average cost per unit: {avg_cost_per_unit:,.0f} VND",
            )

        else:
            no_transfers_msg = "No transfers recommended."
            print(no_transfers_msg)
            self.logger_system.log_progress("rule_based_optimization", no_transfers_msg)

        # Log execution completion
        execution_time = time.time() - start_time
        results = {
            "transfers_generated": len(self.transfer_plan),
            "total_units": (
                self.transfer_plan["units"].sum() if not self.transfer_plan.empty else 0
            ),
            "total_cost": (
                self.transfer_plan["transport_cost"].sum()
                if not self.transfer_plan.empty
                else 0
            ),
            "avg_cost_per_unit": (
                (
                    self.transfer_plan["transport_cost"].sum()
                    / self.transfer_plan["units"].sum()
                )
                if not self.transfer_plan.empty
                and self.transfer_plan["units"].sum() > 0
                else 0
            ),
        }

        self.logger_system.log_execution_end(
            "rule_based_optimization", execution_time, results
        )

        return self.transfer_plan

    def add_store_product_names(self, stores_df=None, products_df=None):
        """
        Add store and product names to the transfer plan for better readability.

        Args:
            stores_df: DataFrame containing store information
            products_df: DataFrame containing product information
        """
        if self.transfer_plan is None or self.transfer_plan.empty:
            return

        if stores_df is not None:
            # Add store names
            store_name_map = stores_df.set_index("store_id")["store_name"].to_dict()
            self.transfer_plan["from_store"] = self.transfer_plan["from_store_id"].map(
                store_name_map
            )
            self.transfer_plan["to_store"] = self.transfer_plan["to_store_id"].map(
                store_name_map
            )

        if products_df is not None:
            # Add product names
            product_name_map = products_df.set_index("product_id")[
                "product_name"
            ].to_dict()
            self.transfer_plan["product"] = self.transfer_plan["product_id"].map(
                product_name_map
            )


if __name__ == "__main__":
    import os
    from pathlib import Path

    from src.engine.analyzer import InventoryAnalyzer

    # Check if data files exist (relative to project root)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"

    # Create results directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)

    required_files = [
        "sales_data.csv",
        "inventory_data.csv",
        "distance_matrix.csv",
        "transport_cost_matrix.csv",
    ]

    print("Checking required data files...")
    for file in required_files:
        if not (data_dir / file).exists():
            print(f"Required file {file} not found. Please run data generator first.")
            exit(1)
        else:
            print(f"[OK] {file}")

    # Create analyzer
    analyzer = InventoryAnalyzer()

    # Load data
    analyzer.load_data(
        sales_path=str(data_dir / "sales_data.csv"),
        inventory_path=str(data_dir / "inventory_data.csv"),
        stores_path=(
            str(data_dir / "stores.csv") if (data_dir / "stores.csv").exists() else None
        ),
        products_path=(
            str(data_dir / "products.csv")
            if (data_dir / "products.csv").exists()
            else None
        ),
    )

    # Analyze data
    analyzer.analyze_sales_data()

    # Identify imbalances
    excess_df, needed_df = analyzer.identify_inventory_imbalances()

    # Create optimizer
    optimizer = RuleBasedOptimizer()

    # Load matrices
    optimizer.load_matrices(
        distance_path=str(data_dir / "distance_matrix.csv"),
        cost_path=str(data_dir / "transport_cost_matrix.csv"),
    )

    # Generate transfer plan
    transfer_plan = optimizer.optimize(excess_df, needed_df)

    # Add store and product names if data available
    if (data_dir / "stores.csv").exists() and (data_dir / "products.csv").exists():
        stores_df = pd.read_csv(str(data_dir / "stores.csv"))
        products_df = pd.read_csv(str(data_dir / "products.csv"))
        optimizer.add_store_product_names(stores_df, products_df)

    # Evaluate impact
    if not transfer_plan.empty:
        impact_df, _ = analyzer.evaluate_plan_impact(transfer_plan)

        # Save transfer plan
        output_path = str(results_dir / "rule_based_transfers.csv")
        transfer_plan.to_csv(output_path, index=False)
        print(f"Transfer plan saved to {output_path}")
