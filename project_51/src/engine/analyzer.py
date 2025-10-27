import numpy as np
import pandas as pd

from src.config import MAX_INVENTORY_DAYS, MIN_INVENTORY_DAYS


class InventoryAnalyzer:
    def __init__(self, sales_df=None, inventory_df=None, stores=None, products=None):
        """
        Initialize with sales and inventory data.

        Args:
            sales_df: DataFrame containing sales records
            inventory_df: DataFrame containing current inventory levels
            stores: List of Store objects or DataFrame with store data
            products: List of Product objects or DataFrame with product data
        """
        self.sales_df = sales_df
        self.inventory_df = inventory_df
        self.stores = stores
        self.products = products
        self.analysis_df = None
        self.excess_inventory = None
        self.needed_inventory = None

        # Ensure date column is datetime if sales_df is provided
        if self.sales_df is not None and "date" in self.sales_df.columns:
            if self.sales_df["date"].dtype == "object":  # String type
                print("Converting date column to datetime format...")
                self.sales_df["date"] = pd.to_datetime(self.sales_df["date"])

    def load_data(
        self, sales_path, inventory_path, stores_path=None, products_path=None
    ):
        """
        Load data from CSV files.

        Args:
            sales_path: Path to sales data CSV
            inventory_path: Path to inventory data CSV
            stores_path: Optional path to stores data CSV
            products_path: Optional path to products data CSV
        """
        print("Loading data from CSV files...")

        # Load sales data
        self.sales_df = pd.read_csv(sales_path, parse_dates=["date"])

        # Load inventory data
        self.inventory_df = pd.read_csv(inventory_path)

        # Optionally load store and product data
        if stores_path:
            self.stores = pd.read_csv(stores_path)

        if products_path:
            self.products = pd.read_csv(products_path)

        print(
            f"Loaded {len(self.sales_df)} sales records and {len(self.inventory_df)} inventory records"
        )

    def analyze_sales_data(self):
        """
        Analyze sales data to identify patterns and calculate key metrics.

        Returns:
            DataFrame with sales analysis metrics
        """
        print("Analyzing sales data...")

        # Check if sales_df is loaded
        if self.sales_df is None:
            raise ValueError(
                "Sales data not loaded. Please provide sales_df in constructor or call load_data()"
            )

        # Add store city to sales data if stores data is available
        if isinstance(self.stores, pd.DataFrame):
            store_city_map = self.stores.set_index("store_id")["city"].to_dict()
            self.sales_df["city"] = self.sales_df["store_id"].map(store_city_map)

        # Add product category to sales data if product data is available
        if isinstance(self.products, pd.DataFrame):
            product_category_map = self.products.set_index("product_id")[
                "category"
            ].to_dict()
            self.sales_df["category"] = self.sales_df["product_id"].map(
                product_category_map
            )

        # Calculate sales metrics
        sales_metrics = (
            self.sales_df.groupby(["store_id", "product_id"])
            .agg(
                {
                    "quantity": ["sum", "mean", "std", "count"],
                    "revenue": ["sum", "mean"],
                }
            )
            .reset_index()
        )

        # Flatten the multi-level columns
        sales_metrics.columns = [
            "_".join(col).strip("_") for col in sales_metrics.columns.values
        ]

        # Calculate coefficient of variation (CV) to measure sales consistency
        sales_metrics["quantity_cv"] = (
            sales_metrics["quantity_std"] / sales_metrics["quantity_mean"]
        )
        sales_metrics["quantity_cv"].fillna(0, inplace=True)

        # Calculate days since last sale
        last_sale_date = (
            self.sales_df.groupby(["store_id", "product_id"])["date"]
            .max()
            .reset_index()
        )
        last_sale_date.columns = ["store_id", "product_id", "last_sale_date"]

        end_date = self.sales_df["date"].max()
        last_sale_date["days_since_last_sale"] = (
            end_date - last_sale_date["last_sale_date"]
        ).dt.days

        # Merge metrics with inventory data
        analysis_df = pd.merge(
            self.inventory_df, sales_metrics, on=["store_id", "product_id"], how="left"
        )
        analysis_df = pd.merge(
            analysis_df, last_sale_date, on=["store_id", "product_id"], how="left"
        )

        # Fill NaN values for products with no sales
        for col in [
            "quantity_sum",
            "quantity_mean",
            "quantity_std",
            "quantity_count",
            "revenue_sum",
            "revenue_mean",
            "quantity_cv",
        ]:
            analysis_df[col].fillna(0, inplace=True)

        # Calculate days of inventory based on average daily sales
        # Use the mean quantity per sale (matches data generator logic)
        analysis_df["avg_daily_sales"] = analysis_df["quantity_mean"]
        analysis_df["avg_daily_sales"].replace(
            0, 0.01, inplace=True
        )  # Avoid division by zero

        analysis_df["days_of_inventory"] = (
            analysis_df["current_stock"] / analysis_df["avg_daily_sales"]
        )

        # Add city information if available
        if isinstance(self.stores, pd.DataFrame):
            store_city_map = self.stores.set_index("store_id")["city"].to_dict()
            analysis_df["city"] = analysis_df["store_id"].map(store_city_map)

        # Add product category if available
        if isinstance(self.products, pd.DataFrame):
            product_category_map = self.products.set_index("product_id")[
                "category"
            ].to_dict()
            analysis_df["category"] = analysis_df["product_id"].map(
                product_category_map
            )

        # Classification into fast/slow moving based on sales velocity
        sales_velocity_threshold = analysis_df["avg_daily_sales"].quantile(0.5)
        analysis_df["sales_velocity"] = np.where(
            analysis_df["avg_daily_sales"] >= sales_velocity_threshold, "Fast", "Slow"
        )

        self.analysis_df = analysis_df
        print("Sales data analysis complete.")
        return analysis_df

    def identify_inventory_imbalances(self, min_days=None, max_days=None):
        """Use config defaults if not provided."""
        min_days = min_days or MIN_INVENTORY_DAYS
        max_days = max_days or MAX_INVENTORY_DAYS
        """
        Identify inventory imbalances across stores.
        - Excess: more than max_days of inventory
        - Needed: less than min_days of inventory

        Args:
            min_days: Threshold for needed inventory (in days)
            max_days: Threshold for excess inventory (in days)

        Returns:
            Tuple of (excess_inventory_df, needed_inventory_df)
        """
        print("Identifying inventory imbalances...")

        if self.analysis_df is None:
            self.analyze_sales_data()

        # Create copies to avoid modifying the original
        self.analysis_df["inventory_status"] = "Balanced"

        # Identify excess inventory (more than max_days of stock)
        excess_mask = (self.analysis_df["days_of_inventory"] > max_days) & (
            self.analysis_df["current_stock"] > 0
        )
        self.analysis_df.loc[excess_mask, "inventory_status"] = "Excess"

        # Identify needed inventory (less than min_days of stock)
        need_mask = self.analysis_df["days_of_inventory"] < min_days
        self.analysis_df.loc[need_mask, "inventory_status"] = "Needed"

        # Calculate excess units (units above max_days of inventory)
        self.analysis_df["excess_units"] = np.where(
            excess_mask,
            self.analysis_df["current_stock"]
            - (max_days * self.analysis_df["avg_daily_sales"]).astype(int),
            0,
        )

        # Calculate needed units (units needed to reach min_days of inventory)
        self.analysis_df["needed_units"] = np.where(
            need_mask,
            (
                (min_days * self.analysis_df["avg_daily_sales"])
                - self.analysis_df["current_stock"]
            ).astype(int),
            0,
        )

        # Create separate DataFrames for excess and needed inventory
        self.excess_inventory = self.analysis_df[excess_mask].copy()
        self.needed_inventory = self.analysis_df[need_mask].copy()

        # Summary statistics
        excess_count = self.excess_inventory.shape[0]
        needed_count = self.needed_inventory.shape[0]
        balanced_count = self.analysis_df.shape[0] - excess_count - needed_count

        total_excess_units = self.excess_inventory["excess_units"].sum()
        total_needed_units = self.needed_inventory["needed_units"].sum()

        print(f"Inventory Status Summary:")
        print(
            f"- Excess inventory items: {excess_count} (total units: {total_excess_units})"
        )
        print(
            f"- Needed inventory items: {needed_count} (total units: {total_needed_units})"
        )
        print(f"- Balanced inventory items: {balanced_count}")

        return self.excess_inventory, self.needed_inventory

    def evaluate_plan_impact(self, transfer_plan):
        """
        Evaluate the impact of a transfer plan on inventory levels.

        Args:
            transfer_plan: DataFrame containing transfer recommendations

        Returns:
            Tuple of (impact_summary_df, post_transfer_inventory_df)
        """
        print("Evaluating transfer plan impact...")

        if self.analysis_df is None:
            self.analyze_sales_data()

        if transfer_plan is None or transfer_plan.empty:
            print("No transfer plan to evaluate.")
            return None, self.analysis_df

        # Create a copy of the current inventory to simulate post-transfer inventory
        post_inventory = self.inventory_df.copy()

        # Apply transfers to inventory
        for _, transfer in transfer_plan.iterrows():
            from_store_id = transfer["from_store_id"]
            to_store_id = transfer["to_store_id"]
            product_id = transfer["product_id"]
            units = transfer["units"]

            # Update inventory at source store
            from_mask = (post_inventory["store_id"] == from_store_id) & (
                post_inventory["product_id"] == product_id
            )
            post_inventory.loc[from_mask, "current_stock"] -= units

            # Update inventory at destination store
            to_mask = (post_inventory["store_id"] == to_store_id) & (
                post_inventory["product_id"] == product_id
            )
            post_inventory.loc[to_mask, "current_stock"] += units

        # Re-analyze inventory status after transfers
        # Create a copy of the analysis with post-transfer inventory
        post_analysis = self.analysis_df.copy()
        post_analysis["current_stock"] = post_inventory["current_stock"]

        # Calculate days of inventory based on average daily sales
        post_analysis["days_of_inventory"] = (
            post_analysis["current_stock"] / post_analysis["avg_daily_sales"]
        )
        post_analysis["days_of_inventory"].replace(
            np.inf, 365, inplace=True
        )  # Cap at 1 year for zero sales

        # Identify inventory status after transfers
        min_days = MIN_INVENTORY_DAYS
        max_days = MAX_INVENTORY_DAYS

        post_analysis["post_inventory_status"] = "Balanced"

        # Identify excess inventory (more than max_days of stock)
        excess_mask = (post_analysis["days_of_inventory"] > max_days) & (
            post_analysis["current_stock"] > 0
        )
        post_analysis.loc[excess_mask, "post_inventory_status"] = "Excess"

        # Identify needed inventory (less than min_days of stock)
        need_mask = post_analysis["days_of_inventory"] < min_days
        post_analysis.loc[need_mask, "post_inventory_status"] = "Needed"

        # Compare before and after
        # Count items in each status before
        before_counts = self.analysis_df["inventory_status"].value_counts()

        # Count items in each status after
        after_counts = post_analysis["post_inventory_status"].value_counts()

        # Calculate average days of inventory before and after
        avg_days_before = self.analysis_df["days_of_inventory"].mean()
        avg_days_after = post_analysis["days_of_inventory"].mean()

        # Calculate standard deviation of days of inventory before and after
        std_days_before = self.analysis_df["days_of_inventory"].std()
        std_days_after = post_analysis["days_of_inventory"].std()

        # Calculate total transfers and costs
        total_transfers = len(transfer_plan)
        total_units = transfer_plan["units"].sum()
        total_cost = (
            transfer_plan["transport_cost"].sum()
            if "transport_cost" in transfer_plan.columns
            else 0
        )

        # Calculate product turnover improvement
        turnover_before = 365 / avg_days_before if avg_days_before > 0 else 0
        turnover_after = 365 / avg_days_after if avg_days_after > 0 else 0
        turnover_improvement = (
            (turnover_after - turnover_before) / turnover_before * 100
            if turnover_before > 0
            else 0
        )

        # Calculate inventory balance improvement
        imbalance_before = std_days_before
        imbalance_after = std_days_after
        balance_improvement = (
            (imbalance_before - imbalance_after) / imbalance_before * 100
            if imbalance_before > 0
            else 0
        )

        # Calculate product values to estimate financial impact
        if isinstance(self.products, pd.DataFrame) and "cost" in self.products.columns:
            product_value_map = self.products.set_index("product_id")["cost"].to_dict()
            post_analysis["product_value"] = post_analysis["product_id"].map(
                product_value_map
            )

            # Calculate inventory value before and after
            inventory_value_before = (
                self.analysis_df["current_stock"]
                * self.analysis_df["product_id"].map(product_value_map)
            ).sum()
            inventory_value_after = (
                post_analysis["current_stock"] * post_analysis["product_value"]
            ).sum()

            # Calculate excess inventory value before and after
            excess_value_before = (
                self.analysis_df.loc[
                    self.analysis_df["inventory_status"] == "Excess", "current_stock"
                ]
                * self.analysis_df.loc[
                    self.analysis_df["inventory_status"] == "Excess", "product_id"
                ].map(product_value_map)
            ).sum()

            excess_value_after = (
                post_analysis.loc[
                    post_analysis["post_inventory_status"] == "Excess", "current_stock"
                ]
                * post_analysis.loc[
                    post_analysis["post_inventory_status"] == "Excess", "product_value"
                ]
            ).sum()
        else:
            # If product cost data is not available, use placeholder values
            inventory_value_before = self.analysis_df["current_stock"].sum()
            inventory_value_after = post_analysis["current_stock"].sum()
            excess_value_before = self.analysis_df.loc[
                self.analysis_df["inventory_status"] == "Excess", "current_stock"
            ].sum()
            excess_value_after = post_analysis.loc[
                post_analysis["post_inventory_status"] == "Excess", "current_stock"
            ].sum()

        # Create impact summary
        impact_summary = {
            "Before Transfer": {
                "Excess Items": before_counts.get("Excess", 0),
                "Needed Items": before_counts.get("Needed", 0),
                "Balanced Items": before_counts.get("Balanced", 0),
                "Avg Days of Inventory": avg_days_before,
                "Inventory Imbalance (StdDev)": std_days_before,
                "Product Turnover": turnover_before,
                "Total Inventory Value": inventory_value_before,
                "Excess Inventory Value": excess_value_before,
            },
            "After Transfer": {
                "Excess Items": after_counts.get("Excess", 0),
                "Needed Items": after_counts.get("Needed", 0),
                "Balanced Items": after_counts.get("Balanced", 0),
                "Avg Days of Inventory": avg_days_after,
                "Inventory Imbalance (StdDev)": std_days_after,
                "Product Turnover": turnover_after,
                "Total Inventory Value": inventory_value_after,
                "Excess Inventory Value": excess_value_after,
            },
            "Improvement": {
                "Reduction in Excess Items": before_counts.get("Excess", 0)
                - after_counts.get("Excess", 0),
                "Reduction in Needed Items": before_counts.get("Needed", 0)
                - after_counts.get("Needed", 0),
                "Increase in Balanced Items": after_counts.get("Balanced", 0)
                - before_counts.get("Balanced", 0),
                "Product Turnover Improvement": f"{turnover_improvement:.2f}%",
                "Inventory Balance Improvement": f"{balance_improvement:.2f}%",
                "Reduction in Excess Value": excess_value_before - excess_value_after,
            },
            "Transfer Plan": {
                "Total Transfers": total_transfers,
                "Total Units Transferred": total_units,
                "Total Transport Cost": total_cost,
                "Avg Cost Per Unit": total_cost / total_units if total_units > 0 else 0,
            },
        }

        # Convert to DataFrame for easier viewing
        impact_df = pd.DataFrame(impact_summary)

        # Print summary
        print("\nTransfer Plan Impact Summary:")
        print(impact_df)

        return impact_df, post_analysis


if __name__ == "__main__":
    import os

    # Check if data files exist
    data_dir = "data"
    required_files = ["sales_data.csv", "inventory_data.csv"]

    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"Required file {file} not found. Please run data generator first.")
            exit(1)

    # Create analyzer
    analyzer = InventoryAnalyzer()

    # Load data
    analyzer.load_data(
        sales_path=os.path.join(data_dir, "sales_data.csv"),
        inventory_path=os.path.join(data_dir, "inventory_data.csv"),
        stores_path=(
            os.path.join(data_dir, "stores.csv")
            if os.path.exists(os.path.join(data_dir, "stores.csv"))
            else None
        ),
        products_path=(
            os.path.join(data_dir, "products.csv")
            if os.path.exists(os.path.join(data_dir, "products.csv"))
            else None
        ),
    )

    # Analyze data
    analysis_df = analyzer.analyze_sales_data()

    # Identify imbalances
    excess_df, needed_df = analyzer.identify_inventory_imbalances()

    # Print additional insights
    if "category" in analysis_df.columns:
        print("\nInventory Status by Category:")
        category_status = (
            analysis_df.groupby(["category", "inventory_status"])
            .size()
            .unstack()
            .fillna(0)
        )
        print(category_status)

    if "city" in analysis_df.columns:
        print("\nInventory Status by City:")
        city_status = (
            analysis_df.groupby(["city", "inventory_status"]).size().unstack().fillna(0)
        )
        print(city_status)
