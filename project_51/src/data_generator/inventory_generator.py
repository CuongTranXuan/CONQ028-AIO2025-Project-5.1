import numpy as np
import pandas as pd
from datetime import datetime


class InventoryGenerator:
    """
    Generate inventory_data.csv by synthesizing current_stock per (store_id, product_id)
    using average daily sales and a random days-of-stock window.
    - Handles missing values by imputing with per-product and global averages.
    """

    def __init__(self, sales_df: pd.DataFrame, random_seed: int = 42):
        self.sales_df = sales_df.copy()
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

        # Normalize types
        if "date" in self.sales_df.columns:
            self.sales_df["date"] = pd.to_datetime(
                self.sales_df["date"], errors="coerce")
        # Ensure expected columns exist
        expected = {"date", "store_id", "product_id",
                    "quantity", "revenue", "cost"}
        missing = expected - set(self.sales_df.columns)
        if missing:
            raise ValueError(f"sales_df missing columns: {missing}")

        # Impute missing quantities with product/store means then global mean
        self.sales_df["quantity"] = pd.to_numeric(
            self.sales_df["quantity"], errors="coerce")
        # product-store mean daily sales will be computed later; here handle raw NaNs
        if self.sales_df["quantity"].isna().any():
            # fill with product mean
            self.sales_df["quantity"] = self.sales_df.groupby("product_id")["quantity"].transform(
                lambda s: s.fillna(s.mean())
            )
            # fill remaining with store mean
            self.sales_df["quantity"] = self.sales_df.groupby("store_id")["quantity"].transform(
                lambda s: s.fillna(s.mean())
            )
            # global fallback
            self.sales_df["quantity"] = self.sales_df["quantity"].fillna(
                self.sales_df["quantity"].mean())

        # Make sure non-negative
        self.sales_df["quantity"] = self.sales_df["quantity"].clip(lower=0)

    def _avg_daily_sales(self) -> pd.DataFrame:
        """Compute average daily sales per (store_id, product_id)."""
        # Ensure daily level by counting per date
        grp = (
            self.sales_df.groupby(["store_id", "product_id", "date"], as_index=False)[
                "quantity"]
            .sum()
        )
        # Average per day (over the dates we have)
        avg = (
            grp.groupby(["store_id", "product_id"], as_index=False)["quantity"]
            .mean()
            .rename(columns={"quantity": "avg_daily_sales"})
        )

        # Impute missing avg_daily_sales using product-wise mean then global
        avg["avg_daily_sales"] = pd.to_numeric(
            avg["avg_daily_sales"], errors="coerce")
        # product mean
        avg["avg_daily_sales"] = avg.groupby("product_id")["avg_daily_sales"].transform(
            lambda s: s.fillna(s.mean())
        )
        # global fallback
        avg["avg_daily_sales"] = avg["avg_daily_sales"].fillna(
            avg["avg_daily_sales"].mean())

        # Replace zeros with a tiny baseline to avoid zero-inventory explosion
        avg["avg_daily_sales"] = avg["avg_daily_sales"].replace(0, 0.1)
        return avg

    def generate_inventory_data(
        self,
        output_path: str = "/mnt/data/inventory_data.csv",
        min_days: int = 20,
        max_days: int = 100,
        baseline_zero_sales_max: int = 10,
    ) -> pd.DataFrame:
        """
        Create inventory data:
        - For (store, product) with sales: current_stock = avg_daily_sales * U(min_days, max_days)
        - For pairs with zero history: assign small random stock [0, baseline_zero_sales_max]
        Missing values are handled with per-product and global averages.
        """
        print("Generating inventory_data.csv ...")

        # Unique pairs observed anywhere in sales
        pairs = self.sales_df[["store_id", "product_id"]].drop_duplicates()

        # Last date as 'last_updated'
        end_date = self.sales_df["date"].max()
        if pd.isna(end_date):
            end_date = pd.Timestamp(datetime.utcnow().date())

        # Avg daily sales
        avg = self._avg_daily_sales()

        # Left join to ensure all pairs appear
        df = pairs.merge(avg, on=["store_id", "product_id"], how="left")

        # Impute any remaining NaNs with product/global means
        df["avg_daily_sales"] = df.groupby("product_id")["avg_daily_sales"].transform(
            lambda s: s.fillna(s.mean())
        )
        df["avg_daily_sales"] = df["avg_daily_sales"].fillna(
            df["avg_daily_sales"].mean())

        # Replace zeros (or still-NaNs) with tiny baseline to avoid zero stock
        df["avg_daily_sales"] = pd.to_numeric(
            df["avg_daily_sales"], errors="coerce").fillna(0.1)
        df.loc[df["avg_daily_sales"] <= 0, "avg_daily_sales"] = 0.1

        # Random days-of-stock per row
        days = np.random.uniform(min_days, max_days, size=len(df))

        # Compute inventory for pairs with sales signal
        current_stock = (df["avg_daily_sales"] * days).round().astype(int)

        # For entries that truly had no history in original data, allow small random stock
        # Identify original zero-history pairs by checking appearance in avg table
        has_history = df[["store_id", "product_id"]].merge(
            avg[["store_id", "product_id"]], how="left", indicator=True)["_merge"] == "both"
        no_hist_idx = ~has_history.values
        small_stock = np.random.randint(
            0, baseline_zero_sales_max + 1, size=no_hist_idx.sum())
        current_stock.loc[no_hist_idx] = small_stock

        # Ensure at least 1 unit (except allow 0 for no-history)
        current_stock = current_stock.clip(lower=0)
        current_stock.loc[~no_hist_idx] = current_stock.loc[~no_hist_idx].clip(
            lower=1)

        inventory_df = pd.DataFrame(
            {
                "store_id": df["store_id"].astype(str),
                "product_id": df["product_id"].astype(str),
                "current_stock": current_stock.astype(int),
                "last_updated": end_date.normalize(),
            }
        )

        # Save
        if output_path:
            inventory_df.to_csv(output_path, index=False)
            print(
                f"Saved inventory data to {output_path} (rows={len(inventory_df)})")

        return inventory_df


if __name__ == "__main__":
    # Load sales_data.csv synthesized earlier
    sales_path = "sales_data.csv"
    sales_df = pd.read_csv(sales_path, parse_dates=["date"])

    gen = InventoryGenerator(sales_df, random_seed=42)
    inv = gen.generate_inventory_data(
        output_path="inventory_data.csv")
    print(inv.head(10))
