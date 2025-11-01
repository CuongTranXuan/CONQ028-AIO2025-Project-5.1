import os
import pandas as pd


class SalesGenerator:
    """
    Builds sales_data.csv from the Olist order items dataset.
    Only this class should be modified to change generation behavior.
    """

    def __init__(
        self,
        src_csv: str = "./data/raw/olist_order_items_dataset.csv",
        out_csv: str = "./data/processed/sales_data.csv",
    ) -> None:
        self.src_csv = src_csv
        self.out_csv = out_csv

    def generate(self) -> pd.DataFrame:
        """
        Generate the required CSV with columns:
        date, store_id, product_id, quantity, revenue, cost

        Mapping:
            - date: date part of shipping_limit_date
            - store_id: seller_id
            - product_id: as-is
            - quantity: count of items (each row in order_items is one item)
            - revenue: sum(price)
            - cost: sum(freight_value)  # proxy for cost
        """
        df = pd.read_csv(self.src_csv)

        # Validate expected columns
        required = {"product_id", "seller_id",
                    "shipping_limit_date", "price", "freight_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing expected columns in {self.src_csv}: {missing}")

        # Parse to date-only
        df["date"] = pd.to_datetime(
            df["shipping_limit_date"], errors="coerce").dt.date
        df = df.dropna(subset=["date"])

        # Aggregate to target schema
        sales = (
            df.groupby(["date", "seller_id", "product_id"], as_index=False)
            .agg(
                quantity=("product_id", "size"),
                revenue=("price", "sum"),
                cost=("freight_value", "sum"),
            )
            .rename(columns={"seller_id": "store_id"})
            .sort_values(["date", "store_id", "product_id"])
        )

        # Save sales_data.csv
        os.makedirs(os.path.dirname(self.out_csv) or ".", exist_ok=True)
        sales.to_csv(self.out_csv, index=False)
        return sales


if __name__ == "__main__":
    generator = SalesGenerator(
        src_csv="./data/raw/olist_order_items_dataset.csv",
        out_csv="./data/processed/sales_data.csv"
    )
    out_df = generator.generate()
    print(f"Wrote {len(out_df)} rows to {generator.out_csv}")
    print("Columns:", ", ".join(out_df.columns))
