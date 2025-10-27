import datetime as dt
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import RANDOM_SEED


class SalesGenerator:
    def __init__(self, stores, products, random_seed=None):
        """
        Initialize the sales generator with store and product data.

        Args:
            stores: List of Store objects
            products: List of Product objects
            random_seed: Optional random seed for reproducibility (uses config default if None)
        """
        self.stores = stores
        self.products = products
        self.random_seed = random_seed or RANDOM_SEED
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def generate_sales_data(self, days=365, output_path=None):
        """
        Generate sales data for the specified number of days.

        Args:
            days: Number of days to generate sales for
            output_path: Optional path to save sales data to CSV

        Returns:
            DataFrame containing sales records
        """
        print(f"Generating sales data for {days} days...")

        # Create date range
        end_date = dt.datetime.now().date()
        start_date = end_date - dt.timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Generate sales records
        sales_records = []

        for date in tqdm(date_range, desc="Generating sales data"):
            for store in self.stores:
                # Create different sales patterns based on city
                city_factor = 1.0
                if store.city == "Hanoi":
                    # Higher electronics and clothing sales
                    category_factors = {
                        "Electronics": 1.5,
                        "Clothing": 1.3,
                        "Home Goods": 0.9,
                        "Food": 1.0,
                        "Beauty": 0.8,
                    }
                elif store.city == "Da Nang":
                    # Higher home goods and beauty sales
                    category_factors = {
                        "Electronics": 0.9,
                        "Clothing": 1.0,
                        "Home Goods": 1.4,
                        "Food": 1.1,
                        "Beauty": 1.3,
                    }
                else:  # Ho Chi Minh City
                    # Higher food and electronics sales
                    category_factors = {
                        "Electronics": 1.3,
                        "Clothing": 1.1,
                        "Home Goods": 1.0,
                        "Food": 1.5,
                        "Beauty": 1.2,
                    }

                # Seasonal factors (e.g., higher electronics sales near Tet holiday)
                month = date.month
                if month == 1 or month == 12:  # Tet and Christmas seasons
                    category_factors["Electronics"] *= 1.5
                    category_factors["Home Goods"] *= 1.3
                if month >= 6 and month <= 8:  # Summer season
                    category_factors["Clothing"] *= 1.2
                    category_factors["Beauty"] *= 1.3

                # Generate sales for a subset of products each day
                num_products_sold = np.random.randint(10, 30)
                products_sold = random.sample(self.products, num_products_sold)

                for product in products_sold:
                    # Base quantity with some randomness
                    base_quantity = np.random.randint(1, 10)

                    # Apply category and city factors
                    adjusted_quantity = int(
                        base_quantity * category_factors[product.category] * city_factor
                    )

                    # Ensure at least 1 unit is sold
                    quantity = max(1, adjusted_quantity)

                    sales_records.append(
                        {
                            "date": date,
                            "store_id": store.id,
                            "product_id": product.id,
                            "quantity": quantity,
                            "revenue": quantity * product.price,
                            "cost": quantity * product.cost,
                        }
                    )

        # Create sales dataframe
        sales_df = pd.DataFrame(sales_records)

        # If output path is provided, save to CSV
        if output_path:
            sales_df.to_csv(output_path, index=False)
            print(f"Saved {len(sales_df)} sales records to {output_path}")

        return sales_df


if __name__ == "__main__":
    import os

    from src.data_generator.product_generator import ProductGenerator
    from src.data_generator.store_generator import StoreGenerator

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Generate stores and products
    stores = StoreGenerator().generate_stores("data/stores.csv")
    products = ProductGenerator().generate_products(100, "data/products.csv")

    # Generate sales data
    generator = SalesGenerator(stores, products)
    sales_df = generator.generate_sales_data(
        30, "data/sales_data.csv"
    )  # 30 days of data

    # Print summary statistics
    print("\nSales Summary:")
    print(f"Total sales records: {len(sales_df)}")
    print(f"Date range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"Total revenue: {sales_df['revenue'].sum():,.0f} VND")
    print(f"Total units sold: {sales_df['quantity'].sum()}")
