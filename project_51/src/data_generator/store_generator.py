import random

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, STORE_CITIES
from src.engine.data_model import Store


class StoreGenerator:
    def __init__(self, random_seed=None):
        """Initialize with optional random seed for reproducibility."""
        self.random_seed = random_seed or RANDOM_SEED
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Use city configuration from config
        self.cities = STORE_CITIES

        # Define realistic store name patterns
        self.brand_names = [
            "Saigon Retail",
            "VinMart",
            "Hapro",
            "Mega Market",
            "Bach Hoa Xanh",
            "Lotus Market",
            "Co.op Mart",
            "FiviMart",
            "CircleK",
            "Aeon",
            "Lan Chi",
            "Satra",
            "Nam An Market",
            "Sunshine",
            "Golden Star",
        ]

        # Define location-specific store names
        self.location_names = {
            "Hanoi": [
                "Ba Đình",
                "Hoàn Kiếm",
                "Tây Hồ",
                "Long Biên",
                "Cầu Giấy",
                "Đống Đa",
                "Hai Bà Trưng",
                "Hoàng Mai",
                "Thanh Xuân",
                "Mỹ Đình",
            ],
            "Da Nang": [
                "Hải Châu",
                "Thanh Khê",
                "Sơn Trà",
                "Ngũ Hành Sơn",
                "Liên Chiểu",
                "Hòa Vang",
                "Cẩm Lệ",
                "An Hải",
                "Mỹ Khê",
                "Hòa Khánh",
            ],
            "Ho Chi Minh City": [
                "Quận 1",
                "Quận 3",
                "Quận 5",
                "Quận 7",
                "Thủ Đức",
                "Bình Thạnh",
                "Phú Nhuận",
                "Tân Bình",
                "Gò Vấp",
                "Bình Tân",
                "Tân Phú",
                "Nhà Bè",
            ],
        }

        # Define mall names for larger store locations
        self.mall_names = {
            "Hanoi": [
                "Vincom Center Bà Triệu",
                "Lotte Center",
                "Aeon Mall Long Biên",
                "Times City",
                "Royal City",
                "Tràng Tiền Plaza",
                "Indochina Plaza",
            ],
            "Da Nang": [
                "Vincom Plaza",
                "Lotte Mart",
                "Indochina Riverside Mall",
                "Sense Market",
                "Han Market Complex",
            ],
            "Ho Chi Minh City": [
                "Vincom Center",
                "Saigon Centre",
                "Vivo City",
                "Crescent Mall",
                "Estella Place",
                "SC Vivocity",
                "Landmark 81",
                "Diamond Plaza",
            ],
        }

    def generate_stores(self, output_path=None):
        """
        Generate store data based on predefined city parameters.

        Args:
            output_path: Optional path to save the stores to CSV

        Returns:
            List of Store objects
        """
        stores = []
        store_id = 1

        for city, info in self.cities.items():
            # Get location names for this city
            locations = self.location_names[city]
            malls = self.mall_names[city]

            for i in range(info["count"]):
                lat = np.random.uniform(info["lat_range"][0], info["lat_range"][1])
                lon = np.random.uniform(info["lon_range"][0], info["lon_range"][1])

                # Decide store name format (50% chance of brand+location, 30% mall location, 20% local market)
                name_type = random.choices(
                    ["brand", "mall", "market"], weights=[0.5, 0.3, 0.2], k=1
                )[0]

                if name_type == "brand":
                    # Brand store in a location
                    brand = random.choice(self.brand_names)
                    location = random.choice(locations)
                    store_name = f"{brand} {location}"
                elif name_type == "mall":
                    # Store in a shopping mall
                    brand = random.choice(self.brand_names)
                    mall = random.choice(malls)
                    store_name = f"{brand} - {mall}"
                else:
                    # Local market store
                    location = random.choice(locations)
                    store_name = f"{location} Market Center"

                stores.append(Store(store_id, store_name, city, lat, lon))
                store_id += 1

        # If output path is provided, save to CSV
        if output_path:
            stores_df = pd.DataFrame(
                [
                    {
                        "store_id": store.id,
                        "store_name": store.name,
                        "city": store.city,
                        "latitude": store.latitude,
                        "longitude": store.longitude,
                    }
                    for store in stores
                ]
            )
            stores_df.to_csv(output_path, index=False)
            print(f"Saved {len(stores)} stores to {output_path}")

        return stores


if __name__ == "__main__":
    generator = StoreGenerator()
    stores = generator.generate_stores("data/stores.csv")
    print(f"Generated {len(stores)} stores")
    for store in stores:
        print(store)
