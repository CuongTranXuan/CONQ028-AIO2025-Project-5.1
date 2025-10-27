import os

import numpy as np
import pandas as pd


class DistanceCalculator:
    def __init__(self, stores, use_google_maps=False, api_key=None):
        """
        Initialize with store data and calculation method.

        Args:
            stores: List of Store objects or DataFrame with store data
            use_google_maps: Whether to use Google Maps API (default: False)
            api_key: Google Maps API key (required if use_google_maps is True)
        """
        self.stores = stores
        self.use_google_maps = use_google_maps
        self.api_key = api_key

        # If stores is a DataFrame, convert to list of store-like objects
        if isinstance(stores, pd.DataFrame):
            self.store_data = stores
        else:
            self.store_data = pd.DataFrame(
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

    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    def generate_distance_matrix(self, output_path=None):
        """
        Generate a matrix of distances between all stores.

        Args:
            output_path: Optional path to save distance matrix to CSV

        Returns:
            DataFrame with distances between stores
        """
        print("Generating distance matrix...")

        num_stores = len(self.store_data)
        store_ids = self.store_data["store_id"].tolist()

        # Initialize empty matrix using integer store IDs
        distance_matrix = pd.DataFrame(index=store_ids, columns=store_ids)

        # Fill diagonal with zeros
        for store_id in store_ids:
            distance_matrix.loc[store_id, store_id] = 0

        if self.use_google_maps and self.api_key:
            # Use Google Maps Distance Matrix API
            try:
                from googlemaps import Client

                gmaps = Client(key=self.api_key)

                # Prepare origins and destinations
                origins = [
                    (row["latitude"], row["longitude"])
                    for _, row in self.store_data.iterrows()
                ]
                destinations = origins.copy()

                # Use Distance Matrix API
                result = gmaps.distance_matrix(
                    origins=origins,
                    destinations=destinations,
                    mode="driving",
                    units="metric",
                )

                # Parse results
                for i, row in enumerate(result["rows"]):
                    from_id = self.store_data.iloc[i]["store_id"]
                    for j, cell in enumerate(row["elements"]):
                        to_id = self.store_data.iloc[j]["store_id"]
                        if from_id != to_id:  # Skip diagonal (already set to 0)
                            # Distance in kilometers
                            distance_matrix.loc[from_id, to_id] = (
                                cell["distance"]["value"] / 1000
                            )

                print("Used Google Maps API for distance calculations")

            except ImportError:
                print(
                    "googlemaps package not installed. Falling back to haversine distance."
                )
                self.use_google_maps = False
            except Exception as e:
                print(
                    f"Error using Google Maps API: {str(e)}. Falling back to haversine distance."
                )
                self.use_google_maps = False

        if not self.use_google_maps:
            # Use haversine formula
            for i, row1 in self.store_data.iterrows():
                from_id = row1["store_id"]
                for j, row2 in self.store_data.iterrows():
                    to_id = row2["store_id"]
                    if from_id != to_id:  # Skip diagonal (already set to 0)
                        distance = self.calculate_haversine_distance(
                            row1["latitude"],
                            row1["longitude"],
                            row2["latitude"],
                            row2["longitude"],
                        )
                        distance_matrix.loc[from_id, to_id] = distance

            print("Used haversine formula for distance calculations")

        # If output path is provided, save to CSV
        if output_path:
            # Keep store IDs as integers throughout
            distance_matrix.index = distance_matrix.index.astype(int)
            distance_matrix.columns = distance_matrix.columns.astype(int)
            distance_matrix.to_csv(output_path)
            print(f"Saved distance matrix to {output_path}")

        return distance_matrix

    def generate_transport_cost_matrix(self, distance_matrix=None, output_path=None):
        """
        Generate a transport cost matrix based on distances.

        Args:
            distance_matrix: Optional pre-calculated distance matrix
            output_path: Optional path to save transport cost matrix to CSV

        Returns:
            DataFrame with transport costs between stores
        """
        print("Generating transport cost matrix...")

        # If no distance matrix provided, calculate it
        if distance_matrix is None:
            distance_matrix = self.generate_distance_matrix()

        # Create a transport cost matrix with the same dimensions
        transport_cost_matrix = pd.DataFrame(
            index=distance_matrix.index, columns=distance_matrix.columns
        )

        # Get city information for each store
        store_city_map = self.store_data.set_index("store_id")["city"].to_dict()

        # Cost parameters
        base_cost = 2_000  # Base cost per km in VND
        intercity_factor = 1.2  # 50% more expensive between cities

        # Distance factors (shorter distances are less efficient due to fixed costs)
        distance_factors = {
            100: 1.2,  # < 100 km: 20% more expensive per km
            500: 1.0,  # 50-500 km: standard rate
            9999: 0.5,  # > 500 km: 50% less expensive per km (economies of scale)
        }

        # Calculate transport costs
        for from_id in transport_cost_matrix.index:
            for to_id in transport_cost_matrix.columns:
                if from_id != to_id:  # Skip diagonal
                    # Get distance
                    distance = distance_matrix.loc[from_id, to_id]

                    # Determine city factor
                    from_city = store_city_map[from_id]
                    to_city = store_city_map[to_id]
                    city_factor = intercity_factor if from_city != to_city else 1.0

                    # Determine distance factor
                    distance_factor = None
                    for threshold, factor in sorted(distance_factors.items()):
                        if distance < threshold:
                            distance_factor = factor
                            break

                    if distance_factor is None:
                        distance_factor = list(distance_factors.values())[-1]

                    # Calculate cost
                    cost = base_cost * distance * city_factor * distance_factor
                    transport_cost_matrix.loc[from_id, to_id] = cost

        # If output path is provided, save to CSV
        if output_path:
            # Keep store IDs as integers throughout
            transport_cost_matrix.index = transport_cost_matrix.index.astype(int)
            transport_cost_matrix.columns = transport_cost_matrix.columns.astype(int)
            transport_cost_matrix.to_csv(output_path)
            print(f"Saved transport cost matrix to {output_path}")

        return transport_cost_matrix


if __name__ == "__main__":
    import os

    # Check if store data exists
    if not os.path.exists("data/stores.csv"):
        print("Store data not found. Please run store_generator.py first.")
        exit(1)

    # Load store data
    store_data = pd.read_csv("data/stores.csv")

    # Create calculator
    calculator = DistanceCalculator(store_data)

    # Generate matrices
    distance_matrix = calculator.generate_distance_matrix("data/distance_matrix.csv")
    transport_cost_matrix = calculator.generate_transport_cost_matrix(
        distance_matrix, "data/transport_cost_matrix.csv"
    )

    # Print summary statistics
    print("\nDistance Matrix Summary:")
    print(f"Average distance: {distance_matrix.values.mean():.2f} km")
    print(f"Maximum distance: {distance_matrix.values.max():.2f} km")

    print("\nTransport Cost Matrix Summary:")
    print(f"Average cost: {transport_cost_matrix.values.mean():,.0f} VND")
    print(f"Maximum cost: {transport_cost_matrix.values.max():,.0f} VND")
