import numpy as np
import pandas as pd


class DistanceCalculator:
    def __init__(self, stores, use_google_maps=False, api_key=None):
        """
        Initialize with store data and calculation method.
        Args:
            stores: List of Store objects or DataFrame with store data
        """
        self.stores = stores
        self.use_google_maps = use_google_maps
        self.api_key = api_key

        # Canonical DataFrame
        if isinstance(stores, pd.DataFrame):
            self.store_data = stores.copy()
        else:
            self.store_data = pd.DataFrame(
                [
                    {
                        "store_id": store.id,
                        "store_name": getattr(store, "name", None),
                        "city": getattr(store, "city", None),
                        "latitude": getattr(store, "latitude", None),
                        "longitude": getattr(store, "longitude", None),
                    }
                    for store in stores
                ]
            )

        # ---- Missing value handling (impute with averages) ----
        for col in ["store_id", "city", "latitude", "longitude"]:
            if col not in self.store_data.columns:
                self.store_data[col] = pd.NA

        self.store_data["store_id"] = self.store_data["store_id"].astype(str)
        self.store_data["city"] = (
            self.store_data["city"].astype(str).replace(
                {"None": pd.NA, "nan": pd.NA}).fillna("unknown")
        )
        self.store_data["latitude"] = pd.to_numeric(
            self.store_data["latitude"], errors="coerce")
        self.store_data["longitude"] = pd.to_numeric(
            self.store_data["longitude"], errors="coerce")

        # Per-city means
        city_means = (
            self.store_data.groupby("city", dropna=False)[
                ["latitude", "longitude"]]
            .mean(numeric_only=True)
        )

        def fill_from_city_mean(row):
            lat, lon = row["latitude"], row["longitude"]
            if pd.isna(lat) or pd.isna(lon):
                if row["city"] in city_means.index:
                    means = city_means.loc[row["city"]]
                    if pd.isna(lat) and pd.notna(means["latitude"]):
                        lat = means["latitude"]
                    if pd.isna(lon) and pd.notna(means["longitude"]):
                        lon = means["longitude"]
            return pd.Series([lat, lon], index=["latitude", "longitude"])

        self.store_data[["latitude", "longitude"]] = self.store_data.apply(
            fill_from_city_mean, axis=1)

        # Global fallback
        global_lat = self.store_data["latitude"].mean(numeric_only=True)
        global_lon = self.store_data["longitude"].mean(numeric_only=True)
        self.store_data["latitude"] = self.store_data["latitude"].fillna(
            global_lat)
        self.store_data["longitude"] = self.store_data["longitude"].fillna(
            global_lon)

    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
            np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return c * r

    def _fill_matrix_nas(self, mat: pd.DataFrame, fill_diagonal_zero: bool = True) -> pd.DataFrame:
        if fill_diagonal_zero:
            np.fill_diagonal(mat.values, 0.0)

        # Row-wise mean fill
        for i, idx in enumerate(mat.index):
            row = pd.to_numeric(mat.loc[idx], errors="coerce")
            # ignore diagonal at i
            non_diag = row.drop(labels=[idx], errors="ignore")
            mean_val = non_diag.mean(skipna=True)
            mat.loc[idx] = row.fillna(mean_val)

        # Global fallback
        global_mean = pd.to_numeric(mat.stack(), errors="coerce").mean()
        mat = mat.fillna(global_mean)
        return mat

    def generate_distance_matrix(self, output_path=None):
        print("Generating distance matrix...")

        store_ids = self.store_data["store_id"].tolist()

        # Vectorized haversine computation
        lat = self.store_data['latitude'].to_numpy(dtype=float)
        lon = self.store_data['longitude'].to_numpy(dtype=float)
        lat_rad = np.radians(lat)[:, None]  # (N,1)
        lon_rad = np.radians(lon)[:, None]
        dlat = lat_rad.T - lat_rad  # (N,N)
        dlon = lon_rad.T - lon_rad
        a = np.sin(dlat / 2.0) ** 2 + \
            np.cos(lat_rad) @ np.cos(lat_rad.T) * (np.sin(dlon / 2.0) ** 2)
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        r = 6371.0
        dist_mat = r * c
        np.fill_diagonal(dist_mat, 0.0)

        distance_matrix = pd.DataFrame(
            dist_mat, index=store_ids, columns=store_ids, dtype=float)

        if output_path:
            distance_matrix.to_csv(output_path)

        return distance_matrix

    def generate_transport_cost_matrix(self, distance_matrix=None, output_path=None):
        print("Generating transport cost matrix...")

        if distance_matrix is None:
            distance_matrix = self.generate_distance_matrix()

        transport_cost_matrix = pd.DataFrame(
            index=distance_matrix.index, columns=distance_matrix.columns, dtype=float
        )

        store_city_map = self.store_data.set_index(
            "store_id")["city"].fillna("unknown").to_dict()

        base_cost = 2000.0
        intercity_factor = 1.2
        distance_factors = {
            100: 1.2,
            500: 1.0,
            float("inf"): 0.5,
        }

        # Vectorized approach for cost: compute per-pair
        for i, from_id in enumerate(transport_cost_matrix.index):
            from_city = store_city_map.get(from_id, "unknown")
            row_dist = pd.to_numeric(
                distance_matrix.loc[from_id], errors="coerce")
            to_cities = [store_city_map.get(cid, "unknown")
                         for cid in transport_cost_matrix.columns]
            city_factors = np.array(
                [intercity_factor if from_city != c else 1.0 for c in to_cities], dtype=float)

            # Distance factors per column
            d = row_dist.to_numpy(dtype=float)
            dfactor = np.where(d < 100, 1.2, np.where(d < 500, 1.0, 0.5))

            costs = base_cost * d * city_factors * dfactor
            costs[i] = 0.0  # diagonal
            transport_cost_matrix.loc[from_id] = costs

        if output_path:
            transport_cost_matrix.to_csv(output_path)

        return transport_cost_matrix


if __name__ == "__main__":
    import pandas as pd

    # Load Olist sellers and geolocation datasets
    sellers = pd.read_csv("olist_sellers_dataset.csv")
    geo = pd.read_csv("olist_geolocation_dataset.csv")

    # Compute average lat/lon per zip prefix
    geo_prefix = (
        geo.groupby("geolocation_zip_code_prefix")[
            ["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "latitude", "geolocation_lng": "longitude"})
        .reset_index()
    )

    # Merge to form store table
    stores_df = sellers.merge(
        geo_prefix,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    )
    stores_df = stores_df.rename(
        columns={"seller_id": "store_id", "seller_city": "city"}
    )[["store_id", "city", "latitude", "longitude"]]
    stores_df = stores_df.drop_duplicates(
        subset=["store_id"]).reset_index(drop=True)

    # Instantiate and generate outputs
    calc = DistanceCalculator(stores_df)
    distance_matrix = calc.generate_distance_matrix("distance_matrix.csv")
    transport_matrix = calc.generate_transport_cost_matrix(
        distance_matrix, "transport_cost_matrix.csv")

    print("Distance and transport cost matrices generated successfully.")
