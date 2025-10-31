from __future__ import annotations
import pandas as pd
from typing import Optional, List
from prophet import Prophet

class ProphetForecaster:
    def __init__(
        self,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
        holidays_df: Optional[pd.DataFrame] = None,
        seasonality_mode: str = "multiplicative",
        interval_width: float = 0.8
    ):
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays_df = holidays_df
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width

    def _build_model(self, extra_regressors: Optional[List[str]] = None) -> Prophet:
        m = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            holidays=self.holidays_df
        )
        if extra_regressors:
            for r in extra_regressors:
                m.add_regressor(r)
        return m

    def fit_predict(
        self,
        sales_df: pd.DataFrame,
        horizon_days: int = 14,
        regressors_df: Optional[pd.DataFrame] = None,
        extra_regressors: Optional[List[str]] = None,
        freq: str = "D",
        min_points_per_series: int = 10
    ) -> pd.DataFrame:
        s = sales_df.copy()
        s["ds"] = pd.to_datetime(s["date"])
        s["y"]  = pd.to_numeric(s["quantity"], errors="coerce").fillna(0.0)

        R = None
        reg_cols: List[str] = []
        if regressors_df is not None:
            R = regressors_df.copy()
            R["ds"] = pd.to_datetime(R["date"])
            merge_cols = ["ds","store_id","product_id"]
            reg_cols = [c for c in R.columns if c not in merge_cols]

        all_fc = []
        for (sid, pid), g in s.groupby(["store_id","product_id"]):
            g = g.sort_values("ds")
            if len(g) < min_points_per_series:
                continue

            train = g[["ds","y"]].copy()
            X = None
            use_regs = reg_cols if extra_regressors is None else extra_regressors
            if R is not None and use_regs:
                X = R[(R["store_id"] == sid) & (R["product_id"] == pid)][["ds"] + use_regs].sort_values("ds")
                train = train.merge(X, on="ds", how="left")

            m = self._build_model(extra_regressors=use_regs)
            m.fit(train)

            future = m.make_future_dataframe(periods=horizon_days, freq=freq)
            if X is not None:
                future = future.merge(X, on="ds", how="left")

            fc = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].copy()
            fc["store_id"] = int(sid)
            fc["product_id"] = int(pid)
            all_fc.append(fc)

        if not all_fc:
            return pd.DataFrame(columns=["store_id","product_id","ds","yhat","yhat_lower","yhat_upper"])
        return pd.concat(all_fc, ignore_index=True)
