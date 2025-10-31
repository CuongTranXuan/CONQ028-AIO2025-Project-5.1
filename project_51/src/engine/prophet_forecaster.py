# src/forecast/prophet_forecaster.py
from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any
from prophet import Prophet

class ProphetForecaster:
    """
    Forecast demand by (store_id, product_id) using Prophet.
    Expect: sales_df has columns: ['date','store_id','product_id','quantity'].
    Option: regressors_df has the same 'date','store_id','product_id' + auxiliary variables (promotion, price, ...).
    """
    def __init__(
        self,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
        holidays_df: Optional[pd.DataFrame] = None,   # cột 'ds','holiday'
        seasonality_mode: str = "additive",           # "additive" | "multiplicative"
        interval_width: float = 0.8
    ):
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays_df = holidays_df
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width

    def _build_model(self, extra_regressors: Optional[list[str]] = None) -> Prophet:
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
        extra_regressors: Optional[list[str]] = None,
        freq: str = "D"
    ) -> pd.DataFrame:
        """
        Returns forecast_df containing: ['store_id','product_id','ds','yhat','yhat_lower','yhat_upper'].
        """
        sales = sales_df.copy()
        sales["ds"] = pd.to_datetime(sales["date"])
        sales["y"]  = sales["quantity"]

        if regressors_df is not None:
            R = regressors_df.copy()
            R["ds"] = pd.to_datetime(R["date"])
            merge_cols = ["ds","store_id","product_id"]
            # keep regressor columns other than merge_cols
            reg_cols = [c for c in R.columns if c not in merge_cols]
        else:
            R, reg_cols = None, []

        all_forecasts = []
        for (sid, pid), g in sales.groupby(["store_id", "product_id"]):
            if len(g) < 10:  # too few data points then ignore
                continue

            g2 = g[["ds","y"]].sort_values("ds")

            # attach regressors of (sid,pid) pair if any
            X = None
            if R is not None:
                X = R[(R["store_id"] == sid) & (R["product_id"] == pid)]
                X = X[["ds"] + reg_cols].sort_values("ds")
                g2 = g2.merge(X, on="ds", how="left")

            model = self._build_model(extra_regressors=reg_cols if extra_regressors is None else extra_regressors)
            model.fit(g2)

            future = model.make_future_dataframe(periods=horizon_days, freq=freq)
            if R is not None:
                # concatenate corresponding regressors for future (if any)
                future = future.merge(X, on="ds", how="left")

            fc = model.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
            fc["store_id"] = sid
            fc["product_id"] = pid
            all_forecasts.append(fc)

        if not all_forecasts:
            return pd.DataFrame(columns=["store_id","product_id","ds","yhat","yhat_lower","yhat_upper"])

        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        return forecast_df
