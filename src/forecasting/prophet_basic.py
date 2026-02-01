"""Basic Prophet forecaster without cycle awareness.

Use this as a baseline to compare against cycle-aware forecasters.
"""

import pandas as pd
from prophet import Prophet

from .base import BaseForecaster, ForecastResult


class ProphetBasicForecaster(BaseForecaster):
    """Basic Prophet forecaster without halving cycle regressors.

    Good baseline for comparing cycle-aware forecasters.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        changepoint_prior_scale: float = 0.1,
        seasonality_mode: str = "multiplicative",
        interval_width: float = 0.95,
    ):
        """Initialize the forecaster.

        Args:
            yearly_seasonality: Include yearly patterns.
            weekly_seasonality: Include weekly patterns.
            changepoint_prior_scale: Flexibility of trend changes.
            seasonality_mode: 'additive' or 'multiplicative'.
            interval_width: Width of confidence intervals.
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width

        self._model: Prophet | None = None
        self._df: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return "Prophet (Basic)"

    def fit(self, df: pd.DataFrame) -> "ProphetBasicForecaster":
        """Fit Prophet to historical data."""
        self._df = df.copy()

        self._model = Prophet(
            interval_width=self.interval_width,
            growth="linear",
            daily_seasonality=False,
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            changepoint_range=0.8,
        )

        self._model.fit(self._df)
        return self

    def predict(self, periods: int) -> ForecastResult:
        """Generate forecast."""
        if self._model is None:
            raise ValueError("Must call fit() before predict()")

        future = self._model.make_future_dataframe(periods=periods, freq="d")
        forecast = self._model.predict(future)

        return ForecastResult(
            forecast=forecast,
            model=self._model,
            config={
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "changepoint_prior_scale": self.changepoint_prior_scale,
                "seasonality_mode": self.seasonality_mode,
                "interval_width": self.interval_width,
            },
        )
