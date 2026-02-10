"""Forecast-based trading strategy.

Generates signals based on forecast direction and magnitude.
Buy when forecast predicts price increase, sell when it predicts decrease.
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, StrategySignal, classify_signal


class ForecastBasedStrategy(BaseStrategy):
    """Strategy that trades based on forecast predictions.

    Buy when forecast > current price, sell when forecast < current price.
    Signal strength scales with the magnitude of predicted change.
    """

    def __init__(
        self,
        forecast_col: str = "yhat_ensemble",
        threshold_pct: float = 5.0,
        lookforward_days: int = 30,
    ):
        """Initialize the strategy.

        Args:
            forecast_col: Column name for forecast values ('yhat' or 'yhat_ensemble').
            threshold_pct: Minimum predicted change % to trigger signal.
            lookforward_days: Days ahead to check forecast.
        """
        self.forecast_col = forecast_col
        self.threshold_pct = threshold_pct
        self.lookforward_days = lookforward_days

    @property
    def name(self) -> str:
        return "Forecast-Based Strategy"

    @property
    def description(self) -> str:
        return f"Trade on {self.lookforward_days}-day forecast direction (>{self.threshold_pct}% threshold)"

    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate signals based on forecast vs current price."""
        self.validate_data(df)
        df = df.copy()

        if forecast is None:
            raise ValueError("ForecastBasedStrategy requires a forecast DataFrame")

        # Merge forecast with data
        forecast = forecast.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        forecast["ds"] = pd.to_datetime(forecast["ds"])

        # Get forecast column
        if self.forecast_col not in forecast.columns:
            if "yhat" in forecast.columns:
                self.forecast_col = "yhat"
            else:
                raise ValueError(f"Forecast missing {self.forecast_col} and yhat columns")

        # For each date, get the forecast N days ahead
        df["signal_score"] = 0.0
        df["signal_reason"] = ""

        for idx, row in df.iterrows():
            current_date = row["ds"]
            current_price = row["y"]
            target_date = current_date + pd.Timedelta(days=self.lookforward_days)

            # Find forecast for target date
            target_forecast = forecast[forecast["ds"] == target_date]

            if target_forecast.empty:
                # No forecast available
                df.loc[idx, "signal_score"] = 0
                df.loc[idx, "signal_reason"] = "No forecast available"
                continue

            forecast_price = target_forecast[self.forecast_col].iloc[0]

            if pd.isna(forecast_price) or pd.isna(current_price) or current_price <= 0:
                df.loc[idx, "signal_score"] = 0
                df.loc[idx, "signal_reason"] = "Invalid price data"
                continue

            # Calculate predicted change
            pct_change = (forecast_price - current_price) / current_price * 100

            # Convert to signal score
            if abs(pct_change) < self.threshold_pct:
                score = 0  # Below threshold
                reason = f"Forecast +{pct_change:.1f}% (below threshold)"
            else:
                # Scale score: ±10% change = ±0.5 score, ±20% = ±1.0
                score = np.clip(pct_change / 20, -1, 1)
                direction = "up" if pct_change > 0 else "down"
                reason = f"Forecast {direction} {abs(pct_change):.1f}% in {self.lookforward_days} days"

            df.loc[idx, "signal_score"] = score
            df.loc[idx, "signal_reason"] = reason

        # Classify signals
        df["signal"] = df["signal_score"].apply(lambda s: classify_signal(s).value)

        return df

    def get_current_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Get the most recent signal."""
        if df.empty:
            raise ValueError("Empty DataFrame")

        latest = df.loc[df["ds"].idxmax()]

        return StrategySignal(
            date=latest["ds"],
            signal=Signal(latest.get("signal", Signal.HOLD.value)),
            score=latest.get("signal_score", 0),
            reason=latest.get("signal_reason", ""),
            metadata={
                "forecast_col": self.forecast_col,
                "lookforward_days": self.lookforward_days,
            },
        )
