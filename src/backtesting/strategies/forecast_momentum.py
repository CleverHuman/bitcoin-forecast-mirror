"""Forecast-based trading strategy.

Trades based on the relationship between current price and forecasted price:
- If predicted price > current price → potential upside → BUY
- If predicted price < current price → potential downside → SELL

Also considers:
- Forecast momentum (is forecast rising or falling?)
- Price trend relative to forecast (converging or diverging?)
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, StrategySignal, classify_signal


class ForecastMomentumStrategy(BaseStrategy):
    """Trade based on predicted price vs current price.

    Core logic:
    - Predicted upside (forecast > price) → BUY
    - Predicted downside (forecast < price) → SELL

    Signal strength increases when:
    - Large gap between price and forecast
    - Forecast momentum confirms direction
    - Price trending toward (not away from) forecast

    Example:
        Price: $80k, Forecast: $100k → 25% predicted upside → BUY
        Price: $100k, Forecast: $80k → 20% predicted downside → SELL
    """

    def __init__(
        self,
        forecast_col: str = "yhat_ensemble",
        lookforward_days: int = 30,
        min_upside_pct: float = 5.0,
        min_trade_interval_days: int = 7,
    ):
        """Initialize the strategy.

        Args:
            forecast_col: Column name for forecast ('yhat' or 'yhat_ensemble').
            lookforward_days: Days ahead to look at forecast.
            min_upside_pct: Minimum predicted change % to trigger signal.
            min_trade_interval_days: Minimum days between trades.
        """
        self.forecast_col = forecast_col
        self.lookforward_days = lookforward_days
        self.min_upside_pct = min_upside_pct
        self.min_trade_interval_days = min_trade_interval_days

    @property
    def name(self) -> str:
        return "Forecast Momentum"

    @property
    def description(self) -> str:
        return f"Trade on {self.lookforward_days}-day predicted upside/downside (>{self.min_upside_pct}%)"

    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate signals based on predicted vs current price."""
        self.validate_data(df)
        df = df.copy()

        if forecast is None:
            raise ValueError("ForecastMomentumStrategy requires a forecast DataFrame")

        df["ds"] = pd.to_datetime(df["ds"])
        forecast = forecast.copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"])

        # Get forecast column
        if self.forecast_col not in forecast.columns:
            self.forecast_col = "yhat" if "yhat" in forecast.columns else None
            if self.forecast_col is None:
                raise ValueError("Forecast missing yhat column")

        # Build lookup for forecast values
        forecast_lookup = forecast.set_index("ds")[self.forecast_col].to_dict()

        # Also get current-day forecast for comparison
        current_forecast_lookup = forecast.set_index("ds")[self.forecast_col].to_dict()

        # Compute forecast momentum (is forecast trending up or down?)
        forecast_sorted = forecast.sort_values("ds")
        forecast_sorted["forecast_momentum"] = (
            forecast_sorted[self.forecast_col].pct_change(periods=7) * 100
        )
        momentum_lookup = forecast_sorted.set_index("ds")["forecast_momentum"].to_dict()

        # Initialize columns
        df["predicted_price"] = np.nan
        df["current_forecast"] = np.nan
        df["predicted_change_pct"] = np.nan
        df["forecast_momentum"] = np.nan
        df["price_vs_forecast_pct"] = np.nan
        df["signal_score"] = 0.0
        df["signal_reason"] = ""

        for idx, row in df.iterrows():
            current_date = row["ds"]
            current_price = row["y"]

            if pd.isna(current_price) or current_price <= 0:
                continue

            # Get forecast for N days ahead
            target_date = current_date + pd.Timedelta(days=self.lookforward_days)
            predicted_price = forecast_lookup.get(target_date)

            # Get current-day forecast (what model thinks price should be today)
            current_forecast = current_forecast_lookup.get(current_date)

            # Get forecast momentum
            forecast_mom = momentum_lookup.get(current_date, 0)

            if predicted_price is None:
                continue

            df.loc[idx, "predicted_price"] = predicted_price
            df.loc[idx, "current_forecast"] = current_forecast
            df.loc[idx, "forecast_momentum"] = forecast_mom

            # =================================================================
            # 1. PREDICTED CHANGE: forecast vs current price
            # =================================================================
            # This is the main signal: how much upside/downside is predicted?
            predicted_change_pct = (predicted_price / current_price - 1) * 100
            df.loc[idx, "predicted_change_pct"] = predicted_change_pct

            # =================================================================
            # 2. PRICE VS CURRENT FORECAST: is price above or below where it "should" be?
            # =================================================================
            if current_forecast and current_forecast > 0:
                price_vs_forecast = (current_price / current_forecast - 1) * 100
                df.loc[idx, "price_vs_forecast_pct"] = price_vs_forecast
            else:
                price_vs_forecast = 0

            # =================================================================
            # COMPUTE SIGNAL SCORE
            # =================================================================
            score = 0.0
            reasons = []

            # Primary signal: predicted upside/downside
            if abs(predicted_change_pct) >= self.min_upside_pct:
                # Scale: 10% predicted change = 0.5 score, 20% = 1.0
                base_score = np.clip(predicted_change_pct / 20, -1, 1)
                score += base_score * 0.5

                direction = "upside" if predicted_change_pct > 0 else "downside"
                reasons.append(f"{abs(predicted_change_pct):.1f}% predicted {direction}")

            # Boost: price below forecast + bullish momentum = strong buy
            if price_vs_forecast < -self.min_upside_pct and forecast_mom > 0:
                score += 0.3
                reasons.append("price below rising forecast")

            # Boost: price above forecast + bearish momentum = strong sell
            elif price_vs_forecast > self.min_upside_pct and forecast_mom < 0:
                score -= 0.3
                reasons.append("price above falling forecast")

            # Moderate boost: trending in right direction
            if predicted_change_pct > 0 and forecast_mom > 2:
                score += 0.2
                reasons.append("bullish momentum")
            elif predicted_change_pct < 0 and forecast_mom < -2:
                score -= 0.2
                reasons.append("bearish momentum")

            df.loc[idx, "signal_score"] = np.clip(score, -1, 1)

            if reasons:
                action = "BUY" if score > 0.2 else ("SELL" if score < -0.2 else "HOLD")
                df.loc[idx, "signal_reason"] = f"{action} - {', '.join(reasons)}"
            else:
                df.loc[idx, "signal_reason"] = "HOLD - No strong signal"

        # Apply minimum trade interval
        df["signal_score"] = self._apply_trade_interval(df)

        # Classify signals
        df["signal"] = df["signal_score"].apply(lambda s: classify_signal(s).value)

        return df

    def _apply_trade_interval(self, df: pd.DataFrame) -> pd.Series:
        """Apply minimum trade interval to avoid overtrading."""
        scores = df["signal_score"].copy()
        result = pd.Series(0.0, index=df.index)

        last_trade_idx = None
        last_direction = 0

        for i, (idx, row) in enumerate(df.iterrows()):
            raw_score = scores.loc[idx]
            current_direction = 1 if raw_score > 0.2 else (-1 if raw_score < -0.2 else 0)

            # Check days since last trade
            if last_trade_idx is not None:
                days_since = i - last_trade_idx
                if days_since < self.min_trade_interval_days:
                    result.loc[idx] = 0  # HOLD
                    continue

            # Signal if direction changed or strong signal
            if current_direction != 0 and (current_direction != last_direction or abs(raw_score) > 0.6):
                result.loc[idx] = raw_score
                last_trade_idx = i
                last_direction = current_direction
            else:
                result.loc[idx] = 0

        return result

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
                "current_price": latest.get("y"),
                "predicted_price": latest.get("predicted_price"),
                "predicted_change_pct": latest.get("predicted_change_pct"),
                "price_vs_forecast_pct": latest.get("price_vs_forecast_pct"),
                "forecast_momentum": latest.get("forecast_momentum"),
            },
        )
