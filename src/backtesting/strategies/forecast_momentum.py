"""Forecast momentum trading strategy.

Trades multiple times per month based on:
1. Forecast momentum (is the forecast trending up or down?)
2. Price vs forecast divergence (is current price above/below forecast?)
3. Forecast confidence (width of prediction interval)

This is an active trading strategy that generates signals more frequently
than the cycle-based strategy.
"""

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, StrategySignal, classify_signal


class ForecastMomentumStrategy(BaseStrategy):
    """Active trading strategy based on forecast momentum and divergence.

    Generates signals by analyzing:
    - Forecast slope (7-day momentum)
    - Price vs forecast divergence
    - Optional: confidence interval width

    Trades more frequently than cycle strategy (~2-4 times per month).
    """

    def __init__(
        self,
        forecast_col: str = "yhat_ensemble",
        momentum_window: int = 7,
        divergence_threshold_pct: float = 5.0,
        momentum_threshold_pct: float = 2.0,
        min_trade_interval_days: int = 7,
        use_confidence_bands: bool = True,
    ):
        """Initialize the strategy.

        Args:
            forecast_col: Column name for forecast ('yhat' or 'yhat_ensemble').
            momentum_window: Days to compute forecast momentum (slope).
            divergence_threshold_pct: Min % divergence between price and forecast to trade.
            momentum_threshold_pct: Min % forecast change over momentum_window to signal.
            min_trade_interval_days: Minimum days between trades (avoid overtrading).
            use_confidence_bands: Use forecast confidence bands for signal strength.
        """
        self.forecast_col = forecast_col
        self.momentum_window = momentum_window
        self.divergence_threshold_pct = divergence_threshold_pct
        self.momentum_threshold_pct = momentum_threshold_pct
        self.min_trade_interval_days = min_trade_interval_days
        self.use_confidence_bands = use_confidence_bands

    @property
    def name(self) -> str:
        return "Forecast Momentum"

    @property
    def description(self) -> str:
        return f"Trade on {self.momentum_window}-day forecast momentum (min {self.min_trade_interval_days}d between trades)"

    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate signals based on forecast momentum and divergence."""
        self.validate_data(df)
        df = df.copy()

        if forecast is None:
            raise ValueError("ForecastMomentumStrategy requires a forecast DataFrame")

        # Merge forecast with price data
        df["ds"] = pd.to_datetime(df["ds"])
        forecast = forecast.copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"])

        # Get forecast column
        if self.forecast_col not in forecast.columns:
            if "yhat" in forecast.columns:
                self.forecast_col = "yhat"
            else:
                raise ValueError(f"Forecast missing {self.forecast_col} column")

        # Merge forecast into df
        forecast_cols = ["ds", self.forecast_col]
        if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
            forecast_cols.extend(["yhat_lower", "yhat_upper"])

        df = df.merge(forecast[forecast_cols], on="ds", how="left")

        # Initialize scores
        df["signal_score"] = 0.0
        df["signal_reason"] = ""

        # =================================================================
        # 1. FORECAST MOMENTUM (is forecast trending up or down?)
        # =================================================================
        df["forecast_momentum"] = (
            df[self.forecast_col].pct_change(periods=self.momentum_window) * 100
        )

        # Momentum score: positive momentum = bullish
        momentum_score = (df["forecast_momentum"] / self.momentum_threshold_pct).clip(-1, 1)
        df["momentum_score"] = momentum_score.fillna(0)

        # =================================================================
        # 2. PRICE VS FORECAST DIVERGENCE
        # =================================================================
        df["divergence_pct"] = ((df["y"] - df[self.forecast_col]) / df[self.forecast_col]) * 100

        # Divergence score:
        # - Price below forecast = bullish (buy the dip)
        # - Price above forecast = bearish (take profits)
        divergence_score = (-df["divergence_pct"] / self.divergence_threshold_pct).clip(-1, 1)
        df["divergence_score"] = divergence_score.fillna(0)

        # =================================================================
        # 3. CONFIDENCE BAND POSITION (optional)
        # =================================================================
        if self.use_confidence_bands and "yhat_lower" in df.columns:
            # Where is price relative to forecast bands?
            band_width = df["yhat_upper"] - df["yhat_lower"]
            price_position = (df["y"] - df["yhat_lower"]) / band_width.replace(0, 1)

            # Position score:
            # - Near lower band (0.0-0.3) = bullish
            # - Near upper band (0.7-1.0) = bearish
            # - Middle (0.3-0.7) = neutral
            band_score = (0.5 - price_position).clip(-0.5, 0.5) * 2
            df["band_score"] = band_score.fillna(0)
        else:
            df["band_score"] = 0

        # =================================================================
        # COMBINE SCORES
        # =================================================================
        # Weight: momentum 40%, divergence 40%, bands 20%
        if self.use_confidence_bands:
            df["raw_score"] = (
                df["momentum_score"] * 0.4 +
                df["divergence_score"] * 0.4 +
                df["band_score"] * 0.2
            )
        else:
            df["raw_score"] = (
                df["momentum_score"] * 0.5 +
                df["divergence_score"] * 0.5
            )

        # =================================================================
        # APPLY MINIMUM TRADE INTERVAL
        # =================================================================
        df["signal_score"] = self._apply_trade_interval(df, "raw_score")

        # Generate reasons
        df["signal_reason"] = df.apply(self._get_reason, axis=1)

        # Classify signals
        df["signal"] = df["signal_score"].apply(lambda s: classify_signal(s).value)

        return df

    def _apply_trade_interval(self, df: pd.DataFrame, score_col: str) -> pd.Series:
        """Apply minimum trade interval to avoid overtrading.

        Only allow signal changes every min_trade_interval_days.
        """
        scores = df[score_col].copy()
        result = pd.Series(0.0, index=df.index)

        last_signal_idx = None
        last_signal_direction = 0  # 1 = buy, -1 = sell, 0 = neutral

        for i, (idx, row) in enumerate(df.iterrows()):
            raw_score = scores.loc[idx]
            current_direction = 1 if raw_score > 0.2 else (-1 if raw_score < -0.2 else 0)

            # Check if enough days since last signal
            if last_signal_idx is not None:
                days_since = i - last_signal_idx
                if days_since < self.min_trade_interval_days:
                    # Too soon - hold previous position
                    result.loc[idx] = 0  # HOLD
                    continue

            # Check if direction changed (or strong enough signal)
            if current_direction != 0 and current_direction != last_signal_direction:
                result.loc[idx] = raw_score
                last_signal_idx = i
                last_signal_direction = current_direction
            elif abs(raw_score) > 0.5:
                # Strong signal in same direction - allow it
                result.loc[idx] = raw_score
                last_signal_idx = i
            else:
                result.loc[idx] = 0  # HOLD

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
                "forecast_momentum": latest.get("forecast_momentum"),
                "divergence_pct": latest.get("divergence_pct"),
                "momentum_score": latest.get("momentum_score"),
                "divergence_score": latest.get("divergence_score"),
            },
        )

    def _get_reason(self, row: pd.Series) -> str:
        """Generate human-readable signal reason."""
        score = row.get("signal_score", 0)
        momentum = row.get("forecast_momentum", 0)
        divergence = row.get("divergence_pct", 0)

        if abs(score) < 0.2:
            return "HOLD - No strong signal"

        parts = []

        # Momentum component
        if abs(momentum) > self.momentum_threshold_pct:
            direction = "up" if momentum > 0 else "down"
            parts.append(f"forecast trending {direction} ({momentum:+.1f}%)")

        # Divergence component
        if abs(divergence) > self.divergence_threshold_pct:
            position = "below" if divergence < 0 else "above"
            parts.append(f"price {position} forecast ({divergence:+.1f}%)")

        if not parts:
            parts.append("moderate signal strength")

        action = "BUY" if score > 0 else "SELL"
        return f"{action} - {', '.join(parts)}"
