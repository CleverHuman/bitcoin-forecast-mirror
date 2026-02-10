"""Data manager for merging historical, real-time, and forecast data."""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Callable

import pandas as pd

from src.trading.config import TradingConfig
from src.trading.exchange.base import BaseExchange

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data integration for live trading.

    Combines:
    - Historical data (from database/files)
    - Real-time price data (from exchange)
    - Forecast data (from ProphetCycleForecaster)

    Provides a unified interface for the strategy to access
    current market state and historical context.

    Attributes:
        exchange: Exchange for real-time prices
        config: Trading configuration
        historical_df: Historical price data
        current_price: Most recent price
        price_history: Recent price observations
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: TradingConfig,
        historical_df: pd.DataFrame | None = None,
    ):
        """Initialize data manager.

        Args:
            exchange: Exchange implementation for real-time data
            config: Trading configuration
            historical_df: Initial historical data (ds, y columns)
        """
        self.exchange = exchange
        self.config = config
        self._historical_df = historical_df
        self._current_price: Decimal | None = None
        self._last_price_update: datetime | None = None

        # Intraday price observations for extending historical data
        self._intraday_prices: list[tuple[datetime, Decimal]] = []

        # Price update callback
        self._price_callbacks: list[Callable[[str, Decimal], None]] = []

    @property
    def current_price(self) -> Decimal | None:
        """Get most recent price."""
        return self._current_price

    @property
    def historical_df(self) -> pd.DataFrame | None:
        """Get historical data with any intraday additions."""
        return self._historical_df

    def set_historical_data(self, df: pd.DataFrame) -> None:
        """Set or update historical data.

        Args:
            df: DataFrame with 'ds' and 'y' columns
        """
        if "ds" not in df.columns or "y" not in df.columns:
            raise ValueError("Historical data must have 'ds' and 'y' columns")

        self._historical_df = df.copy()
        self._historical_df["ds"] = pd.to_datetime(self._historical_df["ds"])
        self._historical_df = self._historical_df.sort_values("ds")

        logger.info(
            f"Historical data set: {len(df)} rows, "
            f"{df['ds'].min()} to {df['ds'].max()}"
        )

    def get_historical(self) -> pd.DataFrame:
        """Get historical data including today's price if available.

        Returns:
            DataFrame with complete historical data
        """
        if self._historical_df is None:
            raise ValueError("No historical data available")

        df = self._historical_df.copy()

        # Add today's price if we have real-time data
        if self._current_price is not None:
            today = pd.Timestamp(datetime.now().date())
            last_date = df["ds"].max()

            if today > last_date:
                # Add new row for today
                new_row = pd.DataFrame({
                    "ds": [today],
                    "y": [float(self._current_price)],
                })
                df = pd.concat([df, new_row], ignore_index=True)
            elif today == last_date:
                # Update today's value
                df.loc[df["ds"] == today, "y"] = float(self._current_price)

        return df

    async def fetch_current_price(self) -> Decimal:
        """Fetch current price from exchange.

        Returns:
            Current price
        """
        price = await self.exchange.get_price(self.config.symbol)
        self._update_price(self.config.symbol, price)
        return price

    def _update_price(self, symbol: str, price: Decimal) -> None:
        """Internal price update handler.

        Args:
            symbol: Trading pair symbol
            price: New price
        """
        self._current_price = price
        self._last_price_update = datetime.now()

        # Store intraday observation
        self._intraday_prices.append((datetime.now(), price))

        # Keep only last 24 hours of intraday data
        cutoff = datetime.now() - timedelta(hours=24)
        self._intraday_prices = [
            (ts, p) for ts, p in self._intraday_prices if ts > cutoff
        ]

        # Notify callbacks
        for callback in self._price_callbacks:
            try:
                callback(symbol, price)
            except Exception as e:
                logger.error(f"Error in price callback: {e}")

    def subscribe_price_updates(
        self,
        callback: Callable[[str, Decimal], None],
    ) -> None:
        """Subscribe to price updates.

        Args:
            callback: Function to call with (symbol, price)
        """
        self._price_callbacks.append(callback)

        # Also subscribe at exchange level
        self.exchange.subscribe_price(
            self.config.symbol,
            self._update_price,
        )

    def unsubscribe_price_updates(self) -> None:
        """Unsubscribe from price updates."""
        self._price_callbacks.clear()
        self.exchange.unsubscribe_price(self.config.symbol)

    def get_price_stats(self) -> dict[str, Any]:
        """Get statistics about recent price data.

        Returns:
            Dictionary with price statistics
        """
        if not self._intraday_prices:
            return {
                "current_price": float(self._current_price) if self._current_price else None,
                "last_update": self._last_price_update.isoformat() if self._last_price_update else None,
            }

        prices = [float(p) for _, p in self._intraday_prices]

        return {
            "current_price": float(self._current_price) if self._current_price else None,
            "last_update": self._last_price_update.isoformat() if self._last_price_update else None,
            "24h_high": max(prices),
            "24h_low": min(prices),
            "24h_avg": sum(prices) / len(prices),
            "observations_24h": len(prices),
        }

    def get_ohlcv(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get OHLCV data from historical data.

        Note: Historical data only has close prices, so O/H/L are approximated.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)

        Returns:
            DataFrame with OHLCV data
        """
        if self._historical_df is None:
            raise ValueError("No historical data available")

        df = self.get_historical()

        if start_date:
            df = df[df["ds"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["ds"] <= pd.Timestamp(end_date)]

        # Create OHLCV format (approximated since we only have daily close)
        ohlcv = pd.DataFrame({
            "timestamp": df["ds"],
            "open": df["y"],
            "high": df["y"],
            "low": df["y"],
            "close": df["y"],
            "volume": 0,  # Not available
        })

        return ohlcv

    def get_returns(self, periods: int = 30) -> pd.DataFrame:
        """Calculate daily returns.

        Args:
            periods: Number of days to include

        Returns:
            DataFrame with date and return columns
        """
        df = self.get_historical().tail(periods + 1)

        df = df.copy()
        df["return"] = df["y"].pct_change() * 100
        df = df.dropna()

        return df[["ds", "y", "return"]]

    def get_volatility(self, window: int = 30) -> float | None:
        """Calculate recent volatility (annualized).

        Args:
            window: Number of days for calculation

        Returns:
            Annualized volatility or None if insufficient data
        """
        returns = self.get_returns(window)

        if len(returns) < window // 2:
            return None

        daily_vol = returns["return"].std()
        annualized_vol = daily_vol * (365 ** 0.5)

        return annualized_vol

    def extend_historical_with_forecast(
        self,
        forecast_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine historical data with forecast for visualization.

        Args:
            forecast_df: Forecast DataFrame with ds, yhat columns

        Returns:
            Combined DataFrame with historical and forecast data
        """
        historical = self.get_historical()

        # Mark data source
        historical = historical.copy()
        historical["source"] = "historical"
        historical["yhat"] = historical["y"]

        forecast = forecast_df.copy()
        forecast["source"] = "forecast"
        if "y" not in forecast.columns:
            forecast["y"] = forecast.get("yhat_ensemble", forecast["yhat"])

        # Only keep forecast dates beyond historical
        last_historical = historical["ds"].max()
        forecast = forecast[forecast["ds"] > last_historical]

        combined = pd.concat([historical, forecast], ignore_index=True)
        combined = combined.sort_values("ds")

        return combined

    def get_market_state(self) -> dict[str, Any]:
        """Get current market state summary.

        Returns:
            Dictionary with current market information
        """
        price_stats = self.get_price_stats()

        state = {
            "symbol": self.config.symbol,
            "current_price": price_stats.get("current_price"),
            "last_update": price_stats.get("last_update"),
        }

        if self._historical_df is not None:
            df = self.get_historical()
            state["historical_days"] = len(df)
            state["historical_start"] = str(df["ds"].min())
            state["historical_end"] = str(df["ds"].max())

            # Recent performance
            if len(df) >= 2:
                state["24h_change_pct"] = (
                    (df["y"].iloc[-1] - df["y"].iloc[-2]) / df["y"].iloc[-2] * 100
                )

            if len(df) >= 8:
                state["7d_change_pct"] = (
                    (df["y"].iloc[-1] - df["y"].iloc[-8]) / df["y"].iloc[-8] * 100
                )

            if len(df) >= 31:
                state["30d_change_pct"] = (
                    (df["y"].iloc[-1] - df["y"].iloc[-31]) / df["y"].iloc[-31] * 100
                )

            volatility = self.get_volatility()
            if volatility:
                state["volatility_30d"] = volatility

        return state

    async def close(self) -> None:
        """Clean up resources."""
        self.unsubscribe_price_updates()
