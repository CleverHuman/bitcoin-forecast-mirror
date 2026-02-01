"""Base forecaster interface.

Forecasters produce price predictions. They are separate from strategies,
which decide when to buy/sell based on those predictions (or other signals).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ForecastResult:
    """Result from a forecaster."""

    forecast: pd.DataFrame  # Must have 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    model: Any  # The trained model (for inspection)
    config: dict  # Configuration used
    metrics: dict | None = None  # Optional evaluation metrics


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseForecaster":
        """Fit the forecaster to historical data.

        Args:
            df: DataFrame with 'ds' (date) and 'y' (price) columns.

        Returns:
            Self for chaining.
        """
        pass

    @abstractmethod
    def predict(self, periods: int) -> ForecastResult:
        """Generate forecast for future periods.

        Args:
            periods: Number of days to forecast.

        Returns:
            ForecastResult with predictions.
        """
        pass

    def fit_predict(self, df: pd.DataFrame, periods: int) -> ForecastResult:
        """Fit and predict in one call.

        Args:
            df: Historical data.
            periods: Days to forecast.

        Returns:
            ForecastResult with predictions.
        """
        return self.fit(df).predict(periods)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the forecaster."""
        pass
