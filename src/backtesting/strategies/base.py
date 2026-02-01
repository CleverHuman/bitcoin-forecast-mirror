"""Base strategy interface.

Strategies decide when to buy/sell. They can use forecasts, technical indicators,
cycle signals, or any combination thereof.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class Signal(Enum):
    """Trading signals."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class StrategySignal:
    """A signal at a specific point in time."""

    date: pd.Timestamp
    signal: Signal
    score: float  # -1 (strong sell) to +1 (strong buy)
    reason: str  # Human-readable explanation
    metadata: dict | None = None  # Strategy-specific data


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate buy/sell signals for the given data.

        Args:
            df: DataFrame with 'ds' (date) and 'y' (price) columns.
            forecast: Optional forecast DataFrame with predictions.

        Returns:
            DataFrame with added columns:
            - 'signal': Signal enum value (strong_buy, buy, hold, sell, strong_sell)
            - 'signal_score': Numeric score from -1 to +1
            - 'signal_reason': Human-readable explanation
        """
        pass

    @abstractmethod
    def get_current_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Get the most recent signal.

        Args:
            df: DataFrame with signals already generated.

        Returns:
            StrategySignal for the most recent date.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the strategy."""
        pass

    @property
    def description(self) -> str:
        """Strategy description."""
        return ""

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate that the DataFrame has required columns."""
        required = ["ds", "y"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")


def classify_signal(score: float) -> Signal:
    """Convert numeric score to Signal enum."""
    if score >= 0.5:
        return Signal.STRONG_BUY
    elif score >= 0.2:
        return Signal.BUY
    elif score >= -0.2:
        return Signal.HOLD
    elif score >= -0.5:
        return Signal.SELL
    else:
        return Signal.STRONG_SELL
