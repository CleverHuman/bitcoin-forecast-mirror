"""Buy and Hold baseline strategy.

Always signals BUY on day 1, then HOLD forever.
Used as a benchmark to compare active strategies against.
"""

import pandas as pd

from .base import BaseStrategy, Signal, StrategySignal


class BuyAndHoldStrategy(BaseStrategy):
    """Buy and hold baseline strategy.

    Buy on day 1, hold forever. Used as benchmark for comparison.
    """

    @property
    def name(self) -> str:
        return "Buy and Hold"

    @property
    def description(self) -> str:
        return "Buy on day 1, hold forever (benchmark)"

    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate buy and hold signals."""
        self.validate_data(df)
        df = df.copy()
        df = df.sort_values("ds").reset_index(drop=True)

        # Buy on first day, hold forever
        df["signal"] = Signal.HOLD.value
        df["signal_score"] = 0.0
        df["signal_reason"] = "Hold position"

        if len(df) > 0:
            df.loc[0, "signal"] = Signal.STRONG_BUY.value
            df.loc[0, "signal_score"] = 1.0
            df.loc[0, "signal_reason"] = "Initial buy (buy and hold)"

        return df

    def get_current_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Get the most recent signal (always HOLD after first day)."""
        if df.empty:
            raise ValueError("Empty DataFrame")

        latest = df.loc[df["ds"].idxmax()]

        return StrategySignal(
            date=latest["ds"],
            signal=Signal(latest.get("signal", Signal.HOLD.value)),
            score=latest.get("signal_score", 0),
            reason=latest.get("signal_reason", "Hold position"),
        )
