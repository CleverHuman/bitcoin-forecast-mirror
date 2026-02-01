"""Trading strategies for backtesting.

Available strategies:
- CycleSignalStrategy: Primary strategy using halving cycle + technicals
- ForecastBasedStrategy: Trade based on forecast direction
- ForecastMomentumStrategy: Active trading on forecast momentum (few times/month)
- BuyAndHoldStrategy: Baseline benchmark
"""

from .base import BaseStrategy, Signal, StrategySignal, classify_signal
from .cycle_signals import CycleSignalStrategy
from .forecast_based import ForecastBasedStrategy
from .forecast_momentum import ForecastMomentumStrategy
from .buy_and_hold import BuyAndHoldStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "StrategySignal",
    "classify_signal",
    "CycleSignalStrategy",
    "ForecastBasedStrategy",
    "ForecastMomentumStrategy",
    "BuyAndHoldStrategy",
]
