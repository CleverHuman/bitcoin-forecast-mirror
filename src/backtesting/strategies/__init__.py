"""Trading strategies for backtesting.

Available strategies:
- CombinedStrategy: Best strategy - combines cycle + forecast + technicals
- CycleSignalStrategy: Primary strategy using halving cycle + technicals
- ForecastMomentumStrategy: Active trading on forecast momentum
- ForecastBasedStrategy: Simple forecast direction trading
- BuyAndHoldStrategy: Baseline benchmark
"""

from .base import BaseStrategy, Signal, StrategySignal, classify_signal
from .combined import CombinedStrategy
from .cycle_signals import CycleSignalStrategy
from .forecast_based import ForecastBasedStrategy
from .forecast_momentum import ForecastMomentumStrategy
from .buy_and_hold import BuyAndHoldStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "StrategySignal",
    "classify_signal",
    "CombinedStrategy",
    "CycleSignalStrategy",
    "ForecastBasedStrategy",
    "ForecastMomentumStrategy",
    "BuyAndHoldStrategy",
]
