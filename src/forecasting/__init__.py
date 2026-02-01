"""Forecasting module for Bitcoin price predictions.

Forecasters produce price predictions. They are separate from strategies,
which decide when to buy/sell.

Available forecasters:
- ProphetBasicForecaster: Basic Prophet without cycle awareness (baseline)
- ProphetCycleForecaster: Prophet with halving cycle regressors (main)
"""

from .base import BaseForecaster, ForecastResult
from .prophet_basic import ProphetBasicForecaster
from .prophet_cycle import ProphetCycleForecaster

__all__ = [
    "BaseForecaster",
    "ForecastResult",
    "ProphetBasicForecaster",
    "ProphetCycleForecaster",
]
