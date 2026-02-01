"""Backtesting module for strategy evaluation.

Run trading strategies through historical data to measure performance.

Usage:
    from src.backtesting import BacktestRunner, BacktestConfig
    from src.backtesting.strategies import CycleSignalStrategy, BuyAndHoldStrategy

    runner = BacktestRunner(df)

    # Run single strategy
    result = runner.run(CycleSignalStrategy())

    # Compare strategies
    from src.backtesting import compare_strategies, print_comparison
    comparison = compare_strategies(df, [
        CycleSignalStrategy(),
        BuyAndHoldStrategy(),
    ])
    print_comparison(comparison)
"""

from .runner import BacktestRunner, BacktestConfig
from .metrics import BacktestMetrics, Trade, print_metrics_report
from .comparison import compare_strategies, print_comparison, StrategyComparison

__all__ = [
    "BacktestRunner",
    "BacktestConfig",
    "BacktestMetrics",
    "Trade",
    "print_metrics_report",
    "compare_strategies",
    "print_comparison",
    "StrategyComparison",
]
