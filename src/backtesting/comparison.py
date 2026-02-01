"""Compare multiple strategies side by side.

Run multiple strategies on the same data and compare their performance.
"""

from dataclasses import dataclass

import pandas as pd

from .strategies.base import BaseStrategy
from .runner import BacktestRunner, BacktestConfig
from .metrics import BacktestMetrics


@dataclass
class StrategyComparison:
    """Result of comparing multiple strategies."""

    results: dict[str, BacktestMetrics]
    summary: pd.DataFrame


def compare_strategies(
    df: pd.DataFrame,
    strategies: list[BaseStrategy],
    config: BacktestConfig | None = None,
    forecast: pd.DataFrame | None = None,
) -> StrategyComparison:
    """Run multiple strategies and compare results.

    Args:
        df: Historical data with 'ds' and 'y' columns.
        strategies: List of strategies to compare.
        config: Backtest configuration (shared across all strategies).
        forecast: Optional forecast for forecast-based strategies.

    Returns:
        StrategyComparison with all results and a summary table.
    """
    runner = BacktestRunner(df, config=config)
    results = {}

    for strategy in strategies:
        print(f"Running: {strategy.name}...")
        result = runner.run(strategy, forecast=forecast)
        results[strategy.name] = result

    # Build summary table
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            "Strategy": name,
            "Return %": metrics.total_return_pct,
            "CAGR %": metrics.cagr_pct,
            "vs B&H %": metrics.outperformance_pct,
            "Sharpe": metrics.sharpe_ratio,
            "Max DD %": metrics.max_drawdown_pct,
            "Calmar": metrics.calmar_ratio,
            "Win Rate %": metrics.win_rate_pct,
            "Trades": metrics.num_trades,
        })

    summary = pd.DataFrame(summary_data)

    # Sort by return
    summary = summary.sort_values("Return %", ascending=False)

    return StrategyComparison(results=results, summary=summary)


def print_comparison(comparison: StrategyComparison) -> None:
    """Print a comparison summary table."""
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON")
    print("=" * 100)
    print()
    print(comparison.summary.to_string(index=False))
    print()
    print("=" * 100)

    # Highlight winner
    best = comparison.summary.iloc[0]
    print(f"\nBest strategy by return: {best['Strategy']} ({best['Return %']:.1f}%)")

    # Best risk-adjusted
    best_sharpe_idx = comparison.summary["Sharpe"].idxmax()
    best_sharpe = comparison.summary.loc[best_sharpe_idx]
    print(f"Best risk-adjusted (Sharpe): {best_sharpe['Strategy']} (Sharpe {best_sharpe['Sharpe']:.2f})")
