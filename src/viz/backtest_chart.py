"""Plot backtest results: equity curves, price, and trade markers."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from src.backtesting.comparison import StrategyComparison


# Colors for strategies (avoid green/red for lines so buy/sell markers stand out)
STRATEGY_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def plot_backtest(
    comparison: "StrategyComparison",
    initial_capital: float,
    best_name: str | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Plot backtest results: BTC price, equity curves for all strategies, buy & hold, and trade markers for best strategy.

    Args:
        comparison: Result of compare_strategies().
        initial_capital: Initial capital used in the backtest.
        best_name: Strategy name to show buy/sell markers for (default: best by return).
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure (call fig.savefig() or plt.show()).
    """
    results = comparison.results
    if not results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No backtest results to plot")
        return fig

    # Price and buy & hold from first result's equity curve (all share same dates/prices)
    first_metrics = next(iter(results.values()))
    eq = first_metrics.equity_curve
    if eq.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Empty equity curve")
        return fig

    dates = pd.to_datetime(eq["ds"])
    price = eq["price"].values
    start_price = price[0] if len(price) > 0 else 1.0
    buy_hold_value = initial_capital * (price / start_price)

    best_name = best_name or comparison.summary.iloc[0]["Strategy"]
    best_metrics = results[best_name]
    trades = best_metrics.trades

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[1.2, 1])
    fig.suptitle("Backtest: Equity vs Buy & Hold", fontsize=14, fontweight="bold")

    # --- Panel 1: BTC price + buy/sell markers for best strategy ---
    ax1 = axes[0]
    ax1.plot(dates, price, color="black", linewidth=1.2, label="BTC price", alpha=0.9)
    ax1.set_ylabel("BTC price (USD)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k"))

    buys = [t for t in trades if t.action == "BUY"]
    sells = [t for t in trades if t.action == "SELL"]
    if buys:
        ax1.scatter(
            [t.date for t in buys],
            [t.price for t in buys],
            color="green",
            marker="^",
            s=60,
            zorder=5,
            label=f"Buy ({len(buys)})",
            alpha=0.9,
        )
    if sells:
        ax1.scatter(
            [t.date for t in sells],
            [t.price for t in sells],
            color="red",
            marker="v",
            s=60,
            zorder=5,
            label=f"Sell ({len(sells)})",
            alpha=0.9,
        )
    ax1.legend(loc="upper left", fontsize=8)

    # --- Panel 2: Equity curves ---
    ax2 = axes[1]
    ax2.plot(dates, buy_hold_value, color="gray", linewidth=1.5, linestyle="--", label="Buy & Hold", alpha=0.8)
    ax2.axhline(initial_capital, color="gray", linewidth=0.8, alpha=0.5)

    for i, (name, metrics) in enumerate(results.items()):
        ec = metrics.equity_curve
        if ec.empty:
            continue
        color = STRATEGY_COLORS[i % len(STRATEGY_COLORS)]
        lbl = name + (f" ({metrics.total_return_pct:.0f}%)" if hasattr(metrics, "total_return_pct") else name)
        ax2.plot(pd.to_datetime(ec["ds"]), ec["equity"], color=color, linewidth=1.2, label=lbl, alpha=0.9)

    ax2.set_ylabel("Portfolio value (USD)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e3:.0f}k" if x >= 1e3 else f"${x:.0f}"))

    plt.tight_layout()
    return fig


def save_backtest_chart(
    comparison: "StrategyComparison",
    initial_capital: float,
    path: str | Path = "reports/backtest_chart.png",
    best_name: str | None = None,
) -> Path:
    """Plot backtest results and save to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_backtest(comparison, initial_capital, best_name=best_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
