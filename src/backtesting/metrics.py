"""Performance metrics for backtesting.

Computes risk-adjusted returns, drawdown analysis, and trade statistics.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Record of a single trade."""

    date: pd.Timestamp
    action: str  # "BUY" or "SELL"
    price: float
    amount_usd: float
    btc: float
    fee: float
    signal: str
    reason: str = ""


@dataclass
class BacktestMetrics:
    """Performance metrics from a backtest."""

    # Returns
    initial_capital: float
    final_value: float
    total_return_pct: float
    buy_hold_return_pct: float
    outperformance_pct: float
    cagr_pct: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    volatility_annual_pct: float
    calmar_ratio: float

    # Trade statistics
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_trade_duration_days: float

    # Time
    total_days: int
    time_in_market_pct: float

    # Data
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: list[Trade],
    initial_capital: float,
    df: pd.DataFrame,
) -> BacktestMetrics:
    """Compute all performance metrics.

    Args:
        equity_curve: DataFrame with 'ds', 'equity', 'btc_held' columns.
        trades: List of Trade objects.
        initial_capital: Starting capital.
        df: Original price data.

    Returns:
        BacktestMetrics with all computed values.
    """
    if equity_curve.empty:
        return _empty_metrics(initial_capital)

    # Final value
    final_value = equity_curve.iloc[-1]["equity"]

    # Buy and hold comparison
    start_price = df.iloc[0]["y"]
    final_price = df.iloc[-1]["y"]
    buy_hold_btc = initial_capital / start_price
    buy_hold_value = buy_hold_btc * final_price
    buy_hold_return = (buy_hold_value / initial_capital - 1) * 100

    total_return = (final_value / initial_capital - 1) * 100
    outperformance = (final_value / buy_hold_value - 1) * 100 if buy_hold_value > 0 else 0

    # Time metrics
    total_days = (df.iloc[-1]["ds"] - df.iloc[0]["ds"]).days
    years = total_days / 365.25
    cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Risk metrics from equity curve
    equity_curve = equity_curve.copy()
    equity_curve["returns"] = equity_curve["equity"].pct_change()
    daily_returns = equity_curve["returns"].dropna()

    if len(daily_returns) > 1:
        volatility = daily_returns.std() * np.sqrt(365) * 100
        avg_return = daily_returns.mean() * 365
        sharpe = avg_return / (daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0

        # Max drawdown
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["peak"]) / equity_curve["peak"]
        max_dd = abs(equity_curve["drawdown"].min()) * 100

        # Drawdown duration
        in_drawdown = equity_curve["drawdown"] < 0
        if in_drawdown.any():
            drawdown_groups = (~in_drawdown).cumsum()
            dd_durations = equity_curve.groupby(drawdown_groups).size()
            max_dd_duration = int(dd_durations.max())
        else:
            max_dd_duration = 0

        calmar = cagr / max_dd if max_dd > 0 else 0
    else:
        volatility = 0
        sharpe = 0
        max_dd = 0
        max_dd_duration = 0
        calmar = 0

    # Trade statistics
    trade_stats = _analyze_trades(trades)

    # Time in market
    time_in_market = (equity_curve["btc_held"] > 0).mean() * 100

    return BacktestMetrics(
        initial_capital=initial_capital,
        final_value=final_value,
        total_return_pct=total_return,
        buy_hold_return_pct=buy_hold_return,
        outperformance_pct=outperformance,
        cagr_pct=cagr,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        volatility_annual_pct=volatility,
        calmar_ratio=calmar,
        num_trades=len(trades),
        num_wins=trade_stats["num_wins"],
        num_losses=trade_stats["num_losses"],
        win_rate_pct=trade_stats["win_rate"],
        avg_win_pct=trade_stats["avg_win"],
        avg_loss_pct=trade_stats["avg_loss"],
        profit_factor=trade_stats["profit_factor"],
        avg_trade_duration_days=trade_stats["avg_duration"],
        total_days=total_days,
        time_in_market_pct=time_in_market,
        trades=trades,
        equity_curve=equity_curve,
    )


def _analyze_trades(trades: list[Trade]) -> dict[str, Any]:
    """Analyze trade performance."""
    if not trades:
        return {
            "num_wins": 0,
            "num_losses": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_duration": 0,
        }

    # Pair buys with sells
    buy_stack = []
    completed = []

    for trade in trades:
        if trade.action == "BUY":
            buy_stack.append(trade)
        elif trade.action == "SELL" and buy_stack:
            buy_trade = buy_stack.pop(0)
            pnl_pct = (trade.price - buy_trade.price) / buy_trade.price * 100
            duration = (trade.date - buy_trade.date).days
            completed.append({"pnl_pct": pnl_pct, "duration": duration})

    if not completed:
        return {
            "num_wins": 0,
            "num_losses": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_duration": 0,
        }

    wins = [t for t in completed if t["pnl_pct"] > 0]
    losses = [t for t in completed if t["pnl_pct"] <= 0]

    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0

    return {
        "num_wins": len(wins),
        "num_losses": len(losses),
        "win_rate": len(wins) / len(completed) * 100 if completed else 0,
        "avg_win": np.mean([t["pnl_pct"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pnl_pct"] for t in losses]) if losses else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_duration": np.mean([t["duration"] for t in completed]),
    }


def _empty_metrics(initial_capital: float) -> BacktestMetrics:
    """Return empty metrics when no data."""
    return BacktestMetrics(
        initial_capital=initial_capital,
        final_value=initial_capital,
        total_return_pct=0,
        buy_hold_return_pct=0,
        outperformance_pct=0,
        cagr_pct=0,
        sharpe_ratio=0,
        max_drawdown_pct=0,
        max_drawdown_duration_days=0,
        volatility_annual_pct=0,
        calmar_ratio=0,
        num_trades=0,
        num_wins=0,
        num_losses=0,
        win_rate_pct=0,
        avg_win_pct=0,
        avg_loss_pct=0,
        profit_factor=0,
        avg_trade_duration_days=0,
        total_days=0,
        time_in_market_pct=0,
    )


def print_metrics_report(metrics: BacktestMetrics, strategy_name: str = "") -> None:
    """Print a formatted metrics report."""
    header = f"BACKTEST REPORT: {strategy_name}" if strategy_name else "BACKTEST REPORT"
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    print("\n--- RETURNS ---")
    print(f"  Initial Capital:     ${metrics.initial_capital:>12,.0f}")
    print(f"  Final Value:         ${metrics.final_value:>12,.0f}")
    print(f"  Total Return:        {metrics.total_return_pct:>12.1f}%")
    print(f"  CAGR:                {metrics.cagr_pct:>12.1f}%")
    print(f"  Buy & Hold Return:   {metrics.buy_hold_return_pct:>12.1f}%")
    print(f"  Outperformance:      {metrics.outperformance_pct:>+12.1f}%")

    print("\n--- RISK METRICS ---")
    print(f"  Sharpe Ratio:        {metrics.sharpe_ratio:>12.2f}")
    print(f"  Max Drawdown:        {metrics.max_drawdown_pct:>12.1f}%")
    print(f"  Max DD Duration:     {metrics.max_drawdown_duration_days:>12} days")
    print(f"  Annual Volatility:   {metrics.volatility_annual_pct:>12.1f}%")
    print(f"  Calmar Ratio:        {metrics.calmar_ratio:>12.2f}")

    print("\n--- TRADE STATISTICS ---")
    print(f"  Total Trades:        {metrics.num_trades:>12}")
    print(f"  Winning Trades:      {metrics.num_wins:>12}")
    print(f"  Losing Trades:       {metrics.num_losses:>12}")
    print(f"  Win Rate:            {metrics.win_rate_pct:>12.1f}%")
    print(f"  Avg Win:             {metrics.avg_win_pct:>+12.1f}%")
    print(f"  Avg Loss:            {metrics.avg_loss_pct:>12.1f}%")
    print(f"  Profit Factor:       {metrics.profit_factor:>12.2f}")
    print(f"  Avg Trade Duration:  {metrics.avg_trade_duration_days:>12.0f} days")

    print("\n--- TIME ---")
    print(f"  Total Days:          {metrics.total_days:>12}")
    print(f"  Time in Market:      {metrics.time_in_market_pct:>12.1f}%")

    print("=" * 70 + "\n")
