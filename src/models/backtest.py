"""Advanced backtesting engine for signal strategies.

Features:
- Performance metrics (Sharpe, max drawdown, win rate)
- Transaction costs and slippage
- Risk management (stop loss, take profit)
- Equity curve tracking
- Walk-forward validation
- Parameter optimization
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .signals import SignalType


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 10000
    position_size: float = 0.25  # Fraction of capital per trade
    max_position: float = 1.0  # Max fraction of capital in BTC
    fee_pct: float = 0.1  # Trading fee (0.1% = 10 bps)
    slippage_pct: float = 0.05  # Slippage estimate
    stop_loss_pct: float | None = None  # e.g., 0.10 = 10% stop loss
    take_profit_pct: float | None = None  # e.g., 0.50 = 50% take profit
    trailing_stop_pct: float | None = None  # e.g., 0.15 = 15% trailing stop
    min_trade_usd: float = 100  # Minimum trade size


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
    reason: str = ""  # Why the trade happened


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    # Core results
    initial_capital: float
    final_value: float
    total_return_pct: float
    buy_hold_return_pct: float
    outperformance_pct: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    volatility_annual_pct: float
    calmar_ratio: float  # CAGR / Max Drawdown

    # Trade statistics
    num_trades: int
    num_wins: int
    num_losses: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float  # Gross profit / Gross loss
    avg_trade_duration_days: float

    # Time metrics
    cagr_pct: float  # Compound Annual Growth Rate
    total_days: int
    time_in_market_pct: float

    # Data
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a full backtest with risk management and metrics.

    Args:
        df: DataFrame with 'ds', 'y', 'signal', 'signal_score' columns.
        config: Backtest configuration. Uses defaults if None.

    Returns:
        BacktestResult with full metrics and trade history.
    """
    if config is None:
        config = BacktestConfig()

    df = df.copy().sort_values("ds").reset_index(drop=True)

    # State
    capital = config.initial_capital
    btc_held = 0.0
    entry_price = 0.0
    highest_since_entry = 0.0
    trades: list[Trade] = []
    equity_history = []

    prev_signal = SignalType.HOLD.value

    for i, row in df.iterrows():
        date = row["ds"]
        price = row["y"]
        signal = row.get("signal", SignalType.HOLD.value)

        if pd.isna(price) or price <= 0:
            continue

        # Track equity
        portfolio_value = capital + (btc_held * price)
        equity_history.append({
            "ds": date,
            "equity": portfolio_value,
            "capital": capital,
            "btc_value": btc_held * price,
            "btc_held": btc_held,
            "price": price,
        })

        # Update highest price since entry (for trailing stop)
        if btc_held > 0:
            highest_since_entry = max(highest_since_entry, price)

        # Check stop loss / take profit / trailing stop
        if btc_held > 0 and entry_price > 0:
            pnl_pct = (price - entry_price) / entry_price

            # Stop loss
            if config.stop_loss_pct and pnl_pct <= -config.stop_loss_pct:
                sell_btc = btc_held
                gross = sell_btc * price
                fee = gross * (config.fee_pct + config.slippage_pct) / 100
                capital += gross - fee
                trades.append(Trade(
                    date=date, action="SELL", price=price,
                    amount_usd=gross, btc=sell_btc, fee=fee,
                    signal=signal, reason="stop_loss"
                ))
                btc_held = 0.0
                entry_price = 0.0
                continue

            # Take profit
            if config.take_profit_pct and pnl_pct >= config.take_profit_pct:
                sell_btc = btc_held
                gross = sell_btc * price
                fee = gross * (config.fee_pct + config.slippage_pct) / 100
                capital += gross - fee
                trades.append(Trade(
                    date=date, action="SELL", price=price,
                    amount_usd=gross, btc=sell_btc, fee=fee,
                    signal=signal, reason="take_profit"
                ))
                btc_held = 0.0
                entry_price = 0.0
                continue

            # Trailing stop
            if config.trailing_stop_pct and highest_since_entry > 0:
                trailing_stop_price = highest_since_entry * (1 - config.trailing_stop_pct)
                if price <= trailing_stop_price:
                    sell_btc = btc_held
                    gross = sell_btc * price
                    fee = gross * (config.fee_pct + config.slippage_pct) / 100
                    capital += gross - fee
                    trades.append(Trade(
                        date=date, action="SELL", price=price,
                        amount_usd=gross, btc=sell_btc, fee=fee,
                        signal=signal, reason="trailing_stop"
                    ))
                    btc_held = 0.0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    continue

        # Signal-based trading
        if signal in [SignalType.STRONG_BUY.value, SignalType.BUY.value]:
            if prev_signal not in [SignalType.STRONG_BUY.value, SignalType.BUY.value]:
                # Check position limit
                current_btc_value = btc_held * price
                current_position_pct = current_btc_value / portfolio_value if portfolio_value > 0 else 0

                if current_position_pct < config.max_position:
                    buy_amount = min(
                        capital * config.position_size,
                        portfolio_value * config.max_position - current_btc_value
                    )

                    if buy_amount >= config.min_trade_usd:
                        fee = buy_amount * (config.fee_pct + config.slippage_pct) / 100
                        net_amount = buy_amount - fee
                        btc_bought = net_amount / price
                        btc_held += btc_bought
                        capital -= buy_amount

                        if entry_price == 0:
                            entry_price = price
                            highest_since_entry = price

                        trades.append(Trade(
                            date=date, action="BUY", price=price,
                            amount_usd=buy_amount, btc=btc_bought, fee=fee,
                            signal=signal, reason="signal"
                        ))

        elif signal in [SignalType.STRONG_SELL.value, SignalType.SELL.value]:
            if prev_signal not in [SignalType.STRONG_SELL.value, SignalType.SELL.value]:
                if btc_held > 0:
                    sell_btc = btc_held * config.position_size
                    gross = sell_btc * price
                    fee = gross * (config.fee_pct + config.slippage_pct) / 100
                    capital += gross - fee
                    btc_held -= sell_btc

                    if btc_held < 0.0001:  # Effectively zero
                        btc_held = 0.0
                        entry_price = 0.0
                        highest_since_entry = 0.0

                    trades.append(Trade(
                        date=date, action="SELL", price=price,
                        amount_usd=gross, btc=sell_btc, fee=fee,
                        signal=signal, reason="signal"
                    ))

        prev_signal = signal

    # Build equity curve
    equity_df = pd.DataFrame(equity_history)
    if equity_df.empty:
        equity_df = pd.DataFrame({"ds": [], "equity": []})

    # Calculate metrics
    final_price = df.iloc[-1]["y"]
    final_value = capital + (btc_held * final_price)

    # Buy and hold
    start_price = df.iloc[0]["y"]
    buy_hold_btc = config.initial_capital / start_price
    buy_hold_value = buy_hold_btc * final_price
    buy_hold_return = (buy_hold_value / config.initial_capital - 1) * 100

    total_return = (final_value / config.initial_capital - 1) * 100
    outperformance = (final_value / buy_hold_value - 1) * 100 if buy_hold_value > 0 else 0

    # Time metrics
    total_days = (df.iloc[-1]["ds"] - df.iloc[0]["ds"]).days
    years = total_days / 365.25
    cagr = ((final_value / config.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Risk metrics from equity curve
    if len(equity_df) > 1:
        equity_df["returns"] = equity_df["equity"].pct_change()
        daily_returns = equity_df["returns"].dropna()

        volatility = daily_returns.std() * np.sqrt(365) * 100 if len(daily_returns) > 1 else 0
        avg_return = daily_returns.mean() * 365
        sharpe = avg_return / (daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0

        # Max drawdown
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
        max_dd = abs(equity_df["drawdown"].min()) * 100

        # Drawdown duration
        in_drawdown = equity_df["drawdown"] < 0
        if in_drawdown.any():
            drawdown_groups = (~in_drawdown).cumsum()
            dd_durations = equity_df.groupby(drawdown_groups).size()
            max_dd_duration = dd_durations.max()
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
    trade_results = _analyze_trades(trades, df)

    # Time in market
    if len(equity_df) > 0:
        time_in_market = (equity_df["btc_held"] > 0).mean() * 100
    else:
        time_in_market = 0

    return BacktestResult(
        initial_capital=config.initial_capital,
        final_value=final_value,
        total_return_pct=total_return,
        buy_hold_return_pct=buy_hold_return,
        outperformance_pct=outperformance,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        max_drawdown_duration_days=int(max_dd_duration),
        volatility_annual_pct=volatility,
        calmar_ratio=calmar,
        num_trades=len(trades),
        num_wins=trade_results["num_wins"],
        num_losses=trade_results["num_losses"],
        win_rate_pct=trade_results["win_rate"],
        avg_win_pct=trade_results["avg_win"],
        avg_loss_pct=trade_results["avg_loss"],
        profit_factor=trade_results["profit_factor"],
        avg_trade_duration_days=trade_results["avg_duration"],
        cagr_pct=cagr,
        total_days=total_days,
        time_in_market_pct=time_in_market,
        trades=trades,
        equity_curve=equity_df,
    )


def _analyze_trades(trades: list[Trade], df: pd.DataFrame) -> dict[str, Any]:
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

    # Pair buys with sells to calculate trade P&L
    buy_stack = []
    completed_trades = []

    for trade in trades:
        if trade.action == "BUY":
            buy_stack.append(trade)
        elif trade.action == "SELL" and buy_stack:
            buy_trade = buy_stack.pop(0)  # FIFO
            pnl_pct = (trade.price - buy_trade.price) / buy_trade.price * 100
            duration = (trade.date - buy_trade.date).days
            completed_trades.append({
                "pnl_pct": pnl_pct,
                "duration": duration,
                "buy_price": buy_trade.price,
                "sell_price": trade.price,
            })

    if not completed_trades:
        return {
            "num_wins": 0,
            "num_losses": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_duration": 0,
        }

    wins = [t for t in completed_trades if t["pnl_pct"] > 0]
    losses = [t for t in completed_trades if t["pnl_pct"] <= 0]

    gross_profit = sum(t["pnl_pct"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl_pct"] for t in losses)) if losses else 0

    return {
        "num_wins": len(wins),
        "num_losses": len(losses),
        "win_rate": len(wins) / len(completed_trades) * 100 if completed_trades else 0,
        "avg_win": np.mean([t["pnl_pct"] for t in wins]) if wins else 0,
        "avg_loss": np.mean([t["pnl_pct"] for t in losses]) if losses else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_duration": np.mean([t["duration"] for t in completed_trades]),
    }


def walk_forward_backtest(
    df: pd.DataFrame,
    train_days: int = 365,
    test_days: int = 90,
    config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    """Walk-forward validation: train on past, test on future.

    Args:
        df: Full dataset with signals.
        train_days: Days to use for "training" (signal generation).
        test_days: Days to test on.
        config: Backtest configuration.

    Returns:
        List of BacktestResult for each test period.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    results = []

    start_idx = 0
    while True:
        train_end_idx = start_idx + train_days
        test_end_idx = train_end_idx + test_days

        if test_end_idx > len(df):
            break

        test_df = df.iloc[train_end_idx:test_end_idx].copy()

        if len(test_df) > 0:
            result = run_backtest(test_df, config)
            results.append(result)

        start_idx += test_days  # Roll forward

    return results


def optimize_parameters(
    df: pd.DataFrame,
    param_grid: dict[str, list],
    metric: str = "sharpe_ratio",
) -> tuple[dict, BacktestResult]:
    """Grid search for optimal backtest parameters.

    Args:
        df: DataFrame with signals.
        param_grid: Dict of parameter names to lists of values to try.
            e.g., {"position_size": [0.1, 0.25, 0.5], "stop_loss_pct": [0.05, 0.1, 0.15]}
        metric: Metric to optimize (from BacktestResult fields).

    Returns:
        Tuple of (best_params, best_result).
    """
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_params = None
    best_result = None
    best_metric = float("-inf")

    for values in product(*param_values):
        params = dict(zip(param_names, values))
        config = BacktestConfig(**params)

        result = run_backtest(df, config)
        metric_value = getattr(result, metric, 0)

        if metric_value > best_metric:
            best_metric = metric_value
            best_params = params
            best_result = result

    return best_params, best_result


def print_backtest_report(result: BacktestResult) -> None:
    """Print a formatted backtest report."""
    print("\n" + "=" * 70)
    print("BACKTEST REPORT")
    print("=" * 70)

    print("\n--- RETURNS ---")
    print(f"  Initial Capital:     ${result.initial_capital:>12,.0f}")
    print(f"  Final Value:         ${result.final_value:>12,.0f}")
    print(f"  Total Return:        {result.total_return_pct:>12.1f}%")
    print(f"  CAGR:                {result.cagr_pct:>12.1f}%")
    print(f"  Buy & Hold Return:   {result.buy_hold_return_pct:>12.1f}%")
    print(f"  Outperformance:      {result.outperformance_pct:>+12.1f}%")

    print("\n--- RISK METRICS ---")
    print(f"  Sharpe Ratio:        {result.sharpe_ratio:>12.2f}")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:>12.1f}%")
    print(f"  Max DD Duration:     {result.max_drawdown_duration_days:>12} days")
    print(f"  Annual Volatility:   {result.volatility_annual_pct:>12.1f}%")
    print(f"  Calmar Ratio:        {result.calmar_ratio:>12.2f}")

    print("\n--- TRADE STATISTICS ---")
    print(f"  Total Trades:        {result.num_trades:>12}")
    print(f"  Winning Trades:      {result.num_wins:>12}")
    print(f"  Losing Trades:       {result.num_losses:>12}")
    print(f"  Win Rate:            {result.win_rate_pct:>12.1f}%")
    print(f"  Avg Win:             {result.avg_win_pct:>+12.1f}%")
    print(f"  Avg Loss:            {result.avg_loss_pct:>12.1f}%")
    print(f"  Profit Factor:       {result.profit_factor:>12.2f}")
    print(f"  Avg Trade Duration:  {result.avg_trade_duration_days:>12.0f} days")

    print("\n--- TIME ---")
    print(f"  Total Days:          {result.total_days:>12}")
    print(f"  Time in Market:      {result.time_in_market_pct:>12.1f}%")

    print("=" * 70 + "\n")
