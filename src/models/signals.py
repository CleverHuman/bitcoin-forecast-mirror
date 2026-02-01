"""Buy/sell signal generation based on halving cycles.

Strategy: Buy during accumulation (before halving), sell during bull run (after halving).
When halving_averages is provided, uses data-driven timing from historical cycles
to determine optimal buy/sell windows.
"""

from enum import Enum
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from .cycle_features import add_cycle_features, CyclePhase

if TYPE_CHECKING:
    from src.metrics import HalvingAverages


class SignalType(Enum):
    """Trading signal types."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


# Cycle-based signal strategy:
# - BUY: Accumulation (1-2 years before halving) and late drawdown (bear market)
# - HOLD: Pre-halving run-up and early post-halving
# - SELL: Bull run peak (6-12 months post halving) and distribution

PHASE_BIAS = {
    # BUY ZONES (positive = bullish)
    CyclePhase.ACCUMULATION.value: 0.7,      # Strong buy - ideal entry window
    CyclePhase.DRAWDOWN.value: 0.5,          # Buy - bear market accumulation

    # HOLD ZONES (neutral)
    CyclePhase.PRE_HALVING_RUNUP.value: 0.1, # Hold - already running up, late to buy
    CyclePhase.POST_HALVING_CONSOLIDATION.value: 0.0,  # Hold - wait for direction

    # SELL ZONES (negative = bearish)
    CyclePhase.BULL_RUN.value: -0.4,         # Sell - take profits during euphoria
    CyclePhase.DISTRIBUTION.value: -0.7,     # Strong sell - cycle top territory
}


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_ma_crossover(prices: pd.Series, short: int = 50, long: int = 200) -> pd.Series:
    """Compute moving average crossover signal (-1 to 1)."""
    ma_short = prices.rolling(window=short).mean()
    ma_long = prices.rolling(window=long).mean()
    crossover = (ma_short - ma_long) / ma_long
    return crossover.clip(-0.5, 0.5) * 2


def generate_signals(
    df: pd.DataFrame,
    price_col: str = "y",
    date_col: str = "ds",
    include_technicals: bool = True,
    cycle_weight: float = 0.6,
    technical_weight: float = 0.4,
    halving_averages: "HalvingAverages | None" = None,
) -> pd.DataFrame:
    """Generate buy/sell signals based on halving cycle position.

    Strategy:
    - BUY during accumulation (before halving) - ideal entry at cycle bottom
    - BUY during late drawdown (bear market bottoms)
    - HOLD during pre-halving run-up (ride the wave)
    - SELL during bull run (post halving peak) - data-driven timing
    - STRONG SELL during distribution (cycle top)

    Args:
        df: DataFrame with date and price columns.
        price_col: Name of price column.
        date_col: Name of date column.
        include_technicals: Whether to include technical indicators.
        cycle_weight: Weight for cycle-based signal (0-1). Default 0.6.
        technical_weight: Weight for technical signal (0-1). Default 0.4.
        halving_averages: HalvingAverages with historical timing data.
            If provided, uses avg days to peak/bottom for buy/sell windows.

    Returns:
        DataFrame with signal columns added.
    """
    df = df.copy()
    df = add_cycle_features(df, date_col=date_col)

    # Base cycle score from phase
    df["cycle_score"] = df["cycle_phase"].map(PHASE_BIAS)

    # Refine based on specific timing within cycle
    days_until = df["days_until_halving"].fillna(9999)
    days_since = df["days_since_halving"].fillna(0)

    # Use data-driven timing if halving_averages provided, else use defaults
    if halving_averages and halving_averages.n_cycles > 0:
        # Data-driven windows based on historical averages
        # BUY: Around the historical average cycle bottom (before halving)
        # Default: ~400-800 days before halving is typical accumulation
        avg_days_to_bottom = getattr(halving_averages, 'avg_days_before_halving_to_low', 500)
        buy_window_start = avg_days_to_bottom + 150  # Start accumulating early
        buy_window_end = avg_days_to_bottom - 100    # Stop before bottom passes

        # SELL: Around the historical average cycle peak (after halving)
        avg_days_to_peak = getattr(halving_averages, 'avg_days_after_halving_to_high', 300)
        sell_window_start = avg_days_to_peak - 60   # Start selling before peak
        sell_window_end = avg_days_to_peak + 120    # Continue selling after peak
    else:
        # Default windows (fallback)
        buy_window_start = 600
        buy_window_end = 300
        sell_window_start = 200
        sell_window_end = 400

    # IDEAL BUY WINDOW: Around historical cycle bottom
    ideal_buy_mask = (days_until >= buy_window_end) & (days_until <= buy_window_start)
    df.loc[ideal_buy_mask, "cycle_score"] += 0.3

    # Store the windows for reference
    df["buy_window"] = ideal_buy_mask

    # LATE TO BUY: Less than 90 days before halving (run-up already started)
    late_buy_mask = (days_until > 0) & (days_until < 90)
    df.loc[late_buy_mask, "cycle_score"] -= 0.2

    # IDEAL SELL WINDOW: Around historical cycle peak (after halving)
    ideal_sell_mask = (days_since >= sell_window_start) & (days_since <= sell_window_end)
    df.loc[ideal_sell_mask, "cycle_score"] -= 0.4

    # Store the windows for reference
    df["sell_window"] = ideal_sell_mask

    # MUST SELL: After the typical peak window (distribution)
    must_sell_mask = days_since > sell_window_end
    df.loc[must_sell_mask, "cycle_score"] -= 0.2

    if include_technicals and price_col in df.columns:
        prices = df[price_col]

        # RSI
        rsi = compute_rsi(prices)
        df["rsi"] = rsi
        df["rsi_score"] = (50 - rsi) / 50

        # MACD
        macd_line, signal_line, histogram = compute_macd(prices)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram
        hist_std = histogram.rolling(window=50).std().replace(0, 1)
        df["macd_score"] = (histogram / hist_std).clip(-2, 2) / 2

        # MA Crossover
        df["ma_crossover"] = compute_ma_crossover(prices)
        df["ma_score"] = df["ma_crossover"]

        # Combined technical score
        df["technical_score"] = (
            df["rsi_score"].fillna(0) * 0.3
            + df["macd_score"].fillna(0) * 0.4
            + df["ma_score"].fillna(0) * 0.3
        )
    else:
        df["technical_score"] = 0

    # Normalize weights
    total_weight = cycle_weight + technical_weight
    cycle_weight = cycle_weight / total_weight
    technical_weight = technical_weight / total_weight

    # Combined signal score
    df["signal_score"] = (
        df["cycle_score"] * cycle_weight + df["technical_score"] * technical_weight
    ).clip(-1, 1)

    # Classify into signal types
    df["signal"] = df["signal_score"].apply(_classify_signal)

    # Add action recommendation
    df["action"] = df.apply(_get_action, axis=1)

    return df


def _classify_signal(score: float) -> str:
    """Convert numeric score to signal type."""
    if score >= 0.5:
        return SignalType.STRONG_BUY.value
    elif score >= 0.2:
        return SignalType.BUY.value
    elif score >= -0.2:
        return SignalType.HOLD.value
    elif score >= -0.5:
        return SignalType.SELL.value
    else:
        return SignalType.STRONG_SELL.value


def _get_action(row: pd.Series) -> str:
    """Get specific action based on cycle position."""
    phase = row.get("cycle_phase", "")
    days_until = row.get("days_until_halving")
    days_since = row.get("days_since_halving")
    signal = row.get("signal", "")

    # Buy zone actions
    if phase == CyclePhase.ACCUMULATION.value:
        if days_until and 300 <= days_until <= 600:
            return "ACCUMULATE - Ideal buy window before halving"
        return "ACCUMULATE - Good entry zone"

    if phase == CyclePhase.DRAWDOWN.value:
        return "ACCUMULATE - Bear market, DCA opportunity"

    # Hold zone actions
    if phase == CyclePhase.PRE_HALVING_RUNUP.value:
        if days_until and days_until < 90:
            return "HOLD - Run-up in progress, late to enter"
        return "HOLD or SMALL BUY - Approaching halving"

    if phase == CyclePhase.POST_HALVING_CONSOLIDATION.value:
        return "HOLD - Post-halving consolidation, wait"

    # Sell zone actions
    if phase == CyclePhase.BULL_RUN.value:
        if days_since and days_since >= 300:
            return "SELL - Bull run maturing, take profits"
        return "HOLD/TRIM - Bull run, consider partial sells"

    if phase == CyclePhase.DISTRIBUTION.value:
        return "SELL - Distribution phase, reduce exposure"

    return "HOLD"


def get_current_signal(
    df: pd.DataFrame,
    cycle_metrics: pd.DataFrame | None = None,
    averages: "HalvingAverages | None" = None,
) -> dict[str, Any]:
    """Get the most recent signal with context and buy timing guidance.

    Args:
        df: DataFrame with signals generated.
        cycle_metrics: Optional cycle metrics for bottom prediction.
        averages: Optional halving averages for timing data.

    Returns:
        Dict with current signal info and buy timing.
    """
    if df.empty:
        return {"error": "No data"}

    # Get the row with the most recent date
    latest = df.loc[df["ds"].idxmax()]
    current_price = latest.get("y")
    current_date = latest.get("ds")

    result = {
        "date": current_date,
        "price": current_price,
        "signal": latest.get("signal"),
        "signal_score": latest.get("signal_score"),
        "cycle_phase": latest.get("cycle_phase"),
        "days_until_halving": latest.get("days_until_halving"),
        "days_since_halving": latest.get("days_since_halving"),
        "rsi": latest.get("rsi"),
        "action": latest.get("action"),
        "recommendation": _get_recommendation(latest),
    }

    # Add buy timing guidance if we have cycle data
    if cycle_metrics is not None and not cycle_metrics.empty and current_price:
        result["buy_timing"] = _get_buy_timing(
            current_price, current_date, cycle_metrics, averages
        )

    return result


def _get_buy_timing(
    current_price: float,
    current_date: pd.Timestamp,
    cycle_metrics: pd.DataFrame,
    averages: "HalvingAverages | None" = None,
) -> dict[str, Any]:
    """Compute buy timing guidance based on predicted bottom."""
    from src.metrics import predict_drawdown, HALVING_DATES

    # Get the last cycle's top
    last_cycle = cycle_metrics.iloc[-1]
    last_top_price = last_cycle["post_high_price"]
    last_top_date = last_cycle["post_high_date"]

    # Simple average prediction
    avg_drawdown = averages.drawdown_pct if averages else 65.0
    avg_drawdown_days = averages.drawdown_days if averages else 300
    simple_bottom_price = last_top_price * (1 - avg_drawdown / 100)
    simple_bottom_date = last_top_date + pd.Timedelta(days=avg_drawdown_days)

    # Decay-adjusted prediction
    decay_pred = predict_drawdown(cycle_metrics, target_cycle=len(cycle_metrics) + 1)
    decay_bottom_price = last_top_price * (1 - decay_pred.predicted_value)

    # Use ±15% around best estimate for actionable buy zone (not the wide confidence interval)
    buy_zone_pct = 0.15
    decay_bottom_low = decay_bottom_price * (1 - buy_zone_pct)
    decay_bottom_high = decay_bottom_price * (1 + buy_zone_pct)

    # How far are we from the predicted bottom?
    price_vs_decay_bottom = (current_price / decay_bottom_price - 1) * 100

    # Days until predicted bottom
    days_to_bottom = (simple_bottom_date - current_date).days if simple_bottom_date > current_date else 0

    # Determine buy recommendation
    if current_price <= decay_bottom_price:
        buy_action = "BUY NOW"
        buy_reason = f"Price is at or below predicted bottom (${decay_bottom_price:,.0f})"
    elif current_price <= decay_bottom_high:
        buy_action = "ACCUMULATE"
        buy_reason = f"Price is within buy zone (${decay_bottom_low:,.0f} - ${decay_bottom_high:,.0f})"
    elif price_vs_decay_bottom < 10:
        buy_action = "START ACCUMULATING"
        buy_reason = f"Only {price_vs_decay_bottom:.1f}% above predicted bottom"
    elif days_to_bottom > 30:
        buy_action = "WAIT"
        buy_reason = f"~{days_to_bottom} days to predicted bottom, {price_vs_decay_bottom:.1f}% above target"
    else:
        buy_action = "DCA"
        buy_reason = f"Near bottom window, dollar-cost average in"

    return {
        "action": buy_action,
        "reason": buy_reason,
        "last_top_price": last_top_price,
        "last_top_date": last_top_date,
        # Simple average prediction
        "simple_bottom_price": simple_bottom_price,
        "simple_bottom_date": simple_bottom_date,
        # Decay-adjusted prediction
        "decay_bottom_price": decay_bottom_price,
        "decay_bottom_range": (decay_bottom_low, decay_bottom_high),
        "decay_drawdown_pct": decay_pred.predicted_value * 100,
        # Current position
        "price_vs_decay_bottom_pct": price_vs_decay_bottom,
        "days_to_predicted_bottom": days_to_bottom,
    }


def _get_recommendation(row: pd.Series) -> str:
    """Generate human-readable recommendation."""
    phase = row.get("cycle_phase", "unknown")
    days_until = row.get("days_until_halving")
    days_since = row.get("days_since_halving")
    action = row.get("action", "")

    if action:
        return action

    # Fallback recommendations
    if days_until and days_until > 0:
        if days_until > 300:
            return f"ACCUMULATION ZONE: {int(days_until)} days to halving. Ideal time to buy."
        elif days_until > 90:
            return f"PRE-HALVING: {int(days_until)} days to halving. Consider buying."
        else:
            return f"HALVING IMMINENT: {int(days_until)} days. Hold positions."

    if days_since and days_since > 0:
        if days_since < 120:
            return f"POST-HALVING: {int(days_since)} days since halving. Hold, wait for bull run."
        elif days_since < 365:
            return f"BULL RUN: {int(days_since)} days post-halving. Watch for top, consider selling."
        elif days_since < 545:
            return f"DISTRIBUTION: {int(days_since)} days post-halving. Sell into strength."
        else:
            return f"DRAWDOWN: {int(days_since)} days post-halving. Accumulate on dips."

    return f"Phase: {phase}"


def backtest_signals(
    df: pd.DataFrame,
    initial_capital: float = 10000,
    position_size: float = 0.25,
) -> dict[str, Any]:
    """Simple backtest of signal performance.

    Args:
        df: DataFrame with 'y' (price), 'signal', 'ds' columns.
        initial_capital: Starting capital in USD.
        position_size: Fraction of capital to use per trade.

    Returns:
        Dict with backtest results.
    """
    df = df.copy().sort_values("ds").reset_index(drop=True)

    capital = initial_capital
    btc_held = 0.0
    trades = []

    prev_signal = SignalType.HOLD.value

    for i, row in df.iterrows():
        signal = row["signal"]
        price = row["y"]
        date = row["ds"]

        if pd.isna(price) or price <= 0:
            continue

        # Buy signals
        if signal in [SignalType.STRONG_BUY.value, SignalType.BUY.value]:
            if prev_signal not in [SignalType.STRONG_BUY.value, SignalType.BUY.value]:
                buy_amount = capital * position_size
                if buy_amount > 100:
                    btc_bought = buy_amount / price
                    btc_held += btc_bought
                    capital -= buy_amount
                    trades.append({
                        "date": date,
                        "action": "BUY",
                        "price": price,
                        "amount_usd": buy_amount,
                        "btc": btc_bought,
                    })

        # Sell signals
        elif signal in [SignalType.STRONG_SELL.value, SignalType.SELL.value]:
            if prev_signal not in [SignalType.STRONG_SELL.value, SignalType.SELL.value]:
                if btc_held > 0:
                    sell_btc = btc_held * position_size
                    sell_amount = sell_btc * price
                    btc_held -= sell_btc
                    capital += sell_amount
                    trades.append({
                        "date": date,
                        "action": "SELL",
                        "price": price,
                        "amount_usd": sell_amount,
                        "btc": sell_btc,
                    })

        prev_signal = signal

    # Final portfolio value
    final_price = df.iloc[-1]["y"]
    final_value = capital + (btc_held * final_price)

    # Buy and hold comparison
    buy_hold_btc = initial_capital / df.iloc[0]["y"]
    buy_hold_value = buy_hold_btc * final_price

    return {
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": (final_value / initial_capital - 1) * 100,
        "buy_hold_value": buy_hold_value,
        "buy_hold_return_pct": (buy_hold_value / initial_capital - 1) * 100,
        "outperformance_pct": (final_value / buy_hold_value - 1) * 100,
        "num_trades": len(trades),
        "trades": trades,
        "final_btc_held": btc_held,
        "final_cash": capital,
    }
