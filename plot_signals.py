#!/usr/bin/env python3
"""Plot price, forecast, and trading signals on a chart.

Generates a visualization showing:
1. Historical price and forecast
2. Buy/sell signals from the Combined Strategy
3. Predicted top/bottom price levels
4. Cycle phases (shaded regions)
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.db import fetch_btc_data
from src.metrics import compute_cycle_metrics, compute_halving_averages, HALVING_DATES
from src.forecasting import ProphetCycleForecaster
from src.backtesting.strategies import CombinedStrategy, Signal


def main():
    print("=" * 70)
    print("SIGNAL VISUALIZATION")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading BTC data...")
    start_date = "2015-01-01"  # Start from 2015 for cleaner chart
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = fetch_btc_data(start_date=start_date, end_date=end_date)
    print(f"  Loaded {len(df)} days")

    # 2. Compute cycle metrics
    print("\n[2/4] Computing cycle metrics...")
    cycle_metrics = compute_cycle_metrics(df)
    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print(f"  {averages.n_cycles} cycles analyzed")

    # 3. Generate forecast
    print("\n[3/4] Generating forecast...")
    forecaster = ProphetCycleForecaster(
        halving_averages=averages,
        cycle_metrics=cycle_metrics,
    )
    result = forecaster.fit_predict(df, periods=365)
    forecast = result.forecast
    print(f"  Forecast through {forecast['ds'].max().date()}")

    # 4. Generate signals
    print("\n[4/4] Generating signals...")
    strategy = CombinedStrategy(
        halving_averages=averages,
        cycle_metrics=cycle_metrics,
    )
    df_signals = strategy.generate_signals(df, forecast=forecast)
    print(f"  Signals generated")

    # Get current signal
    current = strategy.get_current_signal(df_signals)
    print(f"\n  Current Signal: {current.signal.value.upper()}")
    print(f"  Reason: {current.reason}")

    # Plot
    print("\n[5/5] Creating chart...")
    fig = create_signal_chart(df_signals, forecast, cycle_metrics, averages)

    # Save
    output_path = Path("signals_chart.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved to: {output_path.absolute()}")

    plt.show()


def create_signal_chart(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    cycle_metrics: pd.DataFrame,
    averages,
) -> plt.Figure:
    """Create the signal visualization chart."""

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
    fig.suptitle("BTC Price, Forecast & Trading Signals", fontsize=14, fontweight="bold")

    # =================================================================
    # Panel 1: Price + Forecast + Signals
    # =================================================================
    ax1 = axes[0]

    # Plot price
    ax1.plot(df["ds"], df["y"], color="black", linewidth=1.5, label="Price", zorder=3)

    # Plot forecast
    forecast_future = forecast[forecast["ds"] > df["ds"].max()]
    ax1.plot(forecast["ds"], forecast["yhat_ensemble"],
             color="blue", linewidth=1, alpha=0.7, label="Forecast")
    ax1.fill_between(forecast["ds"],
                     forecast["yhat_ensemble_lower"],
                     forecast["yhat_ensemble_upper"],
                     color="blue", alpha=0.1, label="95% CI")

    # Plot predicted bottom/top prices if available
    if "predicted_bottom_price" in df.columns:
        bottom_price = df["predicted_bottom_price"].iloc[-1]
        top_price = df["predicted_top_price"].iloc[-1]
        if pd.notna(bottom_price):
            ax1.axhline(bottom_price, color="green", linestyle="--", alpha=0.5,
                       label=f"Predicted Bottom: ${bottom_price:,.0f}")
        if pd.notna(top_price):
            ax1.axhline(top_price, color="red", linestyle="--", alpha=0.5,
                       label=f"Predicted Top: ${top_price:,.0f}")

    # Plot buy signals
    buy_signals = df[df["signal"].isin([Signal.STRONG_BUY.value, Signal.BUY.value])]
    ax1.scatter(buy_signals["ds"], buy_signals["y"],
                color="green", marker="^", s=50, zorder=5, label="Buy Signal", alpha=0.7)

    # Plot sell signals
    sell_signals = df[df["signal"].isin([Signal.STRONG_SELL.value, Signal.SELL.value])]
    ax1.scatter(sell_signals["ds"], sell_signals["y"],
                color="red", marker="v", s=50, zorder=5, label="Sell Signal", alpha=0.7)

    # Shade halving events
    for halving_date in HALVING_DATES:
        if df["ds"].min() <= halving_date <= forecast["ds"].max():
            ax1.axvline(halving_date, color="orange", linestyle="-", alpha=0.5, linewidth=2)
            ax1.annotate("Halving", xy=(halving_date, ax1.get_ylim()[1]),
                        fontsize=8, color="orange", ha="center", va="bottom")

    # =================================================================
    # Plot ACTUAL and PREDICTED tops/bottoms for ALL cycles
    # =================================================================
    from datetime import timedelta

    for idx, row in cycle_metrics.iterrows():
        halving_date = row["halving_date"]

        # Calculate predicted dates for this cycle using averages
        predicted_top = halving_date + timedelta(days=int(averages.avg_days_to_top))
        predicted_bottom = halving_date + timedelta(days=int(averages.avg_days_to_bottom))

        # Get actual dates
        actual_top = row.get("post_high_date")
        actual_bottom = row.get("post_low_date")

        # Determine if this is a future/current cycle (no actual data yet)
        today = pd.Timestamp.now()
        is_current_cycle = halving_date == HALVING_DATES[-1]  # Most recent halving

        # Plot PREDICTED top (dashed line)
        if df["ds"].min() <= predicted_top <= forecast["ds"].max():
            ax1.axvline(predicted_top, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
            label = f"PRED TOP\n{predicted_top.strftime('%Y-%m')}"
            if is_current_cycle:
                label = f"PRED TOP\n{predicted_top.strftime('%Y-%m-%d')}\n(+{int(averages.avg_days_to_top)}d)"
            ax1.annotate(label,
                        xy=(predicted_top, ax1.get_ylim()[1] * 0.75),
                        fontsize=7, color="red", ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8, edgecolor="red"))

        # Plot ACTUAL top (solid line) if available
        if pd.notna(actual_top) and df["ds"].min() <= actual_top <= forecast["ds"].max():
            ax1.axvline(actual_top, color="darkred", linestyle="-", alpha=0.8, linewidth=2)
            # Calculate error vs prediction
            error_days = (actual_top - predicted_top).days
            error_str = f"+{error_days}d" if error_days > 0 else f"{error_days}d"
            ax1.annotate(f"ACTUAL TOP\n{actual_top.strftime('%Y-%m')}\n({error_str} vs pred)",
                        xy=(actual_top, ax1.get_ylim()[1] * 0.92),
                        fontsize=7, color="darkred", ha="center", va="top", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

        # Plot PREDICTED bottom (dashed line)
        if df["ds"].min() <= predicted_bottom <= forecast["ds"].max():
            ax1.axvline(predicted_bottom, color="green", linestyle="--", alpha=0.6, linewidth=1.5)
            label = f"PRED BTM\n{predicted_bottom.strftime('%Y-%m')}"
            if is_current_cycle:
                label = f"PRED BTM\n{predicted_bottom.strftime('%Y-%m-%d')}\n(+{int(averages.avg_days_to_bottom)}d)"
            ax1.annotate(label,
                        xy=(predicted_bottom, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.20),
                        fontsize=7, color="green", ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8, edgecolor="green"))

        # Plot ACTUAL bottom (solid line) if available
        if pd.notna(actual_bottom) and df["ds"].min() <= actual_bottom <= forecast["ds"].max():
            ax1.axvline(actual_bottom, color="darkgreen", linestyle="-", alpha=0.8, linewidth=2)
            # Calculate error vs prediction
            error_days = (actual_bottom - predicted_bottom).days
            error_str = f"+{error_days}d" if error_days > 0 else f"{error_days}d"
            ax1.annotate(f"ACTUAL BTM\n{actual_bottom.strftime('%Y-%m')}\n({error_str} vs pred)",
                        xy=(actual_bottom, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05),
                        fontsize=7, color="darkgreen", ha="center", va="bottom", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

    # Shade buy/sell zones
    buy_zone = df[df["buy_zone"] == True]
    if not buy_zone.empty:
        for _, group in buy_zone.groupby((~buy_zone["buy_zone"].shift().fillna(False)).cumsum()):
            ax1.axvspan(group["ds"].min(), group["ds"].max(),
                       color="green", alpha=0.05, zorder=1)

    sell_zone = df[df["sell_zone"] == True]
    if not sell_zone.empty:
        for _, group in sell_zone.groupby((~sell_zone["sell_zone"].shift().fillna(False)).cumsum()):
            ax1.axvspan(group["ds"].min(), group["ds"].max(),
                       color="red", alpha=0.05, zorder=1)

    ax1.set_ylabel("Price (USD)", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # =================================================================
    # Panel 2: Signal Score Components
    # =================================================================
    ax2 = axes[1]

    ax2.fill_between(df["ds"], 0, df["cycle_score"],
                     where=df["cycle_score"] > 0, color="blue", alpha=0.3, label="Cycle (bullish)")
    ax2.fill_between(df["ds"], 0, df["cycle_score"],
                     where=df["cycle_score"] < 0, color="blue", alpha=0.3)

    ax2.fill_between(df["ds"], 0, df["forecast_score"],
                     where=df["forecast_score"] > 0, color="green", alpha=0.3, label="Forecast (bullish)")
    ax2.fill_between(df["ds"], 0, df["forecast_score"],
                     where=df["forecast_score"] < 0, color="green", alpha=0.3)

    ax2.plot(df["ds"], df["signal_score"], color="black", linewidth=1, label="Combined Score")

    ax2.axhline(0.2, color="green", linestyle=":", alpha=0.5)
    ax2.axhline(-0.2, color="red", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)

    ax2.set_ylabel("Signal Score", fontsize=10)
    ax2.set_ylim(-1.2, 1.2)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # =================================================================
    # Panel 3: RSI + Technical Score
    # =================================================================
    ax3 = axes[2]

    if "rsi" in df.columns:
        ax3.plot(df["ds"], df["rsi"], color="purple", linewidth=1, label="RSI(14)")
        ax3.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)")
        ax3.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)")
        ax3.axhline(50, color="gray", linestyle=":", alpha=0.3)
        ax3.set_ylim(0, 100)

    ax3.set_ylabel("RSI", fontsize=10)
    ax3.set_xlabel("Date", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
