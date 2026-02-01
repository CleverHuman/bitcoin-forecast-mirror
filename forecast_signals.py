"""BTC forecast with cycle-aware signals for buy/sell timing.

Uses halving cycle position + technical indicators to generate signals.
"""

import io
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

from src.db import DatabricksConnector
from src.metrics import (
    HALVING_DATES,
    backtest_predictions,
    compute_cycle_metrics,
    compute_halving_averages,
    print_halving_summary,
)
from src.models import (
    train_simple_ensemble,
    generate_signals,
    get_current_signal,
    add_cycle_features,
    backtest_signals,
    # Advanced backtesting
    BacktestConfig,
    run_backtest,
    print_backtest_report,
    optimize_parameters,
)

load_dotenv()

# Forecast horizon in days (set FORECAST_DAYS in .env; default 365)
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "365"))

# Reports directory
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def fetch_btc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch BTC trade data from Databricks."""
    connector = DatabricksConnector()

    sql = f"""
        SELECT date, avg_price
        FROM default.bitmex_trade_daily_stats
        WHERE symbol LIKE '%XBTUSD%'
          AND side = 'Sell'
          AND to_date(date) > '{start_date}'
          AND to_date(date) < '{end_date}'
        ORDER BY to_date(date) DESC
    """

    df = connector.query(sql)
    df = df.rename(columns={"date": "ds", "avg_price": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def plot_signals(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    cycle_metrics: pd.DataFrame | None = None,
) -> None:
    """Plot price with buy/sell signals and cycle phases.

    Args:
        df: Historical price data with signals.
        forecast: Forecasted prices.
        cycle_metrics: Optional cycle metrics for plotting tops/bottoms.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Sort data by date
    df = df.sort_values("ds").reset_index(drop=True)
    forecast = forecast.sort_values("ds").reset_index(drop=True)

    # Top: Price with forecast
    ax1 = axes[0]
    ax1.plot(df["ds"], df["y"], "k-", label="Actual Price", alpha=0.8, linewidth=1)

    if "yhat_ensemble" in forecast.columns:
        ax1.plot(forecast["ds"], forecast["yhat_ensemble"], "b--", label="Ensemble Forecast", alpha=0.7)
        ax1.fill_between(
            forecast["ds"],
            forecast["yhat_ensemble_lower"],
            forecast["yhat_ensemble_upper"],
            alpha=0.2,
        )
    else:
        ax1.plot(forecast["ds"], forecast["yhat"], "b--", label="Prophet Forecast", alpha=0.7)

    # Mark halving dates
    for i, h in enumerate(HALVING_DATES):
        if df["ds"].min() <= h <= forecast["ds"].max():
            ax1.axvline(x=h, color="red", linestyle="--", alpha=0.7, linewidth=2)
            ax1.text(h, ax1.get_ylim()[1] * 0.9, f"Halving {i+1}", rotation=90, va="top", fontsize=8)

    # Plot actual and predicted tops/bottoms from cycle_metrics
    if cycle_metrics is not None and not cycle_metrics.empty:
        # Actual tops (filled red triangle pointing down)
        actual_tops = cycle_metrics[["post_high_date", "post_high_price"]].dropna()
        ax1.scatter(
            actual_tops["post_high_date"],
            actual_tops["post_high_price"],
            marker="v",
            s=120,
            c="red",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=10,
            label="Actual Top",
        )

        # Actual bottoms (filled green triangle pointing up)
        actual_bottoms = cycle_metrics[["post_low_date", "post_low_price"]].dropna()
        ax1.scatter(
            actual_bottoms["post_low_date"],
            actual_bottoms["post_low_price"],
            marker="^",
            s=120,
            c="limegreen",
            edgecolors="darkgreen",
            linewidths=1.5,
            zorder=10,
            label="Actual Bottom",
        )

        # Get predicted tops/bottoms from backtesting (using prior cycles)
        backtest = backtest_predictions(cycle_metrics)
        if not backtest.empty:
            # Predicted tops (hollow red diamond)
            ax1.scatter(
                backtest["predicted_top"],
                backtest["actual_top_price"],  # Use actual price at predicted date for y-axis
                marker="D",
                s=80,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=9,
                label="Predicted Top",
            )

            # Predicted bottoms (hollow green diamond)
            ax1.scatter(
                backtest["predicted_bottom"],
                backtest["actual_bottom_price"],  # Use actual price at predicted date for y-axis
                marker="D",
                s=80,
                facecolors="none",
                edgecolors="limegreen",
                linewidths=2,
                zorder=9,
                label="Predicted Bottom",
            )

    last_historical = df["ds"].max()
    ax1.axvline(x=last_historical, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("BTC Price with Cycle-Aware Forecast")
    ax1.legend(loc="upper left")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Middle: Signal score
    ax2 = axes[1]
    colors = ["green" if x > 0 else "red" for x in df["signal_score"]]
    ax2.bar(df["ds"], df["signal_score"], color=colors, alpha=0.6, width=2)
    ax2.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=0.2, color="lightgreen", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    ax2.axhline(y=-0.2, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=-0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(x=last_historical, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Signal Score")
    ax2.set_ylim(-1, 1)
    ax2.set_yticks([-1, -0.5, -0.2, 0, 0.2, 0.5, 1])
    ax2.set_yticklabels(["Strong Sell", "Sell", "", "Hold", "", "Buy", "Strong Buy"], fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Bottom: Cycle phase
    ax3 = axes[2]
    phase_colors = {
        "accumulation": "green",
        "pre_halving_runup": "lightgreen",
        "post_halving_consolidation": "yellow",
        "bull_run": "orange",
        "distribution": "red",
        "drawdown": "purple",
    }
    # Historical phases (solid fill)
    for phase, color in phase_colors.items():
        mask = df["cycle_phase"] == phase
        if mask.any():
            ax3.fill_between(df["ds"], 0, 1, where=mask, color=color, alpha=0.6, label=phase.replace("_", " ").title())

    # Predicted phases (dotted shading + dotted boundary line)
    forecast_with_phase = add_cycle_features(forecast[["ds"]].copy(), "ds")
    pred = forecast_with_phase[forecast_with_phase["ds"] > last_historical]
    if not pred.empty:
        ax3.axvline(x=last_historical, color="gray", linestyle=":", linewidth=2, alpha=0.9)
        first_pred_label = True
        for phase, color in phase_colors.items():
            mask = pred["cycle_phase"] == phase
            if mask.any():
                lbl = "Predicted phases" if first_pred_label else None
                first_pred_label = False
                ax3.fill_between(
                    pred["ds"], 0, 1, where=mask, color=color, alpha=0.4, hatch="..", label=lbl
                )

    ax3.set_ylabel("Cycle Phase")
    ax3.set_yticks([])
    ax3.set_ylim(0, 1)
    ax3.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.15))

    # Format x-axis dates
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    # Rotate and align the tick labels
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")

    ax3.set_xlabel("Date")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    # Second figure: price only, linear scale (proper USD prices)
    fig2, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.plot(df["ds"], df["y"], "k-", label="Actual Price", alpha=0.8, linewidth=1)
    if "yhat_ensemble" in forecast.columns:
        ax.plot(forecast["ds"], forecast["yhat_ensemble"], "b--", label="Ensemble Forecast", alpha=0.7)
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_ensemble_lower"],
            forecast["yhat_ensemble_upper"],
            alpha=0.2,
        )
    else:
        ax.plot(forecast["ds"], forecast["yhat"], "b--", label="Prophet Forecast", alpha=0.7)
    for i, h in enumerate(HALVING_DATES):
        if df["ds"].min() <= h <= forecast["ds"].max():
            ax.axvline(x=h, color="red", linestyle="--", alpha=0.7, linewidth=2)
            ax.text(h, ax.get_ylim()[1] * 0.9, f"Halving {i+1}", rotation=90, va="top", fontsize=8)

    # Plot actual and predicted tops/bottoms on linear scale too
    if cycle_metrics is not None and not cycle_metrics.empty:
        # Actual tops
        actual_tops = cycle_metrics[["post_high_date", "post_high_price"]].dropna()
        ax.scatter(
            actual_tops["post_high_date"],
            actual_tops["post_high_price"],
            marker="v",
            s=120,
            c="red",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=10,
            label="Actual Top",
        )

        # Actual bottoms
        actual_bottoms = cycle_metrics[["post_low_date", "post_low_price"]].dropna()
        ax.scatter(
            actual_bottoms["post_low_date"],
            actual_bottoms["post_low_price"],
            marker="^",
            s=120,
            c="limegreen",
            edgecolors="darkgreen",
            linewidths=1.5,
            zorder=10,
            label="Actual Bottom",
        )

        # Predicted tops/bottoms
        backtest = backtest_predictions(cycle_metrics)
        if not backtest.empty:
            ax.scatter(
                backtest["predicted_top"],
                backtest["actual_top_price"],
                marker="D",
                s=80,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=9,
                label="Predicted Top",
            )
            ax.scatter(
                backtest["predicted_bottom"],
                backtest["actual_bottom_price"],
                marker="D",
                s=80,
                facecolors="none",
                edgecolors="limegreen",
                linewidths=2,
                zorder=9,
                label="Predicted Bottom",
            )

    ax.axvline(x=last_historical, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    ax.set_title("BTC Price with Forecast (linear scale)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig2.tight_layout()

    # Close both figures when either window is closed
    def _close_all(_event):
        plt.close("all")

    fig.canvas.mpl_connect("close_event", _close_all)
    fig2.canvas.mpl_connect("close_event", _close_all)

    plt.show()


def print_signal_summary(current: dict, backtest: dict) -> None:
    """Print summary of current signal and backtest results."""
    print("\n" + "=" * 60)
    print("CURRENT SIGNAL")
    print("=" * 60)
    print(f"  Date: {current.get('date')}")
    print(f"  Price: ${current.get('price'):,.0f}" if current.get('price') else "  Price: N/A")
    print(f"  Signal: {current.get('signal', 'N/A').upper()}")
    print(f"  Score: {current.get('signal_score', 0):.2f}")
    print(f"  Cycle Phase: {current.get('cycle_phase', 'N/A')}")

    days_until = current.get("days_until_halving")
    days_since = current.get("days_since_halving")
    if days_until:
        print(f"  Days Until Halving: {int(days_until)}")
    if days_since:
        print(f"  Days Since Halving: {int(days_since)}")

    if current.get("rsi"):
        print(f"  RSI: {current['rsi']:.1f}")

    # Buy timing guidance
    buy_timing = current.get("buy_timing")
    if buy_timing:
        print("\n" + "-" * 60)
        print("BUY TIMING GUIDANCE")
        print("-" * 60)
        print(f"  >>> {buy_timing['action']} <<<")
        print(f"  {buy_timing['reason']}")
        print()
        print(f"  Last cycle top: ${buy_timing['last_top_price']:,.0f} on {buy_timing['last_top_date'].strftime('%Y-%m-%d')}")
        print()
        print("  Predicted bottom (decay-adjusted):")
        low, high = buy_timing['decay_bottom_range']
        print(f"    Price range:  ${low:,.0f} - ${high:,.0f}")
        print(f"    Best estimate: ${buy_timing['decay_bottom_price']:,.0f}")
        print(f"    Expected drawdown: {buy_timing['decay_drawdown_pct']:.1f}%")
        print()
        print(f"  Current price vs predicted bottom: {buy_timing['price_vs_decay_bottom_pct']:+.1f}%")
        if buy_timing['days_to_predicted_bottom'] > 0:
            print(f"  Days to predicted bottom: ~{buy_timing['days_to_predicted_bottom']}")

    print(f"\n  Recommendation: {current.get('recommendation', 'N/A')}")

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Signal Strategy vs Buy & Hold)")
    print("=" * 60)
    print(f"  Initial Capital: ${backtest['initial_capital']:,.0f}")
    print(f"  Signal Strategy Final: ${backtest['final_value']:,.0f} ({backtest['total_return_pct']:+.1f}%)")
    print(f"  Buy & Hold Final: ${backtest['buy_hold_value']:,.0f} ({backtest['buy_hold_return_pct']:+.1f}%)")
    print(f"  Outperformance: {backtest['outperformance_pct']:+.1f}%")
    print(f"  Number of Trades: {backtest['num_trades']}")
    print("=" * 60 + "\n")


class TeeOutput:
    """Capture stdout while still printing to console."""

    def __init__(self):
        self.buffer = io.StringIO()
        self.stdout = sys.stdout

    def write(self, text):
        self.buffer.write(text)
        self.stdout.write(text)

    def flush(self):
        self.stdout.flush()

    def getvalue(self):
        return self.buffer.getvalue()


def main():
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Capture output for report
    tee = TeeOutput()
    sys.stdout = tee

    try:
        # Use longer history for better cycle analysis
        start_date = "2015-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Run timestamp: {timestamp}")
        print(f"Fetching BTC data from {start_date} to {end_date}...")
        df = fetch_btc_data(start_date, end_date)
        print(f"Loaded {len(df)} rows.")

        # Compute halving metrics for context
        print("\nComputing halving cycle metrics...")
        cycle_metrics = compute_cycle_metrics(df)
        averages = compute_halving_averages(cycle_metrics=cycle_metrics)

        # Print predicted vs actual comparison
        print_halving_summary(cycle_metrics, averages)

        # Train ensemble model with decay regressor
        print("\nTraining cycle-aware ensemble model with decay adjustment...")
        forecast = train_simple_ensemble(
            df,
            periods=FORECAST_DAYS,
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
        )

        # Generate signals
        print("Generating buy/sell signals...")
        df_signals = generate_signals(df, cycle_weight=0.4, technical_weight=0.6)

        # Get current signal with buy timing
        current = get_current_signal(df_signals, cycle_metrics=cycle_metrics, averages=averages)

        # Backtest
        print("Running backtest...")
        backtest = backtest_signals(df_signals, initial_capital=10000)

        # Print summary
        print_signal_summary(current, backtest)

        # Save outputs to reports folder with timestamp
        print(f"\nSaving results to {REPORTS_DIR}/...")

        signals_path = REPORTS_DIR / f"signals_{timestamp}.csv"
        forecast_path = REPORTS_DIR / f"forecast_{timestamp}.csv"
        report_path = REPORTS_DIR / f"report_{timestamp}.txt"

        df_signals.to_csv(signals_path, index=False)
        forecast.to_csv(forecast_path, index=False)

        print(f"  Signals:  {signals_path.name}")
        print(f"  Forecast: {forecast_path.name}")
        print(f"  Report:   {report_path.name}")

        # Plot
        print("\nPlotting...")
        plot_signals(df_signals, forecast, cycle_metrics=cycle_metrics)

        print("\nDone!")

    finally:
        # Restore stdout and save report
        sys.stdout = tee.stdout
        report_content = tee.getvalue()

        # Save the report
        report_path = REPORTS_DIR / f"report_{timestamp}.txt"
        with open(report_path, "w") as f:
            f.write(report_content)


if __name__ == "__main__":
    main()
