"""BTC forecast with cycle-aware signals for buy/sell timing.

Uses halving cycle position + technical indicators to generate signals.

Usage:
    python forecast_signals.py                     # Forecast from today
    python forecast_signals.py --from-date 2024-01-01  # Backtest: train up to date, forecast from there
    python forecast_signals.py --from-date 2023-06-01 --days 365  # Custom forecast horizon
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.db import fetch_btc_data
from src.metrics import (
    compute_cycle_metrics,
    compute_halving_averages,
    print_halving_summary,
)
from src.models import (
    backtest_signals,
    generate_signals,
    get_current_signal,
    train_simple_ensemble,
)
from src.reporting import print_signal_summary
from src.utils import TeeOutput
from src.viz import plot_signals

load_dotenv()

# Forecast horizon in days (set FORECAST_DAYS in .env; default 365)
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "365"))

# Reports directory
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def compute_forecast_accuracy(
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    forecast_col: str = "yhat_ensemble",
) -> dict:
    """Compute accuracy metrics for forecast vs actual data.

    Args:
        forecast: Forecast DataFrame with 'ds' and forecast_col
        actual: Actual data DataFrame with 'ds' and 'y'
        forecast_col: Column name for forecast values

    Returns:
        Dict with accuracy metrics
    """
    forecast = forecast.copy()
    actual = actual.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    actual["ds"] = pd.to_datetime(actual["ds"])

    # Merge on date
    merged = actual.merge(
        forecast[["ds", forecast_col, "yhat_ensemble_lower", "yhat_ensemble_upper"]],
        on="ds",
        how="inner",
    )

    if merged.empty:
        return {"error": "No overlapping dates between forecast and actual"}

    y_true = merged["y"].values
    y_pred = merged[forecast_col].values

    # Metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Direction accuracy
    if len(merged) >= 2:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        direction_accuracy = None

    # Coverage: what % of actual values fall within confidence interval
    within_ci = (
        (merged["y"] >= merged["yhat_ensemble_lower"])
        & (merged["y"] <= merged["yhat_ensemble_upper"])
    )
    coverage = within_ci.mean() * 100

    # Final price comparison
    final_actual = merged["y"].iloc[-1]
    final_pred = merged[forecast_col].iloc[-1]
    final_error_pct = (final_pred - final_actual) / final_actual * 100

    return {
        "days_compared": len(merged),
        "mae": mae,
        "mape": mape,
        "rmse": rmse,
        "direction_accuracy": direction_accuracy,
        "ci_coverage": coverage,
        "final_actual": final_actual,
        "final_predicted": final_pred,
        "final_error_pct": final_error_pct,
    }


def print_accuracy_report(metrics: dict, from_date: str, to_date: str) -> None:
    """Print formatted accuracy report."""
    print("\n" + "=" * 60)
    print("FORECAST ACCURACY REPORT")
    print(f"Period: {from_date} to {to_date}")
    print("=" * 60)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"\nDays compared: {metrics['days_compared']}")
    print(f"\nError Metrics:")
    print(f"  MAE (Mean Absolute Error):     ${metrics['mae']:,.2f}")
    print(f"  MAPE (Mean Absolute % Error):  {metrics['mape']:.2f}%")
    print(f"  RMSE (Root Mean Square Error): ${metrics['rmse']:,.2f}")

    if metrics["direction_accuracy"] is not None:
        print(f"\nDirection Accuracy: {metrics['direction_accuracy']:.1f}%")
        print("  (% of days where predicted direction matched actual)")

    print(f"\nConfidence Interval Coverage: {metrics['ci_coverage']:.1f}%")
    print("  (% of actual values within 95% CI)")

    print(f"\nFinal Price Comparison:")
    print(f"  Actual:    ${metrics['final_actual']:,.2f}")
    print(f"  Predicted: ${metrics['final_predicted']:,.2f}")
    print(f"  Error:     {metrics['final_error_pct']:+.2f}%")
    print("=" * 60)


def plot_signals_with_actual(
    df_train: pd.DataFrame,
    df_actual: pd.DataFrame,
    forecast: pd.DataFrame,
    forecast_from_date: str,
    cycle_metrics: pd.DataFrame | None = None,
    halving_averages: "HalvingAverages | None" = None,
) -> None:
    """Plot forecast with actual data overlay for backtesting visualization.

    Args:
        df_train: Training data (before forecast_from_date)
        df_actual: Actual data (after forecast_from_date, for comparison)
        forecast: Forecast DataFrame
        forecast_from_date: Date from which forecast starts
        cycle_metrics: Optional cycle metrics
        halving_averages: HalvingAverages for predicted top/bottom dates
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from src.metrics import HALVING_DATES, get_current_cycle_prediction

    # Combine train + actual for full historical view
    df_full = pd.concat([df_train, df_actual]).drop_duplicates(subset=["ds"]).sort_values("ds")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    forecast_start = pd.to_datetime(forecast_from_date)

    # Get current price for predictions (price at forecast start date)
    price_at_forecast = df_train["y"].iloc[-1] if not df_train.empty else None

    # Calculate predicted top/bottom dates from halving averages
    predicted_top_date = None
    predicted_bottom_date = None
    predicted_top_price = None
    predicted_bottom_price = None

    if halving_averages is not None:
        # Find the relevant halving (most recent one before or during forecast period)
        last_halving = None
        for h in HALVING_DATES:
            if h <= forecast["ds"].max():
                last_halving = h

        if last_halving is not None:
            prediction = get_current_cycle_prediction(
                halving_averages,
                halving_date=last_halving,
                current_price=price_at_forecast,
            )
            predicted_top_date = prediction.get("predicted_top_date")
            predicted_bottom_date = prediction.get("predicted_bottom_date")
            predicted_top_price = prediction.get("predicted_top_price")
            predicted_bottom_price = prediction.get("predicted_bottom_price")

            print(f"\nCycle Predictions (from {last_halving.strftime('%Y-%m-%d')} halving):")
            if predicted_top_date:
                print(f"  Predicted TOP:    {predicted_top_date.strftime('%Y-%m-%d')} @ ${predicted_top_price:,.0f}" if predicted_top_price else f"  Predicted TOP:    {predicted_top_date.strftime('%Y-%m-%d')}")
            if predicted_bottom_date:
                print(f"  Predicted BOTTOM: {predicted_bottom_date.strftime('%Y-%m-%d')} @ ${predicted_bottom_price:,.0f}" if predicted_bottom_price else f"  Predicted BOTTOM: {predicted_bottom_date.strftime('%Y-%m-%d')}")

    # Top: Log scale
    ax1 = axes[0]

    # Historical data (before forecast)
    historical = df_full[df_full["ds"] < forecast_start]
    ax1.plot(historical["ds"], historical["y"], "k-", label="Historical", alpha=0.8, linewidth=1)

    # Actual data (after forecast start) - the "truth" to compare against
    actual_period = df_full[df_full["ds"] >= forecast_start]
    if not actual_period.empty:
        ax1.plot(
            actual_period["ds"],
            actual_period["y"],
            "g-",
            label="Actual (truth)",
            alpha=0.9,
            linewidth=2,
        )

    # Forecast
    forecast_period = forecast[forecast["ds"] >= forecast_start]
    if "yhat_ensemble" in forecast_period.columns:
        ax1.plot(
            forecast_period["ds"],
            forecast_period["yhat_ensemble"],
            "b--",
            label="Forecast",
            alpha=0.8,
            linewidth=1.5,
        )
        ax1.fill_between(
            forecast_period["ds"],
            forecast_period["yhat_ensemble_lower"],
            forecast_period["yhat_ensemble_upper"],
            alpha=0.15,
            color="blue",
            label="95% CI",
        )

    # Halving lines
    for i, h in enumerate(HALVING_DATES):
        if df_full["ds"].min() <= h <= forecast["ds"].max():
            ax1.axvline(x=h, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax1.text(h, ax1.get_ylim()[1] * 0.85, f"H{i+1}", rotation=90, va="top", fontsize=8, color="red")

    # Forecast start line
    ax1.axvline(x=forecast_start, color="purple", linestyle=":", linewidth=2, alpha=0.8)
    ax1.text(
        forecast_start,
        ax1.get_ylim()[1] * 0.7,
        "Forecast Start",
        rotation=90,
        va="top",
        fontsize=9,
        color="purple",
    )

    # Predicted TOP date and price
    if predicted_top_date is not None:
        ax1.axvline(x=predicted_top_date, color="darkred", linestyle="-.", linewidth=2, alpha=0.8)
        if predicted_top_price is not None:
            ax1.axhline(y=predicted_top_price, color="darkred", linestyle=":", linewidth=1, alpha=0.5)
            ax1.scatter([predicted_top_date], [predicted_top_price], marker="v", s=150, c="red",
                       edgecolors="darkred", linewidths=2, zorder=10, label=f"Predicted TOP ${predicted_top_price:,.0f}")

    # Predicted BOTTOM date and price
    if predicted_bottom_date is not None:
        ax1.axvline(x=predicted_bottom_date, color="darkgreen", linestyle="-.", linewidth=2, alpha=0.8)
        if predicted_bottom_price is not None:
            ax1.axhline(y=predicted_bottom_price, color="darkgreen", linestyle=":", linewidth=1, alpha=0.5)
            ax1.scatter([predicted_bottom_date], [predicted_bottom_price], marker="^", s=150, c="limegreen",
                       edgecolors="darkgreen", linewidths=2, zorder=10, label=f"Predicted BOTTOM ${predicted_bottom_price:,.0f}")

    ax1.set_yscale("log")
    ax1.set_ylabel("Price (USD) - Log Scale")
    ax1.set_title(f"BTC Forecast vs Actual (from {forecast_from_date})")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom: Linear scale (zoomed to forecast period)
    ax2 = axes[1]

    # Show some context before forecast
    context_start = forecast_start - timedelta(days=90)
    context_data = df_full[df_full["ds"] >= context_start]

    # Historical context
    hist_context = context_data[context_data["ds"] < forecast_start]
    ax2.plot(hist_context["ds"], hist_context["y"], "k-", label="Historical", alpha=0.8, linewidth=1)

    # Actual
    if not actual_period.empty:
        ax2.plot(
            actual_period["ds"],
            actual_period["y"],
            "g-",
            label="Actual",
            alpha=0.9,
            linewidth=2,
        )

    # Forecast
    if "yhat_ensemble" in forecast_period.columns:
        ax2.plot(
            forecast_period["ds"],
            forecast_period["yhat_ensemble"],
            "b--",
            label="Forecast",
            alpha=0.8,
            linewidth=1.5,
        )
        ax2.fill_between(
            forecast_period["ds"],
            forecast_period["yhat_ensemble_lower"],
            forecast_period["yhat_ensemble_upper"],
            alpha=0.15,
            color="blue",
        )

    ax2.axvline(x=forecast_start, color="purple", linestyle=":", linewidth=2, alpha=0.8)

    # Predicted TOP on linear scale
    if predicted_top_date is not None:
        ax2.axvline(x=predicted_top_date, color="darkred", linestyle="-.", linewidth=2, alpha=0.8)
        if predicted_top_price is not None:
            ax2.axhline(y=predicted_top_price, color="darkred", linestyle=":", linewidth=1, alpha=0.5)
            ax2.scatter([predicted_top_date], [predicted_top_price], marker="v", s=150, c="red",
                       edgecolors="darkred", linewidths=2, zorder=10, label=f"Pred TOP ${predicted_top_price:,.0f}")
            ax2.annotate(f"TOP\n{predicted_top_date.strftime('%Y-%m-%d')}\n${predicted_top_price:,.0f}",
                        xy=(predicted_top_date, predicted_top_price),
                        xytext=(10, 20), textcoords="offset points",
                        fontsize=8, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred", alpha=0.7))

    # Predicted BOTTOM on linear scale
    if predicted_bottom_date is not None:
        ax2.axvline(x=predicted_bottom_date, color="darkgreen", linestyle="-.", linewidth=2, alpha=0.8)
        if predicted_bottom_price is not None:
            ax2.axhline(y=predicted_bottom_price, color="darkgreen", linestyle=":", linewidth=1, alpha=0.5)
            ax2.scatter([predicted_bottom_date], [predicted_bottom_price], marker="^", s=150, c="limegreen",
                       edgecolors="darkgreen", linewidths=2, zorder=10, label=f"Pred BOTTOM ${predicted_bottom_price:,.0f}")
            ax2.annotate(f"BOTTOM\n{predicted_bottom_date.strftime('%Y-%m-%d')}\n${predicted_bottom_price:,.0f}",
                        xy=(predicted_bottom_date, predicted_bottom_price),
                        xytext=(10, -30), textcoords="offset points",
                        fontsize=8, color="darkgreen",
                        arrowprops=dict(arrowstyle="->", color="darkgreen", alpha=0.7))

    ax2.set_ylabel("Price (USD)")
    ax2.set_xlabel("Date")
    ax2.set_title(f"Forecast Period Detail (Linear Scale)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="BTC forecast with cycle-aware signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python forecast_signals.py                          # Forecast from today
  python forecast_signals.py --from-date 2024-01-01   # Backtest from Jan 2024
  python forecast_signals.py --from-date 2023-06-01 --days 365
        """,
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Date to start forecast from (YYYY-MM-DD). If set, trains on data before this date "
        "and compares forecast to actual data after.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Forecast horizon in days (default: FORECAST_DAYS env var or {FORECAST_DAYS})",
    )
    parser.add_argument(
        "--no-signals",
        action="store_true",
        help="Skip signal generation and backtest (faster, forecast only)",
    )
    args = parser.parse_args()

    forecast_days = args.days or FORECAST_DAYS
    from_date = args.from_date

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = TeeOutput()
    sys.stdout = tee

    try:
        start_date = "2015-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Run timestamp: {timestamp}")
        print(f"Fetching BTC data from {start_date} to {end_date}...")
        df_full = fetch_btc_data(start_date, end_date)
        df_full = df_full.sort_values("ds").reset_index(drop=True)
        print(f"Loaded {len(df_full)} rows.")

        # If from_date specified, split data for backtesting
        if from_date:
            forecast_start = pd.to_datetime(from_date)
            df_train = df_full[df_full["ds"] < forecast_start].copy()
            df_actual = df_full[df_full["ds"] >= forecast_start].copy()

            if df_train.empty:
                print(f"ERROR: No data before {from_date}")
                return
            if df_actual.empty:
                print(f"WARNING: No actual data after {from_date} to compare against")
                df_actual = None

            print(f"\nBacktest mode: Training on data before {from_date}")
            print(f"  Training data: {len(df_train)} days ({df_train['ds'].min().date()} to {df_train['ds'].max().date()})")
            if df_actual is not None:
                print(f"  Actual data for comparison: {len(df_actual)} days")
            df = df_train
        else:
            df = df_full
            df_actual = None

        print("\nComputing halving cycle metrics...")
        cycle_metrics = compute_cycle_metrics(df)
        averages = compute_halving_averages(cycle_metrics=cycle_metrics)
        print_halving_summary(cycle_metrics, averages)

        print(f"\nTraining cycle-aware ensemble model (forecasting {forecast_days} days)...")
        forecast = train_simple_ensemble(
            df,
            periods=forecast_days,
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
        )

        # If backtesting, compute and show accuracy
        if from_date and df_actual is not None and not df_actual.empty:
            metrics = compute_forecast_accuracy(forecast, df_actual)
            print_accuracy_report(metrics, from_date, df_actual["ds"].max().strftime("%Y-%m-%d"))

        if not args.no_signals:
            print("\nGenerating buy/sell signals...")
            df_signals = generate_signals(df, cycle_weight=0.4, technical_weight=0.6)
            current = get_current_signal(df_signals, cycle_metrics=cycle_metrics, averages=averages)

            print("Running backtest...")
            backtest = backtest_signals(df_signals, initial_capital=10000)
            print_signal_summary(current, backtest)
        else:
            df_signals = df.copy()
            print("\nSkipping signal generation (--no-signals)")

        print(f"\nSaving results to {REPORTS_DIR}/...")
        suffix = f"_{from_date}" if from_date else ""
        signals_path = REPORTS_DIR / f"signals_{timestamp}{suffix}.csv"
        forecast_path = REPORTS_DIR / f"forecast_{timestamp}{suffix}.csv"
        report_path = REPORTS_DIR / f"report_{timestamp}{suffix}.txt"

        df_signals.to_csv(signals_path, index=False)
        forecast.to_csv(forecast_path, index=False)
        print(f"  Signals:  {signals_path.name}")
        print(f"  Forecast: {forecast_path.name}")
        print(f"  Report:   {report_path.name}")

        print("\nPlotting...")
        if from_date and df_actual is not None:
            # Use special backtest plot showing forecast vs actual
            plot_signals_with_actual(
                df_train=df,
                df_actual=df_actual,
                forecast=forecast,
                forecast_from_date=from_date,
                cycle_metrics=cycle_metrics,
                halving_averages=averages,
            )
        else:
            # Standard plot
            plot_signals(df_signals, forecast, cycle_metrics=cycle_metrics)

        print("\nDone!")

    finally:
        sys.stdout = tee.stdout
        suffix = f"_{from_date}" if from_date else ""
        with open(REPORTS_DIR / f"report_{timestamp}{suffix}.txt", "w") as f:
            f.write(tee.getvalue())


if __name__ == "__main__":
    main()
