#!/usr/bin/env python3
"""Plot forecast validation - train on historical data, show predictions vs actuals.

This shows the TRUE performance of the forecast by:
1. Training on data up to a cutoff date (e.g., 2024-12-31)
2. Generating predictions from that point forward
3. Comparing predictions to actual prices
"""

from datetime import datetime, timedelta
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


def main(cutoff_date: str = "2024-12-31"):
    print("=" * 70)
    print("FORECAST VALIDATION CHART")
    print(f"Training cutoff: {cutoff_date}")
    print("=" * 70)

    # 1. Load ALL data
    print("\n[1/4] Loading BTC data...")
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    df_all = fetch_btc_data(start_date=start_date, end_date=end_date)
    print(f"  Total: {len(df_all)} days")

    # 2. Split into train/test
    cutoff = pd.Timestamp(cutoff_date)
    df_train = df_all[df_all["ds"] <= cutoff].copy()
    df_test = df_all[df_all["ds"] > cutoff].copy()
    print(f"  Training: {len(df_train)} days (up to {cutoff_date})")
    print(f"  Test: {len(df_test)} days ({df_test['ds'].min().date()} to {df_test['ds'].max().date()})")

    # 3. Compute cycle metrics using ONLY training data
    print("\n[2/4] Computing cycle metrics (training data only)...")
    cycle_metrics = compute_cycle_metrics(df_train)
    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print(f"  {averages.n_cycles} cycles analyzed")

    # 4. Train forecaster on training data only
    print("\n[3/4] Training forecaster on data up to {cutoff_date}...")
    forecaster = ProphetCycleForecaster(
        halving_averages=averages,
        cycle_metrics=cycle_metrics,
    )
    # Forecast far enough to cover test period
    forecast_periods = len(df_test) + 365
    result = forecaster.fit_predict(df_train, periods=forecast_periods)
    forecast = result.forecast
    print(f"  Forecast from {cutoff_date} through {forecast['ds'].max().date()}")

    # 5. Create chart
    print("\n[4/4] Creating validation chart...")
    fig = create_validation_chart(df_train, df_test, forecast, cutoff_date, cycle_metrics, averages)

    # Save
    output_path = Path("forecast_validation_chart.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved to: {output_path.absolute()}")

    # Print accuracy summary
    print_accuracy_summary(df_test, forecast)

    plt.show()


def create_validation_chart(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    forecast: pd.DataFrame,
    cutoff_date: str,
    cycle_metrics: pd.DataFrame,
    averages,
) -> plt.Figure:
    """Create validation chart showing forecast vs actual."""

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    fig.suptitle(f"Forecast Validation: Trained on data up to {cutoff_date}",
                 fontsize=14, fontweight="bold")

    cutoff = pd.Timestamp(cutoff_date)
    forecast_col = "yhat_ensemble" if "yhat_ensemble" in forecast.columns else "yhat"

    # =================================================================
    # Panel 1: Price vs Forecast
    # =================================================================
    ax1 = axes[0]

    # Training data (gray, before cutoff)
    ax1.plot(df_train["ds"], df_train["y"], color="gray", linewidth=1,
             alpha=0.5, label="Training Data (used for forecast)")

    # Actual price AFTER cutoff (black, solid)
    ax1.plot(df_test["ds"], df_test["y"], color="black", linewidth=2,
             label="Actual Price (not seen by model)", zorder=5)

    # Forecast AFTER cutoff (blue, with CI)
    forecast_after = forecast[forecast["ds"] > cutoff].copy()
    ax1.plot(forecast_after["ds"], forecast_after[forecast_col],
             color="blue", linewidth=2, alpha=0.8, label="Forecast (predicted)")
    ax1.fill_between(forecast_after["ds"],
                     forecast_after[f"{forecast_col}_lower"],
                     forecast_after[f"{forecast_col}_upper"],
                     color="blue", alpha=0.15, label="95% CI")

    # Cutoff line
    ymin, ymax = ax1.get_ylim()
    ax1.axvline(cutoff, color="purple", linestyle="--", linewidth=3, alpha=0.8)
    ax1.annotate(f"FORECAST STARTS HERE\n{cutoff_date}\n(model trained on data before this)",
                xy=(cutoff, ymax * 0.7),
                fontsize=10, color="purple", ha="right", va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95, edgecolor="purple", linewidth=2))

    # Halving dates
    for h in HALVING_DATES:
        if df_train["ds"].min() <= h <= forecast["ds"].max():
            ax1.axvline(h, color="orange", linestyle="-", alpha=0.4, linewidth=1.5)
            ax1.annotate("Halving", xy=(h, ymax * 0.95), fontsize=8, color="orange", ha="center")

    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Zoom to relevant period (1 year before cutoff to end)
    zoom_start = cutoff - timedelta(days=365)
    ax1.set_xlim(zoom_start, forecast["ds"].max())

    # =================================================================
    # Panel 2: Forecast Error Over Time
    # =================================================================
    ax2 = axes[1]

    # Merge forecast with actual
    comparison = pd.merge(
        df_test[["ds", "y"]],
        forecast[["ds", forecast_col]],
        on="ds",
        how="inner",
    )
    comparison = comparison[comparison["y"] > 0].copy()  # Filter invalid
    comparison["error_pct"] = (comparison[forecast_col] - comparison["y"]) / comparison["y"] * 100

    # Plot error
    ax2.fill_between(comparison["ds"], 0, comparison["error_pct"],
                     where=comparison["error_pct"] > 0, color="red", alpha=0.4, label="Over-predicted")
    ax2.fill_between(comparison["ds"], 0, comparison["error_pct"],
                     where=comparison["error_pct"] < 0, color="green", alpha=0.4, label="Under-predicted")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax2.axhline(25, color="gray", linestyle="--", alpha=0.3)
    ax2.axhline(-25, color="gray", linestyle="--", alpha=0.3)

    # Cutoff line
    ax2.axvline(cutoff, color="purple", linestyle="--", linewidth=2, alpha=0.8)

    ax2.set_ylabel("Forecast Error (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-100, 200)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.set_xlim(zoom_start, forecast["ds"].max())

    plt.tight_layout()
    return fig


def print_accuracy_summary(df_test: pd.DataFrame, forecast: pd.DataFrame):
    """Print accuracy metrics."""
    forecast_col = "yhat_ensemble" if "yhat_ensemble" in forecast.columns else "yhat"

    comparison = pd.merge(
        df_test[["ds", "y"]],
        forecast[["ds", forecast_col]],
        on="ds",
        how="inner",
    )
    comparison = comparison[comparison["y"] > 0].copy()
    comparison["error_pct"] = (comparison[forecast_col] - comparison["y"]) / comparison["y"] * 100

    print("\n" + "=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)

    # Monthly averages
    comparison["month"] = comparison["ds"].dt.to_period("M")
    monthly = comparison.groupby("month").agg({
        "y": "mean",
        forecast_col: "mean",
        "error_pct": "mean",
    }).reset_index()

    print(f"\n{'Month':<10} {'Actual':>12} {'Predicted':>12} {'Error':>10}")
    print("-" * 50)
    for _, row in monthly.iterrows():
        actual = row["y"]
        pred = row[forecast_col]
        err = row["error_pct"]
        print(f"{str(row['month']):<10} ${actual:>10,.0f} ${pred:>10,.0f} {err:>+9.1f}%")

    # Overall metrics
    mape = comparison["error_pct"].abs().mean()
    mean_error = comparison["error_pct"].mean()
    print(f"\nOverall MAPE: {mape:.1f}%")
    print(f"Mean Error (bias): {mean_error:+.1f}%")


if __name__ == "__main__":
    import sys
    cutoff = sys.argv[1] if len(sys.argv) > 1 else "2024-12-31"
    main(cutoff_date=cutoff)
