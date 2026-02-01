#!/usr/bin/env python3
"""Validate forecast accuracy by training on historical data and comparing to actuals.

This script:
1. Trains the model using data up to a cutoff date (e.g., end of 2022)
2. Generates predictions for the period after the cutoff
3. Compares predictions vs actual prices
4. Calculates error metrics and visualizes the comparison
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


def validate_forecast(
    cutoff_date: str = "2022-12-31",
    forecast_periods: int = 365 * 3,  # 3 years forward
):
    """Run forecast validation.

    Args:
        cutoff_date: Train on data up to this date
        forecast_periods: Number of days to forecast forward
    """
    print("=" * 70)
    print("FORECAST VALIDATION")
    print(f"Training cutoff: {cutoff_date}")
    print("=" * 70)

    # 1. Load ALL data (for comparison)
    print("\n[1/5] Loading data...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    df_all = fetch_btc_data(start_date="2012-01-01", end_date=end_date)
    print(f"  Total data: {len(df_all)} days ({df_all['ds'].min().date()} to {df_all['ds'].max().date()})")

    # 2. Split into train/test
    cutoff = pd.Timestamp(cutoff_date)
    df_train = df_all[df_all["ds"] <= cutoff].copy()
    df_test = df_all[df_all["ds"] > cutoff].copy()

    print(f"  Training: {len(df_train)} days (up to {cutoff_date})")
    print(f"  Test: {len(df_test)} days ({df_test['ds'].min().date()} to {df_test['ds'].max().date()})")

    # 3. Compute cycle metrics using ONLY training data
    print("\n[2/5] Computing cycle metrics (training data only)...")
    cycle_metrics = compute_cycle_metrics(df_train)
    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print(f"  {averages.n_cycles} cycles analyzed")
    print(f"  Avg days to top: {averages.avg_days_to_top:.0f}")
    print(f"  Avg days to bottom: {averages.avg_days_to_bottom:.0f}")

    # 4. Train forecaster and predict (using exact same config as main forecast)
    print("\n[3/5] Training forecaster...")

    forecaster = ProphetCycleForecaster(
        halving_averages=averages,
        cycle_metrics=cycle_metrics,
    )
    result = forecaster.fit_predict(df_train, periods=forecast_periods)
    forecast = result.forecast
    print(f"  Forecast through {forecast['ds'].max().date()}")

    # 5. Compare forecast to actuals
    print("\n[4/5] Computing accuracy metrics...")

    # Merge forecast with actual test data
    forecast_col = "yhat_ensemble" if "yhat_ensemble" in forecast.columns else "yhat"
    comparison = pd.merge(
        df_test[["ds", "y"]],
        forecast[["ds", forecast_col, f"{forecast_col}_lower", f"{forecast_col}_upper"]],
        on="ds",
        how="inner",
    )
    comparison = comparison.rename(columns={
        "y": "actual",
        forecast_col: "predicted",
        f"{forecast_col}_lower": "predicted_lower",
        f"{forecast_col}_upper": "predicted_upper",
    })

    # Calculate errors
    comparison["error"] = comparison["predicted"] - comparison["actual"]
    comparison["abs_error"] = comparison["error"].abs()
    comparison["pct_error"] = (comparison["error"] / comparison["actual"]) * 100
    comparison["abs_pct_error"] = comparison["pct_error"].abs()

    # Check if actual is within confidence interval
    comparison["within_ci"] = (
        (comparison["actual"] >= comparison["predicted_lower"]) &
        (comparison["actual"] <= comparison["predicted_upper"])
    )

    # Calculate metrics
    mape = comparison["abs_pct_error"].mean()
    rmse = np.sqrt((comparison["error"] ** 2).mean())
    mae = comparison["abs_error"].mean()
    ci_coverage = comparison["within_ci"].mean() * 100

    # Direction accuracy (did we predict up/down correctly?)
    comparison["actual_direction"] = comparison["actual"].diff().apply(lambda x: 1 if x > 0 else -1)
    comparison["predicted_direction"] = comparison["predicted"].diff().apply(lambda x: 1 if x > 0 else -1)
    direction_accuracy = (comparison["actual_direction"] == comparison["predicted_direction"]).mean() * 100

    print(f"\n  ACCURACY METRICS ({len(comparison)} days)")
    print(f"  {'-'*40}")
    print(f"  MAPE (Mean Abs % Error):  {mape:.1f}%")
    print(f"  RMSE:                     ${rmse:,.0f}")
    print(f"  MAE:                      ${mae:,.0f}")
    print(f"  95% CI Coverage:          {ci_coverage:.1f}%")
    print(f"  Direction Accuracy:       {direction_accuracy:.1f}%")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN")
    print(f"  {'-'*40}")
    comparison["month"] = comparison["ds"].dt.to_period("M")
    monthly = comparison.groupby("month").agg({
        "abs_pct_error": "mean",
        "within_ci": "mean",
        "actual": "last",
        "predicted": "last",
    }).reset_index()

    for _, row in monthly.iterrows():
        month_str = str(row["month"])
        mape_m = row["abs_pct_error"]
        ci_m = row["within_ci"] * 100
        actual_m = row["actual"]
        pred_m = row["predicted"]
        print(f"  {month_str}: MAPE={mape_m:.1f}%, CI={ci_m:.0f}%, Actual=${actual_m:,.0f}, Pred=${pred_m:,.0f}")

    # 6. Plot
    print("\n[5/5] Creating visualization...")
    fig = create_validation_chart(df_train, df_test, forecast, comparison, cutoff_date, averages)

    # Save
    output_path = Path("forecast_validation.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  Saved to: {output_path.absolute()}")

    plt.show()

    return comparison


def create_validation_chart(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    forecast: pd.DataFrame,
    comparison: pd.DataFrame,
    cutoff_date: str,
    averages,
) -> plt.Figure:
    """Create validation visualization."""

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
    fig.suptitle(f"Forecast Validation (trained on data up to {cutoff_date})",
                 fontsize=14, fontweight="bold")

    forecast_col = "yhat_ensemble" if "yhat_ensemble" in forecast.columns else "yhat"
    cutoff = pd.Timestamp(cutoff_date)

    # =================================================================
    # Panel 1: Price vs Forecast
    # =================================================================
    ax1 = axes[0]

    # Training data (before cutoff)
    ax1.plot(df_train["ds"], df_train["y"], color="black", linewidth=1.5,
             label="Training Data", alpha=0.7)

    # Test data (after cutoff) - ACTUAL
    ax1.plot(df_test["ds"], df_test["y"], color="black", linewidth=2,
             label="Actual Price (test period)", zorder=5)

    # Forecast
    forecast_after = forecast[forecast["ds"] > cutoff]
    ax1.plot(forecast_after["ds"], forecast_after[forecast_col],
             color="blue", linewidth=1.5, alpha=0.8, label="Forecast")
    ax1.fill_between(forecast_after["ds"],
                     forecast_after[f"{forecast_col}_lower"],
                     forecast_after[f"{forecast_col}_upper"],
                     color="blue", alpha=0.15, label="95% CI")

    # Cutoff line
    ax1.axvline(cutoff, color="purple", linestyle="--", linewidth=2, alpha=0.7)
    ax1.annotate(f"CUTOFF\n{cutoff_date}\n(training ends)",
                xy=(cutoff, ax1.get_ylim()[1] * 0.5 if ax1.get_ylim()[1] > 0 else 50000),
                fontsize=9, color="purple", ha="right", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="purple"))

    # Halving dates
    for h in HALVING_DATES:
        if df_train["ds"].min() <= h <= forecast["ds"].max():
            ax1.axvline(h, color="orange", linestyle="-", alpha=0.5, linewidth=1.5)

    ax1.set_ylabel("Price (USD)", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # =================================================================
    # Panel 2: Prediction Error (%)
    # =================================================================
    ax2 = axes[1]

    ax2.fill_between(comparison["ds"], 0, comparison["pct_error"],
                     where=comparison["pct_error"] > 0, color="red", alpha=0.4, label="Over-predicted")
    ax2.fill_between(comparison["ds"], 0, comparison["pct_error"],
                     where=comparison["pct_error"] < 0, color="green", alpha=0.4, label="Under-predicted")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)

    # Add ±20% reference lines
    ax2.axhline(20, color="red", linestyle="--", alpha=0.3)
    ax2.axhline(-20, color="green", linestyle="--", alpha=0.3)

    mape = comparison["abs_pct_error"].mean()
    ax2.set_ylabel("Error (%)", fontsize=10)
    ax2.set_title(f"Prediction Error (MAPE: {mape:.1f}%)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # =================================================================
    # Panel 3: Cumulative Error
    # =================================================================
    ax3 = axes[2]

    comparison["cumulative_error"] = comparison["pct_error"].cumsum()
    ax3.plot(comparison["ds"], comparison["cumulative_error"], color="purple", linewidth=1.5)
    ax3.fill_between(comparison["ds"], 0, comparison["cumulative_error"],
                     where=comparison["cumulative_error"] > 0, color="red", alpha=0.2)
    ax3.fill_between(comparison["ds"], 0, comparison["cumulative_error"],
                     where=comparison["cumulative_error"] < 0, color="green", alpha=0.2)
    ax3.axhline(0, color="black", linestyle="-", linewidth=0.5)

    ax3.set_ylabel("Cumulative Error (%)", fontsize=10)
    ax3.set_xlabel("Date", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Validate using data up to end of 2022
    validate_forecast(cutoff_date="2022-12-31", forecast_periods=365 * 4)
