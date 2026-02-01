"""Ablation testing for Bitcoin forecasting regressors.

Run forecasts with each regressor disabled to measure individual contribution.
Provides metrics: MAPE, RMSE, and Sharpe ratio from backtest.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import (
    compute_cycle_metrics,
    compute_halving_averages,
    fit_decay_curve,
)
from src.models.ensemble import train_simple_ensemble, evaluate_forecast
from src.models.backtest import run_backtest, BacktestConfig
from src.models.signals import generate_signals


@dataclass
class AblationResult:
    """Results from testing a regressor configuration."""

    config_name: str
    enabled_regressors: dict[str, bool]

    # Forecast metrics
    mape: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0

    # Backtest metrics
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Comparison to baseline
    mape_delta: float = 0.0  # Negative = better than baseline
    sharpe_delta: float = 0.0  # Positive = better than baseline

    # Additional info
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: list[str] = field(default_factory=list)


def _set_regressor_env(config: dict[str, bool]) -> dict[str, str]:
    """Set environment variables for regressor toggles. Returns original values."""
    original = {}
    for name, enabled in config.items():
        env_key = f"REGRESSOR_{name.upper()}"
        original[env_key] = os.environ.get(env_key, "")
        os.environ[env_key] = str(enabled).lower()
    return original


def _restore_env(original: dict[str, str]) -> None:
    """Restore original environment variables."""
    for key, value in original.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


def run_single_ablation(
    df: pd.DataFrame,
    config: dict[str, bool],
    config_name: str,
    halving_averages: Any,
    cycle_metrics: pd.DataFrame,
    holdout_days: int = 90,
    forecast_periods: int = 365,
) -> AblationResult:
    """Run a single ablation test with the specified regressor configuration.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        config: Dict of regressor name -> enabled status.
        config_name: Human-readable name for this configuration.
        halving_averages: HalvingAverages object.
        cycle_metrics: DataFrame from compute_cycle_metrics.
        holdout_days: Days to hold out for evaluation.
        forecast_periods: Days to forecast.

    Returns:
        AblationResult with metrics.
    """
    result = AblationResult(
        config_name=config_name,
        enabled_regressors=config.copy(),
    )

    # Set environment variables
    original_env = _set_regressor_env(config)

    try:
        # Run forecast
        forecast = train_simple_ensemble(
            df,
            periods=forecast_periods,
            halving_averages=halving_averages,
            cycle_metrics=cycle_metrics,
        )
        result.forecast = forecast

        # Evaluate forecast accuracy on holdout period
        metrics = evaluate_forecast(df, forecast, holdout_days=holdout_days)

        if "error" not in metrics:
            result.mape = metrics.get("ensemble_mape", metrics.get("prophet_mape", 0))
            result.rmse = metrics.get("ensemble_rmse", metrics.get("prophet_rmse", 0))
            result.mae = metrics.get("ensemble_mae", metrics.get("prophet_mae", 0))

        # Generate signals and run backtest
        try:
            signals_df = generate_signals(
                df,
                forecast,
                halving_averages=halving_averages,
            )

            bt_config = BacktestConfig(
                initial_capital=10000,
                position_size=0.25,
            )
            bt_result = run_backtest(signals_df, bt_config)

            result.sharpe_ratio = bt_result.sharpe_ratio
            result.calmar_ratio = bt_result.calmar_ratio
            result.total_return_pct = bt_result.total_return_pct
            result.max_drawdown_pct = bt_result.max_drawdown_pct
        except Exception as e:
            result.errors.append(f"Backtest error: {e}")

    except Exception as e:
        result.errors.append(f"Forecast error: {e}")
    finally:
        _restore_env(original_env)

    return result


def run_ablation_study(
    df: pd.DataFrame,
    holdout_days: int = 90,
    forecast_periods: int = 365,
) -> list[AblationResult]:
    """Run a full ablation study testing each regressor's contribution.

    Tests configurations:
    1. Baseline (all regressors enabled)
    2. Pure Prophet (all regressors disabled)
    3. Each regressor disabled individually

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        holdout_days: Days to hold out for evaluation.
        forecast_periods: Days to forecast.

    Returns:
        List of AblationResult, one per configuration tested.
    """
    # Compute cycle metrics
    cycle_metrics = compute_cycle_metrics(df)
    halving_averages = compute_halving_averages(cycle_metrics=cycle_metrics)

    # Define regressor names
    regressor_names = ["cycle", "double_top", "cycle_phase", "decay", "ensemble_adjust"]

    # Define test configurations
    configs = []

    # Baseline: all enabled
    configs.append((
        "baseline_all_on",
        {name: True for name in regressor_names}
    ))

    # Pure Prophet: all disabled
    configs.append((
        "pure_prophet",
        {name: False for name in regressor_names}
    ))

    # Ablate each regressor individually
    for ablate_name in regressor_names:
        config = {name: (name != ablate_name) for name in regressor_names}
        configs.append((f"without_{ablate_name}", config))

    # Run all configurations
    results = []
    baseline_result = None

    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    for i, (name, config) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {name}")
        enabled = [k for k, v in config.items() if v]
        disabled = [k for k, v in config.items() if not v]
        print(f"  Enabled:  {', '.join(enabled) if enabled else 'none'}")
        print(f"  Disabled: {', '.join(disabled) if disabled else 'none'}")

        result = run_single_ablation(
            df=df,
            config=config,
            config_name=name,
            halving_averages=halving_averages,
            cycle_metrics=cycle_metrics,
            holdout_days=holdout_days,
            forecast_periods=forecast_periods,
        )

        if name == "baseline_all_on":
            baseline_result = result

        # Compute deltas from baseline
        if baseline_result and name != "baseline_all_on":
            result.mape_delta = result.mape - baseline_result.mape
            result.sharpe_delta = result.sharpe_ratio - baseline_result.sharpe_ratio

        results.append(result)

        if result.errors:
            print(f"  Errors: {result.errors}")
        else:
            print(f"  MAPE: {result.mape:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")

    return results


def print_ablation_report(results: list[AblationResult]) -> None:
    """Print a formatted ablation study report."""

    print("\n" + "=" * 90)
    print("ABLATION STUDY REPORT")
    print("=" * 90)

    # Find baseline
    baseline = next((r for r in results if r.config_name == "baseline_all_on"), None)

    print("\n--- FORECAST ACCURACY ---")
    print(f"{'Configuration':<25} {'MAPE':>10} {'RMSE':>12} {'vs Baseline':>12}")
    print("-" * 60)

    for r in results:
        delta_str = ""
        if baseline and r.config_name != "baseline_all_on":
            delta = r.mape - baseline.mape
            delta_str = f"{delta:+.2f}%"
        print(f"{r.config_name:<25} {r.mape:>9.2f}% {r.rmse:>11,.0f} {delta_str:>12}")

    print("\n--- BACKTEST PERFORMANCE ---")
    print(f"{'Configuration':<25} {'Sharpe':>10} {'Calmar':>10} {'Return':>10} {'MaxDD':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.config_name:<25} {r.sharpe_ratio:>10.2f} {r.calmar_ratio:>10.2f} "
              f"{r.total_return_pct:>9.1f}% {r.max_drawdown_pct:>9.1f}%")

    # Identify most impactful regressors
    ablations = [r for r in results if r.config_name.startswith("without_")]
    if ablations and baseline:
        print("\n--- REGRESSOR CONTRIBUTION RANKING ---")
        print("(Positive MAPE delta = regressor helps, Negative = regressor hurts)")
        print()

        # Sort by MAPE delta (higher = removing hurt more = regressor helps more)
        sorted_by_mape = sorted(ablations, key=lambda r: -r.mape_delta)

        for r in sorted_by_mape:
            regressor = r.config_name.replace("without_", "")
            impact = "HELPS" if r.mape_delta > 0 else "HURTS" if r.mape_delta < 0 else "NEUTRAL"
            print(f"  {regressor:<20} MAPE delta: {r.mape_delta:+.2f}%  ({impact})")

    print("\n" + "=" * 90)


def compute_regressor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix between regressor values.

    High correlation (> 0.5) suggests redundancy between regressors.

    Args:
        df: DataFrame with regressor columns.

    Returns:
        Correlation matrix as DataFrame.
    """
    regressor_cols = [
        "reg_cycle_sin", "reg_cycle_cos", "reg_pre_halving", "reg_post_halving",
        "double_top_regressor", "cycle_phase_regressor", "decay_regressor",
    ]

    available = [c for c in regressor_cols if c in df.columns]

    if not available:
        return pd.DataFrame()

    return df[available].corr()


if __name__ == "__main__":
    # Example usage
    from src.db.btc_data import fetch_btc_data

    print("Fetching BTC data...")
    df = fetch_btc_data()

    print("Running ablation study...")
    results = run_ablation_study(df, holdout_days=90)

    print_ablation_report(results)
