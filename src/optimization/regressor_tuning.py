"""Hyperparameter optimization for Bitcoin forecasting regressors.

Uses Optuna to search for optimal:
- Gaussian parameters (peak_day, spread)
- Window widths for DOUBLE_TOP
- Boost/reduction percentages for ENSEMBLE_ADJUST

Objective: minimize MAPE on walk-forward validation.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Run: pip install optuna")


@dataclass
class RegressorParams:
    """Tunable regressor parameters."""

    # CYCLE regressor
    post_halving_peak_day: int = 240
    post_halving_spread: int = 150
    pre_halving_window: int = 365

    # DOUBLE_TOP regressor
    first_top_day: int = 280
    second_top_day: int = 520
    top_window_spread: int = 45

    # DECAY regressor
    drawdown_start_day: int = 365
    drawdown_peak_day: int = 550
    drawdown_spread: int = 150
    max_decay_effect: float = 0.05

    # ENSEMBLE_ADJUST
    pre_halving_boost: float = 0.15
    bull_run_boost: float = 0.12
    dist_reduction: float = 0.03


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""

    best_params: RegressorParams
    best_mape: float
    best_sharpe: float
    n_trials: int
    study_name: str
    optimization_history: list[dict] = field(default_factory=list)


def _set_params_env(params: RegressorParams) -> dict[str, str]:
    """Set regressor parameters as environment variables. Returns original values."""
    env_vars = {
        "REGRESSOR_POST_HALVING_PEAK_DAY": str(params.post_halving_peak_day),
        "REGRESSOR_POST_HALVING_SPREAD": str(params.post_halving_spread),
        "REGRESSOR_PRE_HALVING_WINDOW": str(params.pre_halving_window),
        "REGRESSOR_FIRST_TOP_DAY": str(params.first_top_day),
        "REGRESSOR_SECOND_TOP_DAY": str(params.second_top_day),
        "REGRESSOR_TOP_WINDOW_SPREAD": str(params.top_window_spread),
        "REGRESSOR_DRAWDOWN_START_DAY": str(params.drawdown_start_day),
        "REGRESSOR_DRAWDOWN_PEAK_DAY": str(params.drawdown_peak_day),
        "REGRESSOR_DRAWDOWN_SPREAD": str(params.drawdown_spread),
        "REGRESSOR_MAX_DECAY_EFFECT": str(params.max_decay_effect),
        "REGRESSOR_PRE_HALVING_BOOST": str(params.pre_halving_boost),
        "REGRESSOR_BULL_RUN_BOOST": str(params.bull_run_boost),
        "REGRESSOR_DIST_REDUCTION": str(params.dist_reduction),
    }

    original = {}
    for key, value in env_vars.items():
        original[key] = os.environ.get(key, "")
        os.environ[key] = value

    return original


def _restore_env(original: dict[str, str]) -> None:
    """Restore original environment variables."""
    for key, value in original.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]


def create_objective(
    df: pd.DataFrame,
    halving_averages: Any,
    cycle_metrics: pd.DataFrame,
    holdout_days: int = 90,
    forecast_periods: int = 365,
    metric: str = "mape",
) -> callable:
    """Create an Optuna objective function for regressor tuning.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        halving_averages: HalvingAverages object.
        cycle_metrics: DataFrame from compute_cycle_metrics.
        holdout_days: Days to hold out for evaluation.
        forecast_periods: Days to forecast.
        metric: Metric to optimize ('mape', 'sharpe', or 'combined').

    Returns:
        Objective function for Optuna.
    """
    from src.models.ensemble import train_simple_ensemble, evaluate_forecast
    from src.models.backtest import run_backtest, BacktestConfig
    from src.models.signals import generate_signals

    def objective(trial: "optuna.Trial") -> float:
        # Sample hyperparameters
        params = RegressorParams(
            # CYCLE timing
            post_halving_peak_day=trial.suggest_int("post_halving_peak_day", 180, 400),
            post_halving_spread=trial.suggest_int("post_halving_spread", 80, 250),
            pre_halving_window=trial.suggest_int("pre_halving_window", 180, 545),

            # DOUBLE_TOP timing
            first_top_day=trial.suggest_int("first_top_day", 200, 400),
            second_top_day=trial.suggest_int("second_top_day", 400, 700),
            top_window_spread=trial.suggest_int("top_window_spread", 30, 90),

            # DECAY timing
            drawdown_start_day=trial.suggest_int("drawdown_start_day", 300, 450),
            drawdown_peak_day=trial.suggest_int("drawdown_peak_day", 450, 700),
            drawdown_spread=trial.suggest_int("drawdown_spread", 100, 250),
            max_decay_effect=trial.suggest_float("max_decay_effect", 0.02, 0.10),

            # ENSEMBLE_ADJUST magnitudes
            pre_halving_boost=trial.suggest_float("pre_halving_boost", 0.05, 0.25),
            bull_run_boost=trial.suggest_float("bull_run_boost", 0.05, 0.20),
            dist_reduction=trial.suggest_float("dist_reduction", 0.01, 0.10),
        )

        # Set parameters as environment variables
        original_env = _set_params_env(params)

        try:
            # Run forecast
            forecast = train_simple_ensemble(
                df,
                periods=forecast_periods,
                halving_averages=halving_averages,
                cycle_metrics=cycle_metrics,
            )

            # Evaluate forecast accuracy
            metrics = evaluate_forecast(df, forecast, holdout_days=holdout_days)

            if "error" in metrics:
                return float("inf")

            mape = metrics.get("ensemble_mape", metrics.get("prophet_mape", float("inf")))

            # Optionally run backtest for Sharpe
            sharpe = 0.0
            if metric in ["sharpe", "combined"]:
                try:
                    signals_df = generate_signals(df, forecast, halving_averages=halving_averages)
                    bt_config = BacktestConfig(initial_capital=10000, position_size=0.25)
                    bt_result = run_backtest(signals_df, bt_config)
                    sharpe = bt_result.sharpe_ratio
                except Exception:
                    sharpe = 0.0

            # Return objective value (lower is better for MAPE, higher for Sharpe)
            if metric == "mape":
                return mape
            elif metric == "sharpe":
                return -sharpe  # Negate because Optuna minimizes
            else:  # combined
                # Weighted combination: prioritize MAPE but reward Sharpe
                return mape - 0.5 * sharpe

        except Exception as e:
            print(f"Trial failed: {e}")
            return float("inf")
        finally:
            _restore_env(original_env)

    return objective


def tune_regressors(
    df: pd.DataFrame,
    halving_averages: Any = None,
    cycle_metrics: pd.DataFrame | None = None,
    n_trials: int = 50,
    holdout_days: int = 90,
    forecast_periods: int = 365,
    metric: str = "mape",
    study_name: str = "regressor_tuning",
    timeout: int | None = None,
) -> TuningResult:
    """Run hyperparameter optimization for regressor parameters.

    Uses Optuna with TPE sampler for efficient search.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        halving_averages: HalvingAverages object. If None, will be computed.
        cycle_metrics: DataFrame from compute_cycle_metrics. If None, computed.
        n_trials: Number of optimization trials.
        holdout_days: Days to hold out for evaluation.
        forecast_periods: Days to forecast.
        metric: Metric to optimize ('mape', 'sharpe', 'combined').
        study_name: Name for the Optuna study.
        timeout: Optional timeout in seconds.

    Returns:
        TuningResult with best parameters and metrics.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("optuna is required for hyperparameter tuning. Install with: pip install optuna")

    # Compute cycle metrics if not provided
    if cycle_metrics is None or halving_averages is None:
        from src.metrics import compute_cycle_metrics, compute_halving_averages
        cycle_metrics = compute_cycle_metrics(df)
        halving_averages = compute_halving_averages(cycle_metrics=cycle_metrics)

    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER OPTIMIZATION")
    print(f"  Metric: {metric}")
    print(f"  Trials: {n_trials}")
    print(f"  Holdout: {holdout_days} days")
    print("=" * 70)

    # Create study with TPE sampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
    )

    # Create objective function
    objective = create_objective(
        df=df,
        halving_averages=halving_averages,
        cycle_metrics=cycle_metrics,
        holdout_days=holdout_days,
        forecast_periods=forecast_periods,
        metric=metric,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Extract best parameters
    best_trial = study.best_trial
    best_params = RegressorParams(
        post_halving_peak_day=best_trial.params["post_halving_peak_day"],
        post_halving_spread=best_trial.params["post_halving_spread"],
        pre_halving_window=best_trial.params["pre_halving_window"],
        first_top_day=best_trial.params["first_top_day"],
        second_top_day=best_trial.params["second_top_day"],
        top_window_spread=best_trial.params["top_window_spread"],
        drawdown_start_day=best_trial.params["drawdown_start_day"],
        drawdown_peak_day=best_trial.params["drawdown_peak_day"],
        drawdown_spread=best_trial.params["drawdown_spread"],
        max_decay_effect=best_trial.params["max_decay_effect"],
        pre_halving_boost=best_trial.params["pre_halving_boost"],
        bull_run_boost=best_trial.params["bull_run_boost"],
        dist_reduction=best_trial.params["dist_reduction"],
    )

    # Get optimization history
    history = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            history.append({
                "trial": trial.number,
                "value": trial.value,
                "params": trial.params,
            })

    result = TuningResult(
        best_params=best_params,
        best_mape=best_trial.value if metric == "mape" else 0.0,
        best_sharpe=-best_trial.value if metric == "sharpe" else 0.0,
        n_trials=len(study.trials),
        study_name=study_name,
        optimization_history=history,
    )

    print("\n" + "-" * 70)
    print("OPTIMIZATION COMPLETE")
    print("-" * 70)
    print(f"  Best {metric}: {best_trial.value:.4f}")
    print(f"  Trials completed: {len(study.trials)}")
    print("\nBest Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    return result


def print_tuning_report(result: TuningResult) -> None:
    """Print a formatted tuning report."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING REPORT")
    print("=" * 70)

    print(f"\nStudy: {result.study_name}")
    print(f"Trials: {result.n_trials}")
    print(f"Best MAPE: {result.best_mape:.2f}%")
    print(f"Best Sharpe: {result.best_sharpe:.2f}")

    print("\n--- BEST PARAMETERS ---")
    params = result.best_params

    print("\nCYCLE Regressor:")
    print(f"  post_halving_peak_day: {params.post_halving_peak_day}")
    print(f"  post_halving_spread: {params.post_halving_spread}")
    print(f"  pre_halving_window: {params.pre_halving_window}")

    print("\nDOUBLE_TOP Regressor:")
    print(f"  first_top_day: {params.first_top_day}")
    print(f"  second_top_day: {params.second_top_day}")
    print(f"  top_window_spread: {params.top_window_spread}")

    print("\nDECAY Regressor:")
    print(f"  drawdown_start_day: {params.drawdown_start_day}")
    print(f"  drawdown_peak_day: {params.drawdown_peak_day}")
    print(f"  drawdown_spread: {params.drawdown_spread}")
    print(f"  max_decay_effect: {params.max_decay_effect:.3f}")

    print("\nENSEMBLE_ADJUST:")
    print(f"  pre_halving_boost: {params.pre_halving_boost:.1%}")
    print(f"  bull_run_boost: {params.bull_run_boost:.1%}")
    print(f"  dist_reduction: {params.dist_reduction:.1%}")

    print("\n" + "=" * 70)


def export_params_to_env(params: RegressorParams, filepath: str = ".env.tuned") -> None:
    """Export tuned parameters to an env file for later use.

    Args:
        params: Tuned parameters.
        filepath: Path to output file.
    """
    lines = [
        "# Tuned regressor parameters",
        "# Generated by regressor_tuning.py",
        "",
        "# CYCLE regressor",
        f"REGRESSOR_POST_HALVING_PEAK_DAY={params.post_halving_peak_day}",
        f"REGRESSOR_POST_HALVING_SPREAD={params.post_halving_spread}",
        f"REGRESSOR_PRE_HALVING_WINDOW={params.pre_halving_window}",
        "",
        "# DOUBLE_TOP regressor",
        f"REGRESSOR_FIRST_TOP_DAY={params.first_top_day}",
        f"REGRESSOR_SECOND_TOP_DAY={params.second_top_day}",
        f"REGRESSOR_TOP_WINDOW_SPREAD={params.top_window_spread}",
        "",
        "# DECAY regressor",
        f"REGRESSOR_DRAWDOWN_START_DAY={params.drawdown_start_day}",
        f"REGRESSOR_DRAWDOWN_PEAK_DAY={params.drawdown_peak_day}",
        f"REGRESSOR_DRAWDOWN_SPREAD={params.drawdown_spread}",
        f"REGRESSOR_MAX_DECAY_EFFECT={params.max_decay_effect}",
        "",
        "# ENSEMBLE_ADJUST",
        f"REGRESSOR_PRE_HALVING_BOOST={params.pre_halving_boost}",
        f"REGRESSOR_BULL_RUN_BOOST={params.bull_run_boost}",
        f"REGRESSOR_DIST_REDUCTION={params.dist_reduction}",
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"Tuned parameters exported to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Run from main script with data loaded")
    print("Example:")
    print("  from src.optimization import tune_regressors")
    print("  result = tune_regressors(df, n_trials=50)")
    print("  export_params_to_env(result.best_params)")
