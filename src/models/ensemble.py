"""Ensemble model combining Prophet trend with cycle-aware ML.

Prophet captures trend and seasonality, while a gradient boosting model
learns cycle-specific patterns that Prophet misses.
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet

from .cycle_features import add_cycle_features, create_cycle_regressors_for_prophet
from src.metrics import (
    HALVING_DATES,
    HalvingAverages,
    create_decay_regressor,
    create_double_top_regressor,
    create_cycle_phase_regressor,
    fit_decay_curve,
)


def _env_bool(key: str, default: bool = True) -> bool:
    """Read a boolean from environment variable (at call time, not import time)."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def get_regressor_config() -> dict[str, bool]:
    """Get regressor toggles from environment at call time."""
    return {
        "cycle": _env_bool("REGRESSOR_CYCLE", True),
        "double_top": _env_bool("REGRESSOR_DOUBLE_TOP", True),
        "cycle_phase": _env_bool("REGRESSOR_CYCLE_PHASE", True),
        "decay": _env_bool("REGRESSOR_DECAY", True),
        "ensemble_adjust": _env_bool("REGRESSOR_ENSEMBLE_ADJUST", True),
    }


def train_prophet_with_regressors(
    df: pd.DataFrame,
    periods: int = 240,
    use_cycle_regressors: bool = True,
    halving_averages: HalvingAverages | None = None,
    decay_params: tuple[float, float, float] | None = None,
) -> tuple[Prophet, pd.DataFrame]:
    """Train Prophet with cycle-aware regressors.

    Unlike standard Prophet holidays (point effects), regressors allow
    continuous influence based on cycle position.

    Regressors can be disabled via environment variables:
    - REGRESSOR_CYCLE: sin/cos cycle position, pre/post halving weights
    - REGRESSOR_DOUBLE_TOP: double-top pattern detection
    - REGRESSOR_CYCLE_PHASE: continuous cycle phase encoding
    - REGRESSOR_DECAY: drawdown decay curve adjustment

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        periods: Days to forecast.
        use_cycle_regressors: Whether to add cycle position regressors.
        halving_averages: HalvingAverages with double-top timing data.
        decay_params: Tuple of (a, b, c) from fit_decay_curve for drawdown decay.

    Returns:
        Tuple of (trained model, forecast DataFrame).
    """
    df = df.copy()

    # Get env toggles at call time (after load_dotenv)
    cfg = get_regressor_config()

    # Check env toggles combined with data availability
    enable_cycle = use_cycle_regressors and cfg["cycle"]
    enable_double_top = (
        cfg["double_top"]
        and halving_averages is not None
        and halving_averages.double_top_frequency > 0
    )
    enable_cycle_phase = (
        cfg["cycle_phase"]
        and halving_averages is not None
    )
    enable_decay = cfg["decay"] and decay_params is not None

    # Log which regressors are active
    print(f"  Regressors: cycle={enable_cycle}, double_top={enable_double_top}, "
          f"cycle_phase={enable_cycle_phase}, decay={enable_decay}")

    if enable_cycle:
        df = create_cycle_regressors_for_prophet(df)

    if enable_double_top:
        df = create_double_top_regressor(df, halving_averages)

    if enable_cycle_phase:
        df = create_cycle_phase_regressor(df, halving_averages)

    if enable_decay:
        df = create_decay_regressor(df, decay_params, HALVING_DATES)

    model = Prophet(
        interval_width=0.95,
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        changepoint_range=0.8,
        seasonality_prior_scale=10,
        n_changepoints=300,
    )

    # Add custom 4-year seasonality for halving cycle
    model.add_seasonality(
        name="halving_cycle",
        period=365.25 * 4,  # 4-year cycle
        fourier_order=3,  # Low order to avoid overfitting
    )

    if enable_cycle:
        model.add_regressor("reg_cycle_sin", mode="multiplicative")
        model.add_regressor("reg_cycle_cos", mode="multiplicative")
        model.add_regressor("reg_pre_halving", mode="multiplicative")
        model.add_regressor("reg_post_halving", mode="multiplicative")

    if enable_double_top:
        model.add_regressor("double_top_regressor", mode="multiplicative")

    if enable_cycle_phase:
        model.add_regressor("cycle_phase_regressor", mode="multiplicative")

    if enable_decay:
        model.add_regressor("decay_regressor", mode="multiplicative")

    model.fit(df)

    # Create future dataframe with regressors
    future = model.make_future_dataframe(periods=periods, freq="d")

    if enable_cycle:
        future = create_cycle_regressors_for_prophet(future)

    if enable_double_top:
        future = create_double_top_regressor(future, halving_averages)

    if enable_cycle_phase:
        future = create_cycle_phase_regressor(future, halving_averages)

    if enable_decay:
        future = create_decay_regressor(future, decay_params, HALVING_DATES)

    forecast = model.predict(future)

    return model, forecast


def train_simple_ensemble(
    df: pd.DataFrame,
    periods: int = 240,
    prophet_weight: float = 0.7,
    cycle_weight: float = 0.3,
    halving_averages: HalvingAverages | None = None,
    cycle_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simple ensemble: Prophet prediction adjusted by cycle position.

    Applies a multiplier to Prophet's forecast based on historical
    cycle patterns (run-up before halving, drawdown after).

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        periods: Days to forecast.
        prophet_weight: Weight for base Prophet forecast.
        cycle_weight: Weight for cycle adjustment.
        halving_averages: HalvingAverages with double-top timing for regressors.
        cycle_metrics: DataFrame from compute_cycle_metrics for decay fitting.

    Returns:
        Forecast DataFrame with 'yhat_ensemble' column.
    """
    # Fit decay curve if we have cycle metrics
    decay_params = None
    predicted_drawdown = 0.1  # Default fallback

    if cycle_metrics is not None and len(cycle_metrics) >= 2:
        cycle_nums = list(range(1, len(cycle_metrics) + 1))
        drawdowns = (cycle_metrics["drawdown_pct"] / 100).tolist()
        decay_params, r_sq = fit_decay_curve(cycle_nums, drawdowns)

        # Predict drawdown for next cycle
        from src.metrics import exp_decay
        next_cycle = len(cycle_metrics) + 1
        predicted_drawdown = float(exp_decay(next_cycle, *decay_params))
        print(f"  Decay-predicted drawdown for cycle {next_cycle}: {predicted_drawdown:.1%}")

    # Train Prophet with decay regressor
    model, forecast = train_prophet_with_regressors(
        df, periods,
        use_cycle_regressors=True,
        halving_averages=halving_averages,
        decay_params=decay_params,
    )

    # Add cycle features to forecast
    forecast = add_cycle_features(forecast, date_col="ds")

    # Get config to check if ensemble adjustments are enabled
    cfg = get_regressor_config()

    # Compute cycle adjustment multiplier (can be disabled via REGRESSOR_ENSEMBLE_ADJUST=false)
    adjustment = np.ones(len(forecast))

    if cfg["ensemble_adjust"]:
        # Pre-halving: boost forecast (run-up expected)
        # Post-halving peak zone: reduce forecast (distribution)
        # Drawdown: reduce forecast using decay-predicted severity

        # Pre-halving run-up boost (up to +20%)
        pre_mask = forecast["pre_halving_weight"] > 0
        adjustment[pre_mask] += forecast.loc[pre_mask, "pre_halving_weight"] * 0.2

        # Post-halving zones
        days_since = forecast["days_since_halving"].fillna(0)

        # Bull run boost (120-365 days after)
        bull_mask = (days_since > 120) & (days_since <= 365)
        bull_factor = 1 - (days_since[bull_mask] - 120) / 245  # Fades from 1 to 0
        adjustment[bull_mask] += bull_factor * 0.15

        # Distribution/drawdown reduction
        # Note: Prophet's decay_regressor handles most of the drawdown adjustment
        # This is just a small additional nudge for the ensemble weighting
        dist_mask = days_since > 365
        adjustment[dist_mask] -= 0.05  # Small fixed reduction (Prophet handles the rest)

        print(f"  Ensemble adjustment: enabled")
    else:
        print(f"  Ensemble adjustment: disabled (using raw Prophet output)")

    # Apply weighted adjustment
    forecast["cycle_adjustment"] = adjustment
    forecast["yhat_ensemble"] = (
        forecast["yhat"] * prophet_weight
        + forecast["yhat"] * adjustment * cycle_weight
    )

    # Adjust confidence intervals proportionally
    ratio = forecast["yhat_ensemble"] / forecast["yhat"].replace(0, 1)
    forecast["yhat_ensemble_lower"] = forecast["yhat_lower"] * ratio
    forecast["yhat_ensemble_upper"] = forecast["yhat_upper"] * ratio

    return forecast


def evaluate_forecast(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    holdout_days: int = 90,
) -> dict[str, Any]:
    """Evaluate forecast accuracy on holdout period.

    Args:
        df: Original data with 'ds' and 'y'.
        forecast: Forecast DataFrame with 'ds' and 'yhat'/'yhat_ensemble'.
        holdout_days: Number of recent days to use as holdout.

    Returns:
        Dict with error metrics.
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    forecast = forecast.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    # Get holdout period
    cutoff = df["ds"].max() - pd.Timedelta(days=holdout_days)
    holdout = df[df["ds"] > cutoff].copy()

    if holdout.empty:
        return {"error": "No holdout data"}

    # Merge with forecast
    merged = holdout.merge(forecast[["ds", "yhat", "yhat_ensemble"]], on="ds", how="left")
    merged = merged.dropna()

    if merged.empty:
        return {"error": "No overlapping data"}

    # Compute metrics
    def compute_metrics(actual, predicted, prefix=""):
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        return {
            f"{prefix}mae": mae,
            f"{prefix}mape": mape,
            f"{prefix}rmse": rmse,
        }

    results = {
        "holdout_days": holdout_days,
        "n_samples": len(merged),
    }

    if "yhat" in merged.columns:
        results.update(compute_metrics(merged["y"], merged["yhat"], "prophet_"))

    if "yhat_ensemble" in merged.columns:
        results.update(compute_metrics(merged["y"], merged["yhat_ensemble"], "ensemble_"))

    # Compare
    if "prophet_mape" in results and "ensemble_mape" in results:
        results["ensemble_improvement_pct"] = (
            (results["prophet_mape"] - results["ensemble_mape"]) / results["prophet_mape"] * 100
        )

    return results
