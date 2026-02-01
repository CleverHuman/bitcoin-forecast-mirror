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


# Define regressor responsibilities and potential conflicts
REGRESSOR_RESPONSIBILITIES = {
    "cycle": "Position in cycle (sin/cos encoding, proximity to halving)",
    "double_top": "Pattern detection (first/second top windows, mid-cycle correction)",
    "cycle_phase": "Phase-specific behavior (bull run, distribution, accumulation slopes)",
    "decay": "Drawdown magnitude scaling (reduces forecast during expected drawdown)",
    "ensemble_adjust": "Timing-based post-processing (pre-halving boost, bull run boost, dist reduction)",
}

# Known potential conflicts between regressors
REGRESSOR_CONFLICTS = [
    {
        "regressors": ["decay", "ensemble_adjust"],
        "issue": "Both reduce forecast during post-365-day period",
        "severity": "warning",
        "recommendation": "Consider disabling one, or ensure magnitudes don't compound excessively",
    },
    {
        "regressors": ["cycle", "cycle_phase"],
        "issue": "Both encode cycle position (potential redundancy)",
        "severity": "info",
        "recommendation": "May cause multicollinearity; cycle_phase offers more granular control",
    },
    {
        "regressors": ["double_top", "cycle_phase"],
        "issue": "Both model post-halving behavior phases",
        "severity": "info",
        "recommendation": "double_top focuses on pattern; cycle_phase on smooth transitions",
    },
]


def validate_regressor_config(
    config: dict[str, bool] | None = None,
    print_warnings: bool = True,
) -> list[dict]:
    """Validate regressor configuration for potential conflicts.

    Checks for:
    1. Conflicting regressors that may double-count effects
    2. Potentially redundant regressors
    3. Missing data requirements

    Args:
        config: Regressor configuration. If None, reads from environment.
        print_warnings: Whether to print warnings to stdout.

    Returns:
        List of warning/info dicts with keys: severity, message, recommendation
    """
    if config is None:
        config = get_regressor_config()

    warnings = []

    # Check for known conflicts
    for conflict in REGRESSOR_CONFLICTS:
        conflict_regs = conflict["regressors"]
        if all(config.get(r, False) for r in conflict_regs):
            warning = {
                "severity": conflict["severity"],
                "regressors": conflict_regs,
                "message": conflict["issue"],
                "recommendation": conflict["recommendation"],
            }
            warnings.append(warning)

            if print_warnings:
                severity_prefix = "WARNING" if conflict["severity"] == "warning" else "INFO"
                print(f"  [{severity_prefix}] {', '.join(conflict_regs).upper()}: {conflict['issue']}")
                print(f"            Recommendation: {conflict['recommendation']}")

    # Check for all-off (pure Prophet)
    if not any(config.values()):
        warning = {
            "severity": "info",
            "regressors": [],
            "message": "All regressors disabled - using pure Prophet forecast",
            "recommendation": "Enable at least 'cycle' regressor for cycle-aware forecasting",
        }
        warnings.append(warning)
        if print_warnings:
            print(f"  [INFO] All regressors disabled - pure Prophet mode")

    # Check for all-on (maximum coupling)
    if all(config.values()):
        warning = {
            "severity": "info",
            "regressors": list(config.keys()),
            "message": "All regressors enabled - maximum cycle awareness but potential over-fitting",
            "recommendation": "Monitor for over-correction; consider ablation testing",
        }
        warnings.append(warning)
        if print_warnings:
            print(f"  [INFO] All regressors enabled - full cycle awareness")

    return warnings


def print_regressor_responsibilities() -> None:
    """Print the defined responsibility for each regressor."""
    print("\n" + "=" * 70)
    print("REGRESSOR RESPONSIBILITIES")
    print("=" * 70)
    for name, responsibility in REGRESSOR_RESPONSIBILITIES.items():
        print(f"\n  {name.upper()}:")
        print(f"    {responsibility}")
    print("\n" + "=" * 70)


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

    # Validate configuration for conflicts
    validate_regressor_config(cfg, print_warnings=True)

    if enable_cycle:
        df = create_cycle_regressors_for_prophet(df, halving_averages=halving_averages)

    if enable_double_top:
        df = create_double_top_regressor(df, halving_averages)

    if enable_cycle_phase:
        df = create_cycle_phase_regressor(df, halving_averages)

    if enable_decay:
        df = create_decay_regressor(df, decay_params, HALVING_DATES, halving_averages=halving_averages)

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
        future = create_cycle_regressors_for_prophet(future, halving_averages=halving_averages)

    if enable_double_top:
        future = create_double_top_regressor(future, halving_averages)

    if enable_cycle_phase:
        future = create_cycle_phase_regressor(future, halving_averages)

    if enable_decay:
        future = create_decay_regressor(future, decay_params, HALVING_DATES, halving_averages=halving_averages)

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
        # DATA-DRIVEN boost/reduction parameters from halving_averages
        # Compute adjustment magnitudes from historical cycle behavior

        # Pre-halving boost: based on historical run-up percentage
        if halving_averages is not None and halving_averages.run_up_pct > 0:
            # Scale: if avg run-up is 300%, use ~15% boost; if 100%, use ~10%
            # This provides proportional response to historical gains
            pre_boost = min(0.20, halving_averages.run_up_pct / 2000)  # Cap at 20%
        else:
            pre_boost = 0.15  # Default 15%

        # Bull run boost: based on historical post-halving gains
        if halving_averages is not None and halving_averages.avg_days_to_top > 0:
            bull_end_day = int(halving_averages.avg_days_to_top)
            bull_start_day = 90  # Consolidation typically ends ~90 days after
            # Scale boost by run-up magnitude
            bull_boost = min(0.18, halving_averages.run_up_pct / 2500)
        else:
            bull_start_day = 120
            bull_end_day = 365
            bull_boost = 0.12

        # Distribution/drawdown reduction: based on predicted drawdown
        if predicted_drawdown > 0:
            # Scale: if expected drawdown is 50%, reduce by ~3%; if 30%, reduce by ~2%
            # This is a gentle nudge since Prophet's decay_regressor handles most
            dist_reduction = min(0.08, predicted_drawdown / 10)
        else:
            dist_reduction = 0.03

        # Apply adjustments

        # Pre-halving run-up boost
        pre_mask = forecast["pre_halving_weight"] > 0
        adjustment[pre_mask] += forecast.loc[pre_mask, "pre_halving_weight"] * pre_boost

        # Post-halving zones
        days_since = forecast["days_since_halving"].fillna(0)

        # Bull run boost: smooth ramp using Gaussian weighting
        bull_mask = (days_since > bull_start_day) & (days_since <= bull_end_day)
        if bull_mask.any():
            peak_day = (bull_start_day + bull_end_day) / 2
            spread = (bull_end_day - bull_start_day) / 3
            bull_weight = np.exp(-((days_since[bull_mask] - peak_day) ** 2) / (2 * spread ** 2))
            adjustment[bull_mask] += bull_weight * bull_boost

        # Distribution/drawdown reduction (smooth ramp down)
        dist_start = bull_end_day
        dist_mask = days_since > dist_start
        if dist_mask.any():
            # Gradual increase in reduction, capped
            days_into_dist = (days_since[dist_mask] - dist_start).values
            reduction_factor = np.minimum(days_into_dist / 200, 1.0)  # Ramps to full over 200 days
            adjustment[dist_mask] -= reduction_factor * dist_reduction

        print(f"  Ensemble adjustment: enabled (pre_boost={pre_boost:.1%}, "
              f"bull_boost={bull_boost:.1%}, dist_reduction={dist_reduction:.1%})")
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
