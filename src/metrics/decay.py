"""Exponential decay fitting for drawdown/volatility prediction.

Fits decay curves to historical cycle data to predict future drawdowns,
accounting for market maturation (decreasing volatility over time).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class DecayPrediction:
    """Result of decay curve prediction."""

    predicted_value: float
    floor: float  # asymptotic minimum (c parameter)
    decay_rate: float  # how fast it decays (b parameter)
    r_squared: float  # fit quality
    confidence_low: float
    confidence_high: float


def exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential decay: y = a * exp(-b * x) + c

    Args:
        x: Cycle number(s)
        a: Initial amplitude above floor
        b: Decay rate (higher = faster decay)
        c: Asymptotic floor (minimum value as x → infinity)

    Returns:
        Predicted value(s)
    """
    return a * np.exp(-b * x) + c


def fit_decay_curve(
    cycle_nums: list[int],
    values: list[float],
    floor_bounds: tuple[float, float] = (0.2, 0.5),
) -> tuple[np.ndarray, float]:
    """Fit exponential decay to historical values using curve_fit.

    Args:
        cycle_nums: Cycle numbers (1, 2, 3, ...)
        values: Corresponding values (e.g., drawdown percentages as decimals)
        floor_bounds: Min/max bounds for asymptotic floor (c parameter)

    Returns:
        Tuple of (params [a, b, c], r_squared)
    """
    x = np.array(cycle_nums, dtype=float)
    y = np.array(values, dtype=float)

    # Need at least 3 points to fit 3 parameters
    if len(x) < 3:
        avg = float(np.mean(y))
        return np.array([0.0, 0.0, avg]), 0.0

    # Initial guess based on data
    p0 = [y[0] - floor_bounds[0], 0.1, floor_bounds[0]]

    # Bounds: a in [0,1], b in [0,1], c in floor_bounds
    bounds = ([0, 0, floor_bounds[0]], [1, 1, floor_bounds[1]])

    try:
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(500)  # Limit recursion for curve_fit
        try:
            params, _ = curve_fit(
                exp_decay, x, y,
                p0=p0,
                bounds=bounds,
                maxfev=500,
                method='trf'  # Trust Region Reflective - more stable
            )
        finally:
            sys.setrecursionlimit(old_limit)
    except Exception:
        # Fallback: simple linear trend estimation
        avg = float(np.mean(y))
        if len(y) >= 2 and y[0] > y[-1]:
            # Decreasing trend - estimate decay
            decay_per_cycle = (y[0] - y[-1]) / (len(y) - 1)
            a = y[0] - floor_bounds[0]
            b = decay_per_cycle / a if a > 0 else 0.1
            c = floor_bounds[0]
            params = np.array([max(0, a), max(0, min(b, 0.5)), c])
        else:
            params = np.array([0.0, 0.0, avg])

    # Calculate R²
    predicted = exp_decay(x, *params)
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return params, float(r_squared)


def predict_drawdown(
    cycle_metrics: pd.DataFrame,
    target_cycle: int | None = None,
    floor_bounds: tuple[float, float] = (0.2, 0.5),
) -> DecayPrediction:
    """Predict drawdown for a future cycle using decay curve.

    Args:
        cycle_metrics: DataFrame from compute_cycle_metrics() with drawdown_pct column
        target_cycle: Cycle number to predict (default: next cycle)
        floor_bounds: Min/max bounds for asymptotic floor

    Returns:
        DecayPrediction with predicted value and confidence interval
    """
    n_cycles = len(cycle_metrics)

    if target_cycle is None:
        target_cycle = n_cycles + 1

    cycle_nums = list(range(1, n_cycles + 1))
    drawdowns = (cycle_metrics["drawdown_pct"] / 100).tolist()

    params, r_sq = fit_decay_curve(cycle_nums, drawdowns, floor_bounds)
    predicted = float(exp_decay(target_cycle, *params))

    # Confidence interval from residual std error
    residuals = np.array(drawdowns) - exp_decay(np.array(cycle_nums), *params)
    std_err = float(np.std(residuals)) if len(residuals) > 1 else 0.1

    return DecayPrediction(
        predicted_value=predicted,
        floor=params[2],
        decay_rate=params[1],
        r_squared=r_sq,
        confidence_low=max(predicted - 2 * std_err, params[2]),
        confidence_high=min(predicted + 2 * std_err, 1.0),
    )


def predict_volatility(
    daily_returns: pd.Series,
    cycle_boundaries: list[pd.Timestamp],
    target_cycle: int | None = None,
) -> DecayPrediction:
    """Predict volatility for a future cycle using decay curve.

    Args:
        daily_returns: Series of daily log returns
        cycle_boundaries: List of cycle start dates (halving dates)
        target_cycle: Cycle number to predict (default: next cycle)

    Returns:
        DecayPrediction with predicted volatility
    """
    vols = []

    for i in range(len(cycle_boundaries) - 1):
        start = cycle_boundaries[i]
        end = cycle_boundaries[i + 1]
        cycle_returns = daily_returns[(daily_returns.index >= start) & (daily_returns.index < end)]
        if len(cycle_returns) > 30:
            vol = float(cycle_returns.std() * np.sqrt(365))  # Annualized
            vols.append(vol)

    if len(vols) < 2:
        return DecayPrediction(
            predicted_value=0.5,
            floor=0.3,
            decay_rate=0,
            r_squared=0,
            confidence_low=0.3,
            confidence_high=0.7,
        )

    n_cycles = len(vols)
    if target_cycle is None:
        target_cycle = n_cycles + 1

    cycle_nums = list(range(1, n_cycles + 1))

    # Volatility floor bounds: 20-50% annualized
    params, r_sq = fit_decay_curve(cycle_nums, vols, floor_bounds=(0.2, 0.5))
    predicted = float(exp_decay(target_cycle, *params))

    residuals = np.array(vols) - exp_decay(np.array(cycle_nums), *params)
    std_err = float(np.std(residuals)) if len(residuals) > 1 else 0.1

    return DecayPrediction(
        predicted_value=predicted,
        floor=params[2],
        decay_rate=params[1],
        r_squared=r_sq,
        confidence_low=max(predicted - 2 * std_err, params[2]),
        confidence_high=predicted + 2 * std_err,
    )


def print_decay_prediction(pred: DecayPrediction, metric_name: str = "Drawdown") -> None:
    """Print decay prediction summary."""
    print(f"\n{metric_name} Decay Prediction")
    print("-" * 40)
    print(f"  Predicted:     {pred.predicted_value:.1%}")
    print(f"  Range (95%):   {pred.confidence_low:.1%} - {pred.confidence_high:.1%}")
    print(f"  Floor:         {pred.floor:.1%}")
    print(f"  Decay rate:    {pred.decay_rate:.3f}")
    print(f"  R² fit:        {pred.r_squared:.3f}")


def create_decay_regressor(
    df: pd.DataFrame,
    decay_params: tuple[float, float, float],
    halving_dates: pd.DatetimeIndex,
    date_col: str = "ds",
) -> pd.DataFrame:
    """Create a regressor encoding expected drawdown magnitude for Prophet.

    The regressor value represents the expected drawdown severity at each date,
    based on the calibrated decay curve. Prophet learns how strongly to weight
    this signal.

    Values:
    - 0.0: At or before halving (no drawdown expected yet)
    - 0.0 to 1.0: During post-halving period, scaled by expected drawdown
    - Higher values = more severe expected drawdown

    Args:
        df: DataFrame with date column.
        decay_params: Tuple of (a, b, c) from fit_decay_curve.
        halving_dates: Halving dates to use.
        date_col: Name of date column.

    Returns:
        DataFrame with 'decay_regressor' column added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["decay_regressor"] = 0.0

    a, b, c = decay_params

    for i, h in enumerate(halving_dates):
        cycle_num = i + 1

        # Expected drawdown for this cycle
        expected_drawdown = float(exp_decay(cycle_num, a, b, c))

        # Find next halving for window end
        if i + 1 < len(halving_dates):
            next_h = halving_dates[i + 1]
        else:
            next_h = h + pd.Timedelta(days=1461)  # ~4 years

        # Post-halving mask (where drawdown occurs)
        post_mask = (df[date_col] >= h) & (df[date_col] < next_h)

        if not post_mask.any():
            continue

        # Calculate days since halving for timing weight
        days_since = (df.loc[post_mask, date_col] - h).dt.days

        # Drawdown typically peaks 300-500 days after halving
        # Use a bell curve centered around day 400
        peak_day = 400
        spread = 200
        timing_weight = np.exp(-((days_since - peak_day) ** 2) / (2 * spread ** 2))

        # Scale for multiplicative mode: values should be small (-0.1 to 0.1)
        # Negative = reduces forecast during drawdown period
        # Scale expected_drawdown (0.3-0.8) to regressor range (-0.1 to -0.02)
        regressor_scale = 0.15  # Max effect on forecast

        # Combine: negative value scaled by drawdown magnitude and timing
        # Higher drawdown expectation = more negative regressor
        regressor_value = -regressor_scale * expected_drawdown * timing_weight
        df.loc[post_mask, "decay_regressor"] = regressor_value

    return df


def create_decay_adjustment(
    df: pd.DataFrame,
    decay_params: tuple[float, float, float],
    halving_dates: pd.DatetimeIndex,
    date_col: str = "ds",
    price_col: str = "yhat",
) -> pd.DataFrame:
    """Apply decay-based adjustment to price forecasts.

    Adjusts forecast prices downward during expected drawdown periods,
    scaled by the predicted drawdown severity.

    Args:
        df: DataFrame with date and price columns.
        decay_params: Tuple of (a, b, c) from fit_decay_curve.
        halving_dates: Halving dates to use.
        date_col: Name of date column.
        price_col: Name of price column to adjust.

    Returns:
        DataFrame with 'decay_adjusted_price' column added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # First add the regressor
    df = create_decay_regressor(df, decay_params, halving_dates, date_col)

    # Apply adjustment: reduce price by (regressor * expected_drawdown)
    # decay_regressor already encodes expected_drawdown * timing
    adjustment_factor = 1 - df["decay_regressor"]
    df["decay_adjusted_price"] = df[price_col] * adjustment_factor

    return df
