"""
Data-driven halving cycle metrics from actual BTC prices.

Measures run-up (low → pre-halving high), drawdown (high → post-halving low),
and durations per cycle; averages across cycles; and exposes results for
parameterising Prophet or sanity-checking forecasts.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Canonical halving dates (align with forecast_cycle / Prophet holidays)
HALVING_DATES = pd.to_datetime(
    [
        "2012-11-28",
        "2016-07-09",
        "2020-05-11",
        "2024-04-19",
        "2028-04-11",
    ]
)


@dataclass
class DoubleTopInfo:
    """Information about a double-top pattern in a cycle."""

    is_double_top: bool = False
    first_top_date: pd.Timestamp | None = None
    first_top_price: float = 0.0
    second_top_date: pd.Timestamp | None = None
    second_top_price: float = 0.0
    mid_cycle_low_date: pd.Timestamp | None = None
    mid_cycle_low_price: float = 0.0
    mid_cycle_drawdown_pct: float = 0.0  # Drawdown between first and second top
    days_halving_to_first_top: int = 0
    days_halving_to_second_top: int = 0
    days_between_tops: int = 0


@dataclass
class HalvingAverages:
    """Averaged metrics across completed halving cycles."""

    run_up_pct: float
    run_up_days: float
    drawdown_pct: float
    drawdown_days: float
    n_cycles: int
    # Timing predictions (days relative to halving)
    avg_days_to_top: float = 0.0      # Days AFTER halving to cycle peak
    avg_days_to_bottom: float = 0.0   # Days AFTER halving to cycle bottom
    avg_days_before_low: float = 0.0  # Days BEFORE halving to accumulation low
    # Double-top metrics
    double_top_frequency: float = 0.0  # Fraction of cycles with double tops
    avg_days_to_first_top: float = 0.0
    avg_days_to_second_top: float = 0.0
    avg_days_between_tops: float = 0.0
    avg_mid_cycle_drawdown_pct: float = 0.0


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with ds sorted ascending, one row per day."""
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"])
    out = out.groupby("ds", as_index=False)["y"].mean()
    return out.sort_values("ds").reset_index(drop=True)


def detect_double_top(
    df: pd.DataFrame,
    halving_date: pd.Timestamp,
    min_drawdown_pct: float = 25.0,
    min_days_between_peaks: int = 60,
    lookback_window_days: int = 800,
) -> DoubleTopInfo:
    """Detect double-top pattern in post-halving price action.

    A double top is identified when:
    1. There's a significant peak followed by a drawdown of at least min_drawdown_pct
    2. Price recovers to form a second peak (can be higher or lower than first)
    3. The peaks are separated by at least min_days_between_peaks

    Args:
        df: DataFrame with 'ds' (date) and 'y' (price) columns, sorted ascending.
        halving_date: The halving date to analyze.
        min_drawdown_pct: Minimum drawdown between peaks to qualify (default 25%).
        min_days_between_peaks: Minimum days between first and second peak.
        lookback_window_days: How far after halving to look for the pattern.

    Returns:
        DoubleTopInfo with detected pattern details.
    """
    result = DoubleTopInfo()

    # Get post-halving data within window
    end_date = halving_date + pd.Timedelta(days=lookback_window_days)
    post = df[(df["ds"] >= halving_date) & (df["ds"] <= end_date)].copy()

    if len(post) < min_days_between_peaks * 2:
        return result

    prices = post["y"].values
    dates = post["ds"].values

    # Find local maxima using rolling window
    window = 30  # 30-day rolling window for smoothing
    post["rolling_max"] = post["y"].rolling(window=window, center=True).max()
    post["is_peak"] = (post["y"] == post["rolling_max"]) & (post["y"] > post["y"].shift(1)) & (post["y"] > post["y"].shift(-1))

    # Get significant peaks (top 10% of prices in the window)
    price_threshold = np.percentile(prices, 90)
    significant_peaks = post[(post["is_peak"]) & (post["y"] >= price_threshold)].copy()

    if len(significant_peaks) < 2:
        return result

    # Find first major peak
    first_peak_idx = significant_peaks["y"].idxmax()
    first_peak = post.loc[first_peak_idx]

    # Look for drawdown after first peak
    after_first = post[post["ds"] > first_peak["ds"]]
    if after_first.empty:
        return result

    # Find the lowest point after first peak
    trough_idx = after_first["y"].idxmin()
    trough = post.loc[trough_idx]

    # Calculate drawdown from first peak
    drawdown_pct = (1 - trough["y"] / first_peak["y"]) * 100

    if drawdown_pct < min_drawdown_pct:
        return result

    # Look for second peak after trough
    after_trough = post[post["ds"] > trough["ds"]]
    if after_trough.empty:
        return result

    # Find significant peaks after trough
    second_peaks = after_trough[after_trough["y"] >= trough["y"] * 1.2]  # At least 20% recovery
    if second_peaks.empty:
        return result

    second_peak_idx = second_peaks["y"].idxmax()
    second_peak = post.loc[second_peak_idx]

    # Check minimum days between peaks
    days_between = (second_peak["ds"] - first_peak["ds"]).days
    if days_between < min_days_between_peaks:
        return result

    # We have a valid double top
    result.is_double_top = True
    result.first_top_date = first_peak["ds"]
    result.first_top_price = float(first_peak["y"])
    result.second_top_date = second_peak["ds"]
    result.second_top_price = float(second_peak["y"])
    result.mid_cycle_low_date = trough["ds"]
    result.mid_cycle_low_price = float(trough["y"])
    result.mid_cycle_drawdown_pct = drawdown_pct
    result.days_halving_to_first_top = (first_peak["ds"] - halving_date).days
    result.days_halving_to_second_top = (second_peak["ds"] - halving_date).days
    result.days_between_tops = days_between

    return result


def compute_cycle_metrics(
    df: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "y",
    halving_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Compute run-up, drawdown, and duration for each halving cycle.

    For each halving H_i (with previous H_{i-1} and next H_{i+1}):

    - Run-up before halving: low → pre-halving high in [H_{i-1}, H_i].
      Low = argmin(price), high = argmax(price) after low. Measures % gain
      and days from low to high.
    - Drawdown after halving: high → post-halving low in [H_i, H_{i+1}].
      High = argmax(price), low = argmin(price) after high. Measures % drop
      and days from high to low.

    Args:
        df: DataFrame with date and price columns.
        date_col: Name of date column.
        price_col: Name of price column.
        halving_dates: Halving dates to use; defaults to HALVING_DATES.

    Returns:
        DataFrame with one row per halving cycle and columns:
        halving_date, run_up_pct, run_up_days, drawdown_pct, drawdown_days,
        pre_low_date, pre_high_date, post_high_date, post_low_date, etc.
    """
    if halving_dates is None:
        halving_dates = HALVING_DATES

    out = _ensure_sorted(df.rename(columns={date_col: "ds", price_col: "y"}))
    halvings = pd.to_datetime(halving_dates)

    rows = []

    for i in range(1, len(halvings) - 1):
        h_prev, h, h_next = halvings[i - 1], halvings[i], halvings[i + 1]

        pre = out[(out["ds"] >= h_prev) & (out["ds"] < h)]
        # Buffer excludes pre-halving run-up period before next halving
        buffer_days = 90
        post_end = h_next - pd.Timedelta(days=buffer_days)
        post = out[(out["ds"] >= h) & (out["ds"] < post_end)]

        if pre.empty or post.empty:
            continue

        # Run-up: low → pre-halving high (high must be after low)
        low_idx = pre["y"].idxmin()
        low_row = pre.loc[low_idx]
        pre_after_low = pre[pre["ds"] > low_row["ds"]]
        if pre_after_low.empty:
            continue
        high_idx = pre_after_low["y"].idxmax()
        high_row = pre_after_low.loc[high_idx]

        run_up_pct = (float(high_row["y"]) / float(low_row["y"]) - 1) * 100
        run_up_days = (high_row["ds"] - low_row["ds"]).days

        # Drawdown: post-halving high → post-halving low (low must be after high)
        post_high_idx = post["y"].idxmax()
        post_high_row = post.loc[post_high_idx]
        post_after_high = post[post["ds"] > post_high_row["ds"]]
        if post_after_high.empty:
            continue
        post_low_idx = post_after_high["y"].idxmin()
        post_low_row = post_after_high.loc[post_low_idx]

        drawdown_pct = (1 - float(post_low_row["y"]) / float(post_high_row["y"])) * 100
        drawdown_days = (post_low_row["ds"] - post_high_row["ds"]).days

        # Days relative to halving for timing predictions
        days_after_halving_to_high = (post_high_row["ds"] - h).days
        days_after_halving_to_low = (post_low_row["ds"] - h).days
        days_before_halving_to_low = (h - low_row["ds"]).days

        # Detect double-top pattern
        double_top = detect_double_top(out, h)

        rows.append(
            {
                "halving_date": h,
                "run_up_pct": run_up_pct,
                "run_up_days": run_up_days,
                "drawdown_pct": drawdown_pct,
                "drawdown_days": drawdown_days,
                "pre_low_date": low_row["ds"],
                "pre_low_price": low_row["y"],
                "pre_high_date": high_row["ds"],
                "pre_high_price": high_row["y"],
                "post_high_date": post_high_row["ds"],
                "post_high_price": post_high_row["y"],
                "post_low_date": post_low_row["ds"],
                "post_low_price": post_low_row["y"],
                # Timing metrics for predictions
                "days_after_halving_to_high": days_after_halving_to_high,
                "days_after_halving_to_low": days_after_halving_to_low,
                "days_before_halving_to_low": days_before_halving_to_low,
                # Double-top metrics
                "is_double_top": double_top.is_double_top,
                "first_top_date": double_top.first_top_date,
                "first_top_price": double_top.first_top_price,
                "second_top_date": double_top.second_top_date,
                "second_top_price": double_top.second_top_price,
                "mid_cycle_low_date": double_top.mid_cycle_low_date,
                "mid_cycle_low_price": double_top.mid_cycle_low_price,
                "mid_cycle_drawdown_pct": double_top.mid_cycle_drawdown_pct,
                "days_to_first_top": double_top.days_halving_to_first_top,
                "days_to_second_top": double_top.days_halving_to_second_top,
                "days_between_tops": double_top.days_between_tops,
            }
        )

    return pd.DataFrame(rows)


def compute_halving_averages(
    df: pd.DataFrame | None = None,
    cycle_metrics: pd.DataFrame | None = None,
) -> HalvingAverages:
    """Average run-up, drawdown, and durations across halving cycles.

    Either pass the raw price DataFrame `df`, or an already-computed
    `cycle_metrics` from compute_cycle_metrics().

    Returns:
        HalvingAverages with mean run_up_pct, run_up_days, drawdown_pct,
        drawdown_days and n_cycles.
    """
    if cycle_metrics is None:
        if df is None:
            raise ValueError("Provide either df or cycle_metrics")
        cycle_metrics = compute_cycle_metrics(df)

    if cycle_metrics.empty:
        return HalvingAverages(
            run_up_pct=0.0,
            run_up_days=0.0,
            drawdown_pct=0.0,
            drawdown_days=0.0,
            n_cycles=0,
        )

    # Safe mean helper
    def safe_mean(col):
        return float(cycle_metrics[col].mean()) if col in cycle_metrics.columns else 0.0

    # Double-top specific averages (only from cycles with double tops)
    double_top_cycles = cycle_metrics[cycle_metrics["is_double_top"] == True]
    double_top_freq = len(double_top_cycles) / len(cycle_metrics) if len(cycle_metrics) > 0 else 0

    def dt_mean(col):
        if double_top_cycles.empty or col not in double_top_cycles.columns:
            return 0.0
        return float(double_top_cycles[col].mean())

    return HalvingAverages(
        run_up_pct=safe_mean("run_up_pct"),
        run_up_days=safe_mean("run_up_days"),
        drawdown_pct=safe_mean("drawdown_pct"),
        drawdown_days=safe_mean("drawdown_days"),
        n_cycles=len(cycle_metrics),
        avg_days_to_top=safe_mean("days_after_halving_to_high"),
        avg_days_to_bottom=safe_mean("days_after_halving_to_low"),
        avg_days_before_low=safe_mean("days_before_halving_to_low"),
        # Double-top metrics
        double_top_frequency=double_top_freq,
        avg_days_to_first_top=dt_mean("days_to_first_top"),
        avg_days_to_second_top=dt_mean("days_to_second_top"),
        avg_days_between_tops=dt_mean("days_between_tops"),
        avg_mid_cycle_drawdown_pct=dt_mean("mid_cycle_drawdown_pct"),
    )


def get_prophet_params_from_halving(
    averages: HalvingAverages,
) -> dict[str, Any]:
    """Suggest Prophet parameters from halving-cycle averages.

    Uses average run-up/drawdown durations to inform changepoint_range
    and windowing (e.g. how much history to treat as "recent" around
    halving events).

    Returns:
        Dict of parameter names to suggested values for use in Prophet().
    """
    if averages.n_cycles == 0:
        return {}

    # Typical cycle phase length in days (run-up + drawdown) as fraction of ~4y
    avg_phase_days = averages.run_up_days + averages.drawdown_days
    four_years = 365.25 * 4
    # changepoint_range: allow changepoints in this fraction of history
    # Longer run-up/drawdown suggests allowing changepoints over more of history
    suggested_range = min(0.95, max(0.5, avg_phase_days / four_years))

    return {
        "changepoint_range": round(suggested_range, 2),
        "halving_run_up_days_avg": round(averages.run_up_days, 0),
        "halving_drawdown_days_avg": round(averages.drawdown_days, 0),
        "halving_run_up_pct_avg": round(averages.run_up_pct, 1),
        "halving_drawdown_pct_avg": round(averages.drawdown_pct, 1),
    }


def create_double_top_regressor(
    df: pd.DataFrame,
    averages: HalvingAverages,
    date_col: str = "ds",
    halving_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Create a regressor column for Prophet modeling double-top cycle patterns.

    The regressor encodes the expected position within the double-top cycle:
    - 0.0: Outside double-top window
    - 0.5: In first-top window (bullish)
    - -0.5: In mid-cycle correction window (bearish)
    - 0.5: In second-top window (bullish)

    Args:
        df: DataFrame with date column.
        averages: HalvingAverages with double-top timing data.
        date_col: Name of date column.
        halving_dates: Halving dates to use; defaults to HALVING_DATES.

    Returns:
        DataFrame with 'double_top_regressor' column added.
    """
    if halving_dates is None:
        halving_dates = HALVING_DATES

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["double_top_regressor"] = 0.0

    if averages.double_top_frequency == 0 or averages.avg_days_to_first_top == 0:
        return df

    # Define windows based on historical averages
    first_top_start = int(averages.avg_days_to_first_top - 60)
    first_top_end = int(averages.avg_days_to_first_top + 30)

    # Mid-cycle correction (between tops)
    mid_correction_start = first_top_end
    mid_correction_end = int(averages.avg_days_to_second_top - 60)

    # Second top window
    second_top_start = mid_correction_end
    second_top_end = int(averages.avg_days_to_second_top + 60)

    for h in halving_dates:
        # First top window (bullish)
        first_start = h + pd.Timedelta(days=first_top_start)
        first_end = h + pd.Timedelta(days=first_top_end)
        mask = (df[date_col] >= first_start) & (df[date_col] <= first_end)
        df.loc[mask, "double_top_regressor"] = 0.5

        # Mid-cycle correction (bearish)
        mid_start = h + pd.Timedelta(days=mid_correction_start)
        mid_end = h + pd.Timedelta(days=mid_correction_end)
        mask = (df[date_col] >= mid_start) & (df[date_col] <= mid_end)
        df.loc[mask, "double_top_regressor"] = -0.5

        # Second top window (bullish)
        sec_start = h + pd.Timedelta(days=second_top_start)
        sec_end = h + pd.Timedelta(days=second_top_end)
        mask = (df[date_col] >= sec_start) & (df[date_col] <= sec_end)
        df.loc[mask, "double_top_regressor"] = 0.5

    return df


def create_cycle_phase_regressor(
    df: pd.DataFrame,
    averages: HalvingAverages,
    date_col: str = "ds",
    halving_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Create a continuous regressor encoding cycle phase for Prophet.

    Encodes position within the halving cycle as a value from -1 to 1:
    - Accumulation (bear market): -1 to -0.5
    - Pre-halving run-up: -0.5 to 0
    - Halving: 0
    - Post-halving to first top: 0 to 1
    - Mid-cycle correction (if double top): 1 to 0
    - Second top (if double top): 0 to 1
    - Distribution/drawdown: 1 to -1

    Args:
        df: DataFrame with date column.
        averages: HalvingAverages with cycle timing data.
        date_col: Name of date column.
        halving_dates: Halving dates to use.

    Returns:
        DataFrame with 'cycle_phase_regressor' column added.
    """
    if halving_dates is None:
        halving_dates = HALVING_DATES

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["cycle_phase_regressor"] = 0.0

    cycle_length = 1461  # ~4 years

    for i, h in enumerate(halving_dates):
        if i == 0:
            continue

        prev_h = halving_dates[i - 1]
        next_h = halving_dates[i + 1] if i + 1 < len(halving_dates) else h + pd.Timedelta(days=cycle_length)

        pre_days = (h - prev_h).days
        post_days = (next_h - h).days

        # Pre-halving: map days before halving to [-1, 0]
        pre_mask = (df[date_col] >= prev_h) & (df[date_col] < h)
        if pre_mask.any():
            days_to_halving = (h - df.loc[pre_mask, date_col]).dt.days
            df.loc[pre_mask, "cycle_phase_regressor"] = -days_to_halving / pre_days

        # Post-halving: encode based on double-top pattern
        post_mask = (df[date_col] >= h) & (df[date_col] < next_h)
        if post_mask.any():
            if averages.double_top_frequency > 0.5 and averages.avg_days_to_first_top > 0:
                # Double-top pattern
                first_top = averages.avg_days_to_first_top
                second_top = averages.avg_days_to_second_top
                mid_point = (first_top + second_top) / 2

                for idx in df.loc[post_mask].index:
                    d = (df.loc[idx, date_col] - h).days
                    if d <= first_top:
                        df.loc[idx, "cycle_phase_regressor"] = d / first_top
                    elif d <= mid_point:
                        df.loc[idx, "cycle_phase_regressor"] = 1 - (d - first_top) / (mid_point - first_top)
                    elif d <= second_top:
                        df.loc[idx, "cycle_phase_regressor"] = (d - mid_point) / (second_top - mid_point)
                    else:
                        remaining = post_days - second_top
                        if remaining > 0:
                            df.loc[idx, "cycle_phase_regressor"] = 1 - 2 * (d - second_top) / remaining
            else:
                # Single-top pattern
                peak_day = averages.avg_days_to_top if averages.avg_days_to_top > 0 else 365
                for idx in df.loc[post_mask].index:
                    d = (df.loc[idx, date_col] - h).days
                    if d <= peak_day:
                        df.loc[idx, "cycle_phase_regressor"] = d / peak_day
                    else:
                        remaining = post_days - peak_day
                        if remaining > 0:
                            df.loc[idx, "cycle_phase_regressor"] = 1 - 2 * (d - peak_day) / remaining

    return df


def sanity_check_forecast(
    forecast: pd.DataFrame,
    averages: HalvingAverages,
    halving_date: pd.Timestamp,
    window_days: int = 90,
    run_up_tolerance_pct: float = 50.0,
    drawdown_tolerance_pct: float = 50.0,
) -> dict[str, Any]:
    """Check whether forecast around a halving is within historical norms.

    Compares forecasted move (max vs min in window around halving) to
    average run-up/drawdown from past cycles. Use to flag Prophet output
    that is wildly outside data-driven ranges.

    Args:
        forecast: Prophet forecast with 'ds', 'yhat' (and optionally 'yhat_lower'/'yhat_upper').
        averages: HalvingAverages from compute_halving_averages().
        halving_date: Halving date to check (e.g. next or last).
        window_days: Days before/after halving to consider.
        run_up_tolerance_pct: Allow run-up up to this multiple of average (e.g. 50 = 50x).
        drawdown_tolerance_pct: Allow drawdown up to this multiple of average.

    Returns:
        Dict with passed: bool, run_up_forecast_pct, drawdown_forecast_pct,
        run_up_ok, drawdown_ok, message.
    """
    forecast = forecast.copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"])
    start = halving_date - pd.Timedelta(days=window_days)
    end = halving_date + pd.Timedelta(days=window_days)
    window = forecast[(forecast["ds"] >= start) & (forecast["ds"] <= end)]

    if window.empty or averages.n_cycles == 0:
        return {
            "passed": True,
            "run_up_forecast_pct": None,
            "drawdown_forecast_pct": None,
            "run_up_ok": True,
            "drawdown_ok": True,
            "message": "No data or no historical cycles for comparison.",
        }

    pre = window[window["ds"] < halving_date]
    post = window[window["ds"] >= halving_date]

    run_up_forecast_pct = None
    drawdown_forecast_pct = None
    run_up_ok = True
    drawdown_ok = True

    if not pre.empty:
        low_pre = pre["yhat"].min()
        high_pre = pre["yhat"].max()
        if low_pre > 0:
            run_up_forecast_pct = (high_pre / low_pre - 1) * 100
            run_up_ok = (
                averages.run_up_pct == 0
                or run_up_forecast_pct <= averages.run_up_pct * (1 + run_up_tolerance_pct / 100)
            )

    if not post.empty:
        high_post = post["yhat"].max()
        low_post = post["yhat"].min()
        if high_post > 0:
            drawdown_forecast_pct = (1 - low_post / high_post) * 100
            drawdown_ok = (
                averages.drawdown_pct == 0
                or drawdown_forecast_pct <= averages.drawdown_pct * (1 + drawdown_tolerance_pct / 100)
            )

    passed = run_up_ok and drawdown_ok
    message = (
        "Forecast within historical halving-cycle ranges."
        if passed
        else "Forecast outside historical run-up/drawdown ranges; review model or inputs."
    )

    return {
        "passed": passed,
        "run_up_forecast_pct": run_up_forecast_pct,
        "drawdown_forecast_pct": drawdown_forecast_pct,
        "run_up_ok": run_up_ok,
        "drawdown_ok": drawdown_ok,
        "message": message,
        "averages_run_up_pct": averages.run_up_pct,
        "averages_drawdown_pct": averages.drawdown_pct,
    }


def backtest_predictions(cycle_metrics: pd.DataFrame) -> pd.DataFrame:
    """Backtest prediction accuracy: predicted vs actual dates for each cycle.

    For each halving, uses ONLY prior cycle data to make predictions,
    then compares to actual outcomes. This validates the prediction method.

    Args:
        cycle_metrics: DataFrame from compute_cycle_metrics().

    Returns:
        DataFrame with predicted vs actual dates and error in days.
    """
    if len(cycle_metrics) < 2:
        return pd.DataFrame()

    results = []

    for i in range(1, len(cycle_metrics)):
        # Use only cycles BEFORE this one for prediction
        prior_cycles = cycle_metrics.iloc[:i]
        current_cycle = cycle_metrics.iloc[i]

        # Compute averages from prior cycles only
        avg_days_to_top = prior_cycles["days_after_halving_to_high"].mean()
        avg_days_to_bottom = prior_cycles["days_after_halving_to_low"].mean()

        halving = current_cycle["halving_date"]

        # Predicted dates
        predicted_top = halving + pd.Timedelta(days=avg_days_to_top)
        predicted_bottom = halving + pd.Timedelta(days=avg_days_to_bottom)

        # Actual dates
        actual_top = current_cycle["post_high_date"]
        actual_bottom = current_cycle["post_low_date"]

        # Errors (positive = predicted too early, negative = predicted too late)
        top_error_days = (actual_top - predicted_top).days
        bottom_error_days = (actual_bottom - predicted_bottom).days

        results.append({
            "halving_date": halving,
            "cycles_used": i,
            # TOP predictions
            "predicted_top": predicted_top,
            "actual_top": actual_top,
            "top_error_days": top_error_days,
            "actual_top_price": current_cycle["post_high_price"],
            # BOTTOM predictions
            "predicted_bottom": predicted_bottom,
            "actual_bottom": actual_bottom,
            "bottom_error_days": bottom_error_days,
            "actual_bottom_price": current_cycle["post_low_price"],
        })

    return pd.DataFrame(results)


def predict_cycle_dates(
    averages: HalvingAverages,
    halving_date: pd.Timestamp | str,
    current_price: float | None = None,
) -> dict[str, Any]:
    """Predict cycle TOP and BOTTOM dates based on historical averages.

    Args:
        averages: HalvingAverages with timing data.
        halving_date: The halving to predict around.
        current_price: Current BTC price for price predictions.

    Returns:
        Dict with predicted dates and prices.
    """
    halving = pd.Timestamp(halving_date)

    result = {
        "halving_date": halving,
        "n_cycles_used": averages.n_cycles,
    }

    if averages.n_cycles == 0:
        result["error"] = "No historical data"
        return result

    # Predicted TOP (cycle peak after halving)
    if averages.avg_days_to_top > 0:
        result["predicted_top_date"] = halving + pd.Timedelta(days=averages.avg_days_to_top)
        result["avg_days_to_top"] = averages.avg_days_to_top
        if current_price:
            # Rough estimate: price at top = current * (1 + avg_run_up)
            result["predicted_top_price"] = current_price * (1 + averages.run_up_pct / 100)

    # Predicted BOTTOM (cycle low after halving)
    if averages.avg_days_to_bottom > 0:
        result["predicted_bottom_date"] = halving + pd.Timedelta(days=averages.avg_days_to_bottom)
        result["avg_days_to_bottom"] = averages.avg_days_to_bottom
        if current_price and averages.avg_days_to_top > 0:
            # Bottom = top price * (1 - drawdown)
            top_price = current_price * (1 + averages.run_up_pct / 100)
            result["predicted_bottom_price"] = top_price * (1 - averages.drawdown_pct / 100)

    # Predicted accumulation low (before halving)
    if averages.avg_days_before_low > 0:
        result["predicted_accumulation_date"] = halving - pd.Timedelta(days=averages.avg_days_before_low)
        result["avg_days_before_low"] = averages.avg_days_before_low

    return result


def print_halving_summary(cycle_metrics: pd.DataFrame, averages: HalvingAverages) -> None:
    """Print a short summary of cycle metrics and averages to stdout."""
    if cycle_metrics.empty:
        print("No halving cycles computed (insufficient data).")
        return

    print("\n" + "=" * 70)
    print("HISTORICAL CYCLE DATA")
    print("=" * 70)

    import numpy as np

    # Per-cycle details
    for _, row in cycle_metrics.iterrows():
        h_date = row["halving_date"].strftime("%Y-%m-%d")
        print(f"\nHalving: {h_date}")

        # Check for double-top pattern
        if row.get("is_double_top", False):
            print("  DOUBLE TOP DETECTED:")
            first_date = row["first_top_date"]
            second_date = row["second_top_date"]
            mid_date = row["mid_cycle_low_date"]
            print(f"    1st TOP:  {first_date.strftime('%Y-%m-%d')} @ ${row['first_top_price']:,.0f}")
            print(f"              ({row['days_to_first_top']:.0f} days after halving)")
            print(f"    Mid LOW:  {mid_date.strftime('%Y-%m-%d')} @ ${row['mid_cycle_low_price']:,.0f}")
            print(f"              (-{row['mid_cycle_drawdown_pct']:.1f}% correction)")
            print(f"    2nd TOP:  {second_date.strftime('%Y-%m-%d')} @ ${row['second_top_price']:,.0f}")
            print(f"              ({row['days_to_second_top']:.0f} days after halving)")
            print(f"              ({row['days_between_tops']:.0f} days between tops)")
        else:
            print(f"  TOP:    {row['post_high_date'].strftime('%Y-%m-%d')} @ ${row['post_high_price']:,.0f}")
            print(f"          ({row['days_after_halving_to_high']:.0f} days after halving)")

        print(f"  BOTTOM: {row['post_low_date'].strftime('%Y-%m-%d')} @ ${row['post_low_price']:,.0f}")
        print(f"          ({row['days_after_halving_to_low']:.0f} days after halving)")
        print(f"  Run-up: {row['run_up_pct']:.1f}% | Drawdown: {row['drawdown_pct']:.1f}%")

    # Logarithmic decay analysis
    print("\n" + "-" * 70)
    print("LOGARITHMIC CYCLE ANALYSIS")
    print("-" * 70)
    print()
    print("  Cycle | Top Price  | Bottom Price | Drawdown % | Log(Top) | Log(Bottom) | Log Drop")
    print("  " + "-" * 75)

    for i, (_, row) in enumerate(cycle_metrics.iterrows(), 1):
        top_price = row["post_high_price"]
        bottom_price = row["post_low_price"]
        drawdown = row["drawdown_pct"]
        log_top = np.log10(top_price)
        log_bottom = np.log10(bottom_price)
        log_drop = log_top - log_bottom

        print(f"  {i:5} | ${top_price:>9,.0f} | ${bottom_price:>11,.0f} | {drawdown:>9.1f}% | {log_top:>8.3f} | {log_bottom:>11.3f} | {log_drop:>8.3f}")

    # Show trend in drawdowns
    drawdowns = cycle_metrics["drawdown_pct"].tolist()
    if len(drawdowns) >= 2:
        print()
        print("  Drawdown trend:")
        for i in range(1, len(drawdowns)):
            change = drawdowns[i] - drawdowns[i-1]
            print(f"    Cycle {i} → {i+1}: {change:+.1f}% change in drawdown")

    print("\n" + "-" * 70)
    print("AVERAGES")
    print("-" * 70)
    print(f"  Cycles analyzed: {averages.n_cycles}")
    print(f"  Avg run-up:      {averages.run_up_pct:.1f}% over {averages.run_up_days:.0f} days")
    print(f"  Avg drawdown:    {averages.drawdown_pct:.1f}% over {averages.drawdown_days:.0f} days")
    print(f"  Avg days to TOP:    {averages.avg_days_to_top:.0f} days after halving")
    print(f"  Avg days to BOTTOM: {averages.avg_days_to_bottom:.0f} days after halving")

    # Double-top statistics
    if averages.double_top_frequency > 0:
        print()
        print(f"  DOUBLE TOP PATTERN:")
        print(f"    Frequency:          {averages.double_top_frequency:.0%} of cycles")
        print(f"    Avg days to 1st top: {averages.avg_days_to_first_top:.0f} days after halving")
        print(f"    Avg days to 2nd top: {averages.avg_days_to_second_top:.0f} days after halving")
        print(f"    Avg days between:    {averages.avg_days_between_tops:.0f} days")
        print(f"    Avg mid-correction:  {averages.avg_mid_cycle_drawdown_pct:.1f}%")

    # PREDICTED vs ACTUAL comparison
    backtest = backtest_predictions(cycle_metrics)
    if not backtest.empty:
        print("\n" + "-" * 70)
        print("PREDICTED vs ACTUAL (Backtested)")
        print("-" * 70)
        print("Using prior cycles to predict each halving:")
        print()
        for _, row in backtest.iterrows():
            h_date = row["halving_date"].strftime("%Y-%m-%d")
            print(f"  Halving {h_date} (using {row['cycles_used']} prior cycle(s)):")
            print(f"    TOP:")
            print(f"      Predicted: {row['predicted_top'].strftime('%Y-%m-%d')}")
            print(f"      Actual:    {row['actual_top'].strftime('%Y-%m-%d')} @ ${row['actual_top_price']:,.0f}")
            err_sign = "+" if row['top_error_days'] >= 0 else ""
            print(f"      Error:     {err_sign}{row['top_error_days']} days")
            print(f"    BOTTOM:")
            print(f"      Predicted: {row['predicted_bottom'].strftime('%Y-%m-%d')}")
            print(f"      Actual:    {row['actual_bottom'].strftime('%Y-%m-%d')} @ ${row['actual_bottom_price']:,.0f}")
            err_sign = "+" if row['bottom_error_days'] >= 0 else ""
            print(f"      Error:     {err_sign}{row['bottom_error_days']} days")
            print()

        # Summary stats
        avg_top_error = backtest["top_error_days"].abs().mean()
        avg_bottom_error = backtest["bottom_error_days"].abs().mean()
        print(f"  Average absolute error:")
        print(f"    TOP prediction:    {avg_top_error:.0f} days")
        print(f"    BOTTOM prediction: {avg_bottom_error:.0f} days")

    # Predict bottom from most recent top
    from .decay import predict_drawdown

    last_cycle = cycle_metrics.iloc[-1]
    last_top_date = last_cycle["post_high_date"]
    last_top_price = last_cycle["post_high_price"]

    # Simple average prediction
    predicted_bottom_date = last_top_date + pd.Timedelta(days=averages.drawdown_days)
    predicted_bottom_price = last_top_price * (1 - averages.drawdown_pct / 100)

    # Decay-adjusted prediction (accounts for market maturation)
    decay_pred = predict_drawdown(cycle_metrics, target_cycle=len(cycle_metrics) + 1)
    decay_bottom_price = last_top_price * (1 - decay_pred.predicted_value)
    decay_bottom_low = last_top_price * (1 - decay_pred.confidence_high)
    decay_bottom_high = last_top_price * (1 - decay_pred.confidence_low)

    print("\n" + "-" * 70)
    print("DECAY CURVE FITTING")
    print("-" * 70)
    print()
    print("  Formula: drawdown = a * exp(-b * cycle) + c")
    print(f"  Where c = {decay_pred.floor:.1%} (asymptotic floor as market matures)")
    print(f"  Decay rate (b) = {decay_pred.decay_rate:.4f}")
    print()

    from .decay import exp_decay, fit_decay_curve

    drawdowns = (cycle_metrics["drawdown_pct"] / 100).tolist()
    cycle_nums = list(range(1, len(drawdowns) + 1))

    # Get the actual fitted parameters
    params, _ = fit_decay_curve(cycle_nums, drawdowns)
    a, b, c = params

    print(f"  Fitted params: a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print()
    print("  Cycle | Actual DD | Fitted DD | Residual")
    print("  " + "-" * 45)

    for i, dd in enumerate(drawdowns, 1):
        fitted = float(exp_decay(i, a, b, c))
        residual = dd - fitted
        print(f"  {i:5} | {dd:>9.1%} | {fitted:>9.1%} | {residual:>+8.1%}")

    # Show next cycle prediction
    next_cycle = len(drawdowns) + 1
    next_fitted = float(exp_decay(next_cycle, a, b, c))
    print(f"  {next_cycle:5} | {'???':>9} | {next_fitted:>9.1%} | {'(predicted)':>8}")
    print()
    print(f"  Model fit (R²): {decay_pred.r_squared:.3f}")

    print("\n" + "-" * 70)
    print("PREDICTED BOTTOM (Buy Zone)")
    print("-" * 70)
    print(f"  Last TOP:           {last_top_date.strftime('%Y-%m-%d')} @ ${last_top_price:,.0f}")
    print()
    print("  Simple Average:")
    print(f"    Drawdown:         {averages.drawdown_pct:.1f}% over {averages.drawdown_days:.0f} days")
    print(f"    Predicted date:   {predicted_bottom_date.strftime('%Y-%m-%d')}")
    print(f"    Predicted price:  ${predicted_bottom_price:,.0f}")
    print()
    print("  Decay-Adjusted (accounts for market maturation):")
    print(f"    Drawdown:         {decay_pred.predicted_value:.1%} (floor: {decay_pred.floor:.1%})")
    print(f"    Price range:      ${decay_bottom_low:,.0f} - ${decay_bottom_high:,.0f}")
    print(f"    Best estimate:    ${decay_bottom_price:,.0f}")
    print(f"    Model fit (R²):   {decay_pred.r_squared:.3f}")

    print("=" * 70 + "\n")
