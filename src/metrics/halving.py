"""
Data-driven halving cycle metrics from actual BTC prices.

Measures run-up (low → pre-halving high), drawdown (high → post-halving low),
and durations per cycle; averages across cycles; and exposes results for
parameterising Prophet or sanity-checking forecasts.
"""

from dataclasses import dataclass
from typing import Any

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
class HalvingAverages:
    """Averaged metrics across completed halving cycles."""

    run_up_pct: float
    run_up_days: float
    drawdown_pct: float
    drawdown_days: float
    n_cycles: int


def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with ds sorted ascending, one row per day."""
    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["ds", "y"])
    out = out.groupby("ds", as_index=False)["y"].mean()
    return out.sort_values("ds").reset_index(drop=True)


def compute_cycle_metrics(
    df: pd.DataFrame,
    date_col: str = "ds",
    price_col: str = "y",
    halving_dates: pd.DatetimeIndex | None = None,
    buffer_days: int = 90,
) -> pd.DataFrame:
    """Compute run-up, drawdown, and duration for each halving cycle.

    For each halving H_i (with previous H_{i-1} and next H_{i+1}):

    - Run-up before halving: low → pre-halving high in [H_{i-1}, H_i].
      Low = argmin(price), high = argmax(price) after low. The pre-halving
      high is restricted to at least buffer_days before H_i to avoid counting
      the run-up to the next halving.
    - Drawdown after halving: high → post-halving low in [H_i, H_{i+1}].
      The post-halving high is restricted to [H_i + buffer_days, H_{i+1} - buffer_days]
      so it is not in the immediate aftermath nor in the run-up to the next halving.

    Args:
        df: DataFrame with date and price columns.
        date_col: Name of date column.
        price_col: Name of price column.
        halving_dates: Halving dates to use; defaults to HALVING_DATES.
        buffer_days: Minimum days between key prices and halving boundaries.
            Default 90.

    Returns:
        DataFrame with one row per halving cycle and columns:
        halving_date, run_up_pct, run_up_days, drawdown_pct, drawdown_days,
        pre_low_date, pre_high_date, post_high_date, post_low_date, etc.
    """
    if halving_dates is None:
        halving_dates = HALVING_DATES

    out = _ensure_sorted(df.rename(columns={date_col: "ds", price_col: "y"}))
    halvings = pd.to_datetime(halving_dates)
    buffer = pd.Timedelta(days=buffer_days)

    rows = []

    for i in range(1, len(halvings) - 1):
        h_prev, h, h_next = halvings[i - 1], halvings[i], halvings[i + 1]
        h_ts = pd.Timestamp(h)

        pre = out[(out["ds"] >= h_prev) & (out["ds"] < h)]
        post = out[(out["ds"] >= h) & (out["ds"] < h_next)]

        if pre.empty or post.empty:
            continue

        # Run-up: low → pre-halving high (high must be after low and at least buffer_days before halving)
        low_idx = pre["y"].idxmin()
        low_row = pre.loc[low_idx]
        pre_after_low = pre[pre["ds"] > low_row["ds"]]
        pre_after_low = pre_after_low[pre_after_low["ds"] <= h_ts - buffer]
        if pre_after_low.empty:
            continue
        high_idx = pre_after_low["y"].idxmax()
        high_row = pre_after_low.loc[high_idx]

        run_up_pct = (float(high_row["y"]) / float(low_row["y"]) - 1) * 100
        run_up_days = (high_row["ds"] - low_row["ds"]).days

        # Drawdown: post-halving high in [h + buffer, h_next - buffer], then low after that
        post_valid = post[(post["ds"] >= h_ts + buffer) & (post["ds"] <= pd.Timestamp(h_next) - buffer)]
        if post_valid.empty:
            continue
        post_high_idx = post_valid["y"].idxmax()
        post_high_row = post_valid.loc[post_high_idx]
        post_after_high = post[post["ds"] > post_high_row["ds"]]
        if post_after_high.empty:
            continue
        post_low_idx = post_after_high["y"].idxmin()
        post_low_row = post_after_high.loc[post_low_idx]

        drawdown_pct = (1 - float(post_low_row["y"]) / float(post_high_row["y"])) * 100
        drawdown_days = (post_low_row["ds"] - post_high_row["ds"]).days

        # Days before/after halving for each key price (low/high)
        days_before_halving_to_low = (h_ts - pd.Timestamp(low_row["ds"])).days
        days_before_halving_to_high = (h_ts - pd.Timestamp(high_row["ds"])).days
        days_after_halving_to_high = (pd.Timestamp(post_high_row["ds"]) - h_ts).days
        days_after_halving_to_low = (pd.Timestamp(post_low_row["ds"]) - h_ts).days

        rows.append(
            {
                "halving_date": h,
                "run_up_pct": run_up_pct,
                "run_up_days": run_up_days,
                "drawdown_pct": drawdown_pct,
                "drawdown_days": drawdown_days,
                "pre_low_date": low_row["ds"],
                "pre_low_price": low_row["y"],
                "days_before_halving_to_low": days_before_halving_to_low,
                "pre_high_date": high_row["ds"],
                "pre_high_price": high_row["y"],
                "days_before_halving_to_high": days_before_halving_to_high,
                "post_high_date": post_high_row["ds"],
                "post_high_price": post_high_row["y"],
                "days_after_halving_to_high": days_after_halving_to_high,
                "post_low_date": post_low_row["ds"],
                "post_low_price": post_low_row["y"],
                "days_after_halving_to_low": days_after_halving_to_low,
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

    return HalvingAverages(
        run_up_pct=float(cycle_metrics["run_up_pct"].mean()),
        run_up_days=float(cycle_metrics["run_up_days"].mean()),
        drawdown_pct=float(cycle_metrics["drawdown_pct"].mean()),
        drawdown_days=float(cycle_metrics["drawdown_days"].mean()),
        n_cycles=len(cycle_metrics),
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


def print_halving_summary(cycle_metrics: pd.DataFrame, averages: HalvingAverages) -> None:
    """Print a short summary of cycle metrics and averages to stdout."""
    if cycle_metrics.empty:
        print("No halving cycles computed (insufficient data).")
        return

    print("Per-cycle metrics:")
    print(cycle_metrics.to_string())
    print()
    print(
        f"Averages (n={averages.n_cycles}): "
        f"run_up {averages.run_up_pct:.1f}% in {averages.run_up_days:.0f} days, "
        f"drawdown {averages.drawdown_pct:.1f}% in {averages.drawdown_days:.0f} days."
    )
    params = get_prophet_params_from_halving(averages)
    if params:
        print("Suggested Prophet-oriented params:", params)
