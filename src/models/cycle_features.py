"""Cycle-aware feature engineering for BTC price prediction.

Adds features that encode position within the ~4-year halving cycle,
enabling ML models to learn cycle-dependent patterns.
"""

from enum import Enum

import numpy as np
import pandas as pd

from src.metrics import HALVING_DATES


class CyclePhase(Enum):
    """Phases within a halving cycle based on historical patterns."""

    ACCUMULATION = "accumulation"  # Post-drawdown, pre-run-up (buy zone)
    PRE_HALVING_RUNUP = "pre_halving_runup"  # ~180 days before halving (hold)
    POST_HALVING_CONSOLIDATION = "post_halving_consolidation"  # 0-120 days after
    BULL_RUN = "bull_run"  # 120-365 days after halving (potential sell zone)
    DISTRIBUTION = "distribution"  # 365-545 days after (sell zone)
    DRAWDOWN = "drawdown"  # 545+ days after (avoid/accumulate)


# Phase boundaries in days relative to halving (negative = before)
PHASE_BOUNDARIES = {
    CyclePhase.PRE_HALVING_RUNUP: (-180, 0),
    CyclePhase.POST_HALVING_CONSOLIDATION: (0, 120),
    CyclePhase.BULL_RUN: (120, 365),
    CyclePhase.DISTRIBUTION: (365, 545),
    CyclePhase.DRAWDOWN: (545, 1095),  # ~3 years until next pre-halving
    CyclePhase.ACCUMULATION: (-545, -180),  # Between drawdown and run-up
}


def get_cycle_phase(date: pd.Timestamp, halving_dates: pd.DatetimeIndex = None) -> CyclePhase:
    """Determine the cycle phase for a given date.

    Args:
        date: Date to classify.
        halving_dates: Halving dates to use. Defaults to HALVING_DATES.

    Returns:
        CyclePhase enum value.
    """
    if halving_dates is None:
        halving_dates = HALVING_DATES

    date = pd.Timestamp(date)

    # Find the nearest halving (past or future)
    days_to_halvings = [(h - date).days for h in halving_dates]

    # Find closest past halving and next future halving
    past_halvings = [(i, d) for i, d in enumerate(days_to_halvings) if d <= 0]
    future_halvings = [(i, d) for i, d in enumerate(days_to_halvings) if d > 0]

    if past_halvings:
        # Days since last halving (positive number)
        days_since = -max(past_halvings, key=lambda x: x[1])[1]
    else:
        days_since = None

    if future_halvings:
        # Days until next halving (positive number)
        days_until = min(future_halvings, key=lambda x: x[1])[1]
    else:
        days_until = None

    # Classify based on position relative to halvings
    if days_until is not None and days_until <= 180:
        return CyclePhase.PRE_HALVING_RUNUP
    elif days_since is not None:
        if days_since <= 120:
            return CyclePhase.POST_HALVING_CONSOLIDATION
        elif days_since <= 365:
            return CyclePhase.BULL_RUN
        elif days_since <= 545:
            return CyclePhase.DISTRIBUTION
        elif days_since <= 1095:
            return CyclePhase.DRAWDOWN
        else:
            return CyclePhase.ACCUMULATION
    else:
        return CyclePhase.ACCUMULATION


def add_cycle_features(
    df: pd.DataFrame,
    date_col: str = "ds",
    halving_averages: "HalvingAverages | None" = None,
) -> pd.DataFrame:
    """Add halving cycle features to a DataFrame.

    Features added:
    - days_since_halving: Days since the most recent halving
    - days_until_halving: Days until the next halving
    - cycle_progress: 0-1 progress through current ~4-year cycle
    - cycle_phase: Categorical phase (accumulation, run-up, etc.)
    - cycle_sin/cycle_cos: Sinusoidal encoding of cycle position
    - halving_proximity: 0-1 score (1 = at halving, 0 = mid-cycle)
    - pre_halving_weight: Higher weight closer to halving (for run-up)
    - post_halving_weight: Higher weight after halving (data-driven Gaussian)

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.
        halving_averages: Optional HalvingAverages for data-driven parameters.
            If provided, uses avg_days_to_top and historical spread.

    Returns:
        DataFrame with cycle features added.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    halving_dates = HALVING_DATES
    cycle_length_days = 4 * 365.25  # ~4 years between halvings

    days_since = []
    days_until = []
    cycle_progress = []
    phases = []

    for date in df[date_col]:
        date = pd.Timestamp(date)

        # Find past and future halvings
        past = halving_dates[halving_dates <= date]
        future = halving_dates[halving_dates > date]

        if len(past) > 0:
            last_halving = past[-1]
            ds = (date - last_halving).days
        else:
            ds = None

        if len(future) > 0:
            next_halving = future[0]
            du = (next_halving - date).days
        else:
            du = None

        days_since.append(ds)
        days_until.append(du)

        # Cycle progress (0 at halving, 1 at next halving)
        if ds is not None and du is not None:
            total = ds + du
            progress = ds / total if total > 0 else 0
        elif ds is not None:
            progress = min(ds / cycle_length_days, 1.0)
        else:
            progress = 0

        cycle_progress.append(progress)
        phases.append(get_cycle_phase(date, halving_dates).value)

    df["days_since_halving"] = days_since
    df["days_until_halving"] = days_until
    df["cycle_progress"] = cycle_progress
    df["cycle_phase"] = phases

    # Sinusoidal encoding (captures cyclical nature)
    df["cycle_sin"] = np.sin(2 * np.pi * df["cycle_progress"])
    df["cycle_cos"] = np.cos(2 * np.pi * df["cycle_progress"])

    # Proximity to halving (peaks at 0 and 1)
    df["halving_proximity"] = 1 - np.abs(2 * df["cycle_progress"] - 1)

    # Directional weights for run-up and drawdown
    # Use data-driven parameters if available
    if halving_averages is not None and halving_averages.run_up_days > 0:
        pre_halving_window = int(halving_averages.run_up_days)
    else:
        pre_halving_window = 365  # Default: 1 year

    if halving_averages is not None and halving_averages.avg_days_to_top > 0:
        post_halving_peak = halving_averages.avg_days_to_top
        # Estimate spread from drawdown duration or use reasonable default
        post_halving_spread = halving_averages.drawdown_days / 3 if halving_averages.drawdown_days > 0 else 150
    else:
        post_halving_peak = 240  # Default fallback
        post_halving_spread = 150

    # Pre-halving weight: increases as we approach halving
    df["pre_halving_weight"] = np.where(
        df["days_until_halving"].notna() & (df["days_until_halving"] <= pre_halving_window),
        1 - (df["days_until_halving"] / pre_halving_window),
        0,
    )

    # Post-halving weight: Gaussian centered at data-driven peak day
    df["post_halving_weight"] = np.where(
        df["days_since_halving"].notna(),
        np.exp(-((df["days_since_halving"] - post_halving_peak) ** 2) / (2 * post_halving_spread**2)),
        0,
    )

    return df


def create_cycle_regressors_for_prophet(
    df: pd.DataFrame,
    halving_averages: "HalvingAverages | None" = None,
) -> pd.DataFrame:
    """Create cycle-based regressors suitable for Prophet's add_regressor().

    Prophet regressors must be numeric and should be scaled 0-1 or standardized.

    Args:
        df: DataFrame with 'ds' column.
        halving_averages: Optional HalvingAverages for data-driven Gaussian parameters.

    Returns:
        DataFrame with regressor columns added.
    """
    df = add_cycle_features(df, date_col="ds", halving_averages=halving_averages)

    # Normalize days columns to 0-1 range
    max_cycle = 4 * 365

    df["reg_days_since_halving"] = df["days_since_halving"].fillna(0) / max_cycle
    df["reg_days_until_halving"] = df["days_until_halving"].fillna(max_cycle) / max_cycle

    # Keep sinusoidal features (already -1 to 1)
    df["reg_cycle_sin"] = df["cycle_sin"]
    df["reg_cycle_cos"] = df["cycle_cos"]

    # Directional weights (already 0-1)
    df["reg_pre_halving"] = df["pre_halving_weight"]
    df["reg_post_halving"] = df["post_halving_weight"]

    return df
