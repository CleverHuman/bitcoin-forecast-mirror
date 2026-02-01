# Halving Cycle Metrics

Data-driven analysis of Bitcoin halving cycles to measure run-up, drawdown, and durations.

## Overview

For each halving cycle, this module measures:

- **Run-up**: Price increase from cycle low to pre-halving high
- **Drawdown**: Price decrease from post-halving high to cycle low
- **Duration**: Days for each phase

A **buffer** (default 90 days) restricts overlaps with halving boundaries: the pre-halving high must be at least that many days before the halving, and the post-halving high must be at least that many days after the halving and before the next halving. This avoids counting the run-up to the next halving as the “post-halving high” of the current cycle.

These metrics are averaged across cycles to:
- Parameterize Prophet model settings
- Sanity-check forecast outputs against historical norms

## Usage

### Compute Cycle Metrics

```python
from src.metrics import compute_cycle_metrics, compute_halving_averages

# df must have 'ds' (date) and 'y' (price) columns
cycle_metrics = compute_cycle_metrics(df)

print(cycle_metrics)
# Returns DataFrame with columns:
# - halving_date
# - run_up_pct, run_up_days
# - drawdown_pct, drawdown_days
# - pre_low_date, pre_low_price, pre_high_date, pre_high_price
# - post_high_date, post_high_price, post_low_date, post_low_price
```

### Compute Averages

```python
averages = compute_halving_averages(cycle_metrics=cycle_metrics)

print(f"Cycles analyzed: {averages.n_cycles}")
print(f"Avg run-up: {averages.run_up_pct:.1f}% in {averages.run_up_days:.0f} days")
print(f"Avg drawdown: {averages.drawdown_pct:.1f}% in {averages.drawdown_days:.0f} days")
```

### Get Prophet Parameters

Use historical averages to suggest Prophet settings:

```python
from src.metrics import get_prophet_params_from_halving

params = get_prophet_params_from_halving(averages)
# Returns dict with:
# - changepoint_range: suggested value based on cycle phase lengths
# - halving_run_up_days_avg, halving_drawdown_days_avg
# - halving_run_up_pct_avg, halving_drawdown_pct_avg

model = Prophet(
    changepoint_range=params.get("changepoint_range", 0.8),
    # ... other params
)
```

### Sanity Check Forecasts

Validate that Prophet output aligns with historical patterns:

```python
from src.metrics import sanity_check_forecast
import pandas as pd

next_halving = pd.Timestamp("2028-04-11")

check = sanity_check_forecast(
    forecast=forecast_df,       # Prophet forecast with 'ds', 'yhat'
    averages=averages,          # HalvingAverages from compute_halving_averages
    halving_date=next_halving,
    window_days=180,            # Days before/after halving to analyze
)

print(check["message"])
# "Forecast within historical halving-cycle ranges." or warning message

if not check["passed"]:
    print(f"Forecast run-up: {check['run_up_forecast_pct']:.1f}%")
    print(f"Historical avg: {check['averages_run_up_pct']:.1f}%")
```

### Print Summary

```python
from src.metrics import print_halving_summary

print_halving_summary(cycle_metrics, averages)
# Outputs formatted table of per-cycle metrics and averages
```

## API Reference

### Constants

#### `HALVING_DATES`

`pd.DatetimeIndex` of historical and projected halving dates:
- 2012-11-28 (1st)
- 2016-07-09 (2nd)
- 2020-05-11 (3rd)
- 2024-04-19 (4th)
- 2028-04-11 (5th, projected)

### Classes

#### `HalvingAverages`

Dataclass with averaged metrics:

| Field | Type | Description |
|-------|------|-------------|
| `run_up_pct` | float | Average run-up percentage |
| `run_up_days` | float | Average run-up duration in days |
| `drawdown_pct` | float | Average drawdown percentage |
| `drawdown_days` | float | Average drawdown duration in days |
| `n_cycles` | int | Number of cycles with complete data |

### Functions

#### `compute_cycle_metrics(df, date_col, price_col, halving_dates)`

Compute run-up/drawdown metrics for each halving cycle.

**Parameters:**
- `df`: DataFrame with date and price columns
- `date_col`: Name of date column (default: "ds")
- `price_col`: Name of price column (default: "y")
- `halving_dates`: Custom halving dates (default: `HALVING_DATES`)
- `buffer_days`: Minimum days between key prices and halving boundaries (default: 90)

**Returns:** DataFrame with one row per cycle

#### `compute_halving_averages(df, cycle_metrics)`

Average metrics across cycles. Pass either raw `df` or pre-computed `cycle_metrics`.

**Returns:** `HalvingAverages` dataclass

#### `get_prophet_params_from_halving(averages)`

Suggest Prophet parameters from cycle averages.

**Returns:** Dict with `changepoint_range` and summary stats

#### `sanity_check_forecast(forecast, averages, halving_date, window_days, run_up_tolerance_pct, drawdown_tolerance_pct)`

Check if forecast aligns with historical patterns.

**Parameters:**
- `forecast`: Prophet forecast DataFrame
- `averages`: `HalvingAverages` instance
- `halving_date`: Date to check around
- `window_days`: Days before/after to analyze (default: 90)
- `run_up_tolerance_pct`: Allowed deviation (default: 50%)
- `drawdown_tolerance_pct`: Allowed deviation (default: 50%)

**Returns:** Dict with `passed`, metrics, and `message`

#### `print_halving_summary(cycle_metrics, averages)`

Print formatted summary to stdout.
