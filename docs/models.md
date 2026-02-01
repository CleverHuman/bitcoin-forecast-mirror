# Models Module

Cycle-aware algorithms for BTC price forecasting and signal generation.

## Overview

The models module provides:

1. **Cycle Features** - Encode position within the 4-year halving cycle
2. **Signals** - Generate buy/sell recommendations
3. **Ensemble** - Combine Prophet with cycle-aware adjustments

## Cycle Phases

The halving cycle is divided into phases based on historical patterns:

| Phase | Days from Halving | Bias | Description |
|-------|------------------|------|-------------|
| Accumulation | -545 to -180 | Bullish | Post-drawdown, before run-up starts |
| Pre-Halving Run-up | -180 to 0 | Bullish | Price typically rises into halving |
| Post-Halving Consolidation | 0 to 120 | Neutral | Choppy, digesting the event |
| Bull Run | 120 to 365 | Neutral→Bearish | Parabolic phase, watch for top |
| Distribution | 365 to 545 | Bearish | Take profits zone |
| Drawdown | 545 to 1095 | Bullish* | Bear market, accumulation opportunity |

## Usage

### Add Cycle Features

```python
from src.models import add_cycle_features

df = add_cycle_features(df, date_col="ds")

# New columns:
# - days_since_halving, days_until_halving
# - cycle_progress (0 to 1)
# - cycle_phase (categorical)
# - cycle_sin, cycle_cos (sinusoidal encoding)
# - pre_halving_weight, post_halving_weight
```

### Generate Signals

```python
from src.models import generate_signals, get_current_signal

# Add signals combining cycle position + technical indicators
df = generate_signals(
    df,
    cycle_weight=0.4,     # Weight for cycle-based signal
    technical_weight=0.6,  # Weight for RSI/MACD/MA
)

# Get current recommendation
current = get_current_signal(df)
print(current["signal"])        # "strong_buy", "buy", "hold", "sell", "strong_sell"
print(current["recommendation"])  # Human-readable advice
```

### Backtest Signals

```python
from src.models import backtest_signals

results = backtest_signals(df, initial_capital=10000)

print(f"Strategy return: {results['total_return_pct']:.1f}%")
print(f"Buy & hold return: {results['buy_hold_return_pct']:.1f}%")
print(f"Outperformance: {results['outperformance_pct']:.1f}%")
```

### Ensemble Forecast

Combines Prophet trend with cycle-aware adjustments:

```python
from src.models import train_simple_ensemble

forecast = train_simple_ensemble(df, periods=365)

# Contains:
# - yhat: base Prophet forecast
# - yhat_ensemble: cycle-adjusted forecast
# - cycle_adjustment: multiplier applied
```

### Prophet with Cycle Regressors

Add cycle features as Prophet regressors:

```python
from src.models import train_prophet_with_regressors

model, forecast = train_prophet_with_regressors(
    df,
    periods=365,
    use_cycle_regressors=True,  # Adds cycle_sin, cycle_cos, etc.
    halving_averages=averages,  # Data-driven parameters
    decay_params=decay_params,  # From fit_decay_curve()
)
```

---

## Regressor System

The ensemble model uses 5 configurable regressors for cycle-aware forecasting. All regressors use **data-driven parameters** from historical cycle analysis.

### Regressor Overview

| Regressor | Responsibility | Values | Env Variable |
|-----------|---------------|--------|--------------|
| **CYCLE** | Position in cycle (sin/cos, proximity) | -1 to 1 | `REGRESSOR_CYCLE` |
| **DOUBLE_TOP** | Pattern detection (first/second top windows) | -0.4 to 0.6 | `REGRESSOR_DOUBLE_TOP` |
| **CYCLE_PHASE** | Phase-specific behavior slopes | -0.5 to 1.0 | `REGRESSOR_CYCLE_PHASE` |
| **DECAY** | Drawdown magnitude scaling | -0.08 to 0 | `REGRESSOR_DECAY` |
| **ENSEMBLE_ADJUST** | Timing-based post-processing | ±15-20% | `REGRESSOR_ENSEMBLE_ADJUST` |

### Configuration

Toggle regressors in `.env`:

```bash
REGRESSOR_CYCLE=true
REGRESSOR_DOUBLE_TOP=true
REGRESSOR_CYCLE_PHASE=true
REGRESSOR_DECAY=true
REGRESSOR_ENSEMBLE_ADJUST=true
```

### Validation

Check configuration for conflicts:

```python
from src.models import validate_regressor_config, print_regressor_responsibilities

# Show what each regressor does
print_regressor_responsibilities()

# Check for known conflicts
warnings = validate_regressor_config()
# WARNING: DECAY + ENSEMBLE_ADJUST both reduce forecast post-365 days
# INFO: CYCLE + CYCLE_PHASE both encode cycle position (potential redundancy)
```

### Known Conflicts

| Conflict | Issue | Recommendation |
|----------|-------|----------------|
| DECAY + ENSEMBLE_ADJUST | Both reduce forecast during post-365-day period | Ensure magnitudes don't compound excessively |
| CYCLE + CYCLE_PHASE | Both encode cycle position | May cause multicollinearity |
| DOUBLE_TOP + CYCLE_PHASE | Both model post-halving behavior | Different focus: pattern vs smooth transitions |

### Regressor Details

#### CYCLE Regressor (`cycle_features.py`)

Encodes position within the 4-year cycle:

- **`reg_cycle_sin/cos`**: Sinusoidal encoding of cycle progress
- **`reg_pre_halving`**: Ramps up as halving approaches (uses `run_up_days` from history)
- **`reg_post_halving`**: Gaussian centered at `avg_days_to_top` (data-driven)

```python
# Parameters from halving_averages:
post_halving_peak = averages.avg_days_to_top  # Was hardcoded 240
post_halving_spread = averages.drawdown_days / 3  # Was hardcoded 150
pre_halving_window = averages.run_up_days  # Was hardcoded 365
```

#### DOUBLE_TOP Regressor (`halving.py`)

Detects double-top cycle pattern with Gaussian-weighted continuous encoding:

- **Positive values (~0.5)**: Near expected top windows (bullish)
- **Negative values (~-0.3)**: Mid-cycle correction (bearish)
- **Values weighted by** `double_top_frequency` (historical confidence)

#### CYCLE_PHASE Regressor (`halving.py`)

Encodes cycle phase with **bullish pre-halving** behavior. Uses per-halving data from `cycle_metrics` when available.

**Timeline:**
```
         Pre-halving          |  Consol  |   Bull   |  Dist/DD  | Bear
         ────────────────────►|◄────────►|◄────────►|◄─────────►|◄────
Value:   0 ──────────► +0.5   | +0.5→0   | 0→+0.3   | +0.3→-0.3 | -0.2
         (bullish run-up)     | (neutral)| (bullish)| (bearish) |
```

**Phases:**

| Phase | Days | Regressor Value | Behavior |
|-------|------|-----------------|----------|
| Pre-halving run-up | -run_up_days → 0 | 0 → +0.5 | **Bullish** - ramps UP toward halving |
| Post-halving consolidation | 0 → 120 | +0.5 → 0 | Neutral - drops from halving peak |
| Bull run | 120 → peak_day | 0 → +0.3 | Moderately bullish |
| Distribution/Drawdown | peak_day → bottom_day | +0.3 → -0.3 | Bearish - ramps down |
| Late bear | bottom_day+ | -0.2 | Stays moderately bearish |

**Data sources from `cycle_metrics`:**
- `days_after_halving_to_high` → peak timing for each halving
- `days_after_halving_to_low` → bottom timing for each halving
- `run_up_days` → pre-halving ramp duration

**Future halving dampening:**
- Halvings beyond historical data get **50% dampened** effect
- Prevents forecast skyrocketing into unknown territory

#### DECAY Regressor (`decay.py`)

Reduces forecast during expected drawdown period:

- **Only active** after bull run peak (not during bull run)
- **Timing from** `avg_days_to_top` and `avg_days_to_bottom`
- **Values clamped** to [-0.08, 0] to prevent negative prices

#### ENSEMBLE_ADJUST (`ensemble.py`)

Post-processing adjustments to Prophet forecast:

- **Pre-halving boost**: Computed from `run_up_pct / 2000` (max 20%)
- **Bull run boost**: Computed from `run_up_pct / 2500` (max 18%)
- **Distribution reduction**: Computed from `predicted_drawdown / 10` (max 8%)

## Signal Scoring

The signal score (-1 to +1) combines:

### Cycle Component (40% default)
- Phase bias (accumulation = +0.6, distribution = -0.4, etc.)
- Pre-halving proximity boost
- Post-halving peak zone reduction

### Technical Component (60% default)
- **RSI** (30%): Oversold (<30) = bullish, overbought (>70) = bearish
- **MACD** (40%): Positive histogram = bullish
- **MA Crossover** (30%): 50-day above 200-day = bullish

### Signal Classification

| Score | Signal |
|-------|--------|
| >= 0.5 | Strong Buy |
| >= 0.2 | Buy |
| >= -0.2 | Hold |
| >= -0.5 | Sell |
| < -0.5 | Strong Sell |

## API Reference

### cycle_features.py

#### `add_cycle_features(df, date_col="ds")`
Add all cycle-related features to a DataFrame.

#### `get_cycle_phase(date, halving_dates=None)`
Return the `CyclePhase` enum for a given date.

#### `create_cycle_regressors_for_prophet(df)`
Create normalized regressors for Prophet's `add_regressor()`.

### signals.py

#### `generate_signals(df, price_col, date_col, include_technicals, cycle_weight, technical_weight)`
Generate buy/sell signals combining cycle and technical analysis.

#### `get_current_signal(df)`
Get the most recent signal with context and recommendation.

#### `backtest_signals(df, initial_capital, position_size)`
Simple backtest comparing signal strategy to buy-and-hold.

### ensemble.py

#### `train_prophet_with_regressors(df, periods, use_cycle_regressors)`
Train Prophet with optional cycle regressors and 4-year seasonality.

#### `train_simple_ensemble(df, periods, prophet_weight, cycle_weight)`
Prophet forecast with cycle-based adjustment multiplier.

#### `evaluate_forecast(df, forecast, holdout_days)`
Evaluate forecast accuracy on holdout period.

---

## Advanced Backtesting

The `backtest.py` module provides professional-grade backtesting with risk management.

### Features

- **Transaction costs**: Fees and slippage
- **Risk management**: Stop loss, take profit, trailing stop
- **Performance metrics**: Sharpe ratio, max drawdown, win rate, profit factor
- **Walk-forward validation**: Train/test splits
- **Parameter optimization**: Grid search for optimal settings

### Quick Start

```python
from src.models import BacktestConfig, run_backtest, print_backtest_report

config = BacktestConfig(
    initial_capital=10000,
    position_size=0.25,        # 25% per trade
    fee_pct=0.1,               # 0.1% trading fee
    slippage_pct=0.05,         # 0.05% slippage
    stop_loss_pct=0.15,        # 15% stop loss
    take_profit_pct=0.50,      # 50% take profit
    trailing_stop_pct=0.20,    # 20% trailing stop
)

result = run_backtest(df_with_signals, config)
print_backtest_report(result)
```

### BacktestConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 10000 | Starting capital in USD |
| `position_size` | 0.25 | Fraction of capital per trade |
| `max_position` | 1.0 | Maximum fraction in BTC |
| `fee_pct` | 0.1 | Trading fee percentage |
| `slippage_pct` | 0.05 | Slippage estimate |
| `stop_loss_pct` | None | Stop loss (e.g., 0.15 = 15%) |
| `take_profit_pct` | None | Take profit trigger |
| `trailing_stop_pct` | None | Trailing stop distance |
| `min_trade_usd` | 100 | Minimum trade size |

### BacktestResult Metrics

**Returns:**
- `total_return_pct`: Total percentage return
- `cagr_pct`: Compound Annual Growth Rate
- `buy_hold_return_pct`: Buy & hold comparison
- `outperformance_pct`: Strategy vs buy & hold

**Risk:**
- `sharpe_ratio`: Risk-adjusted return (higher is better)
- `max_drawdown_pct`: Largest peak-to-trough decline
- `max_drawdown_duration_days`: Longest drawdown period
- `volatility_annual_pct`: Annualized volatility
- `calmar_ratio`: CAGR / Max Drawdown

**Trades:**
- `num_trades`: Total trades executed
- `win_rate_pct`: Percentage of winning trades
- `avg_win_pct` / `avg_loss_pct`: Average win/loss size
- `profit_factor`: Gross profit / Gross loss
- `avg_trade_duration_days`: Average holding period

### Parameter Optimization

```python
from src.models import optimize_parameters

best_params, best_result = optimize_parameters(
    df_with_signals,
    param_grid={
        "position_size": [0.15, 0.25, 0.35],
        "stop_loss_pct": [0.10, 0.15, 0.20],
        "take_profit_pct": [0.30, 0.50, 0.75],
    },
    metric="sharpe_ratio",  # Optimize for Sharpe
)

print(f"Best params: {best_params}")
print(f"Best Sharpe: {best_result.sharpe_ratio:.2f}")
```

### Walk-Forward Validation

Test strategy on out-of-sample data:

```python
from src.models import walk_forward_backtest

results = walk_forward_backtest(
    df_with_signals,
    train_days=365,  # 1 year training
    test_days=90,    # 3 month test
)

for i, result in enumerate(results):
    print(f"Period {i+1}: Return={result.total_return_pct:.1f}%, Sharpe={result.sharpe_ratio:.2f}")
```

### Equity Curve

Access the equity curve for plotting:

```python
result = run_backtest(df, config)
equity = result.equity_curve

# Plot
import matplotlib.pyplot as plt
plt.plot(equity["ds"], equity["equity"])
plt.title("Equity Curve")
plt.show()

# Save
equity.to_csv("equity_curve.csv", index=False)
```

---

## Diagnostics Module

Tools for analyzing regressor behavior and measuring individual contributions.

### Ablation Testing

Test each regressor's contribution by disabling them one at a time:

```python
from src.diagnostics import run_ablation_study, print_ablation_report

# Run full study: baseline, pure Prophet, each regressor ablated
results = run_ablation_study(df, holdout_days=90, forecast_periods=365)

# Print comparison table
print_ablation_report(results)
```

**Output:**
```
ABLATION STUDY REPORT
===============================================================================
--- FORECAST ACCURACY ---
Configuration              MAPE         RMSE   vs Baseline
baseline_all_on           12.34%      8,234
pure_prophet              18.56%     12,456      +6.22%
without_cycle             14.12%      9,567      +1.78%
without_decay             13.89%      9,234      +1.55%
...

--- REGRESSOR CONTRIBUTION RANKING ---
  cycle               MAPE delta: +1.78%  (HELPS)
  decay               MAPE delta: +1.55%  (HELPS)
  ensemble_adjust     MAPE delta: +0.89%  (HELPS)
  ...
```

### Regressor Correlation

Check for redundancy between regressors (high correlation > 0.5):

```python
from src.diagnostics import compute_regressor_correlation

corr_matrix = compute_regressor_correlation(df_with_regressors)
print(corr_matrix)
```

### AblationResult Fields

| Field | Description |
|-------|-------------|
| `config_name` | Name of configuration tested |
| `enabled_regressors` | Dict of which regressors were on |
| `mape` | Mean Absolute Percentage Error |
| `rmse` | Root Mean Square Error |
| `sharpe_ratio` | From backtest |
| `mape_delta` | Difference from baseline (positive = regressor helps) |

---

## Visualization Module

Diagnostic plots for regressor analysis.

### Regressor Timeseries

Plot all regressor values over time with price:

```python
from src.viz import plot_regressor_timeseries

fig = plot_regressor_timeseries(
    df_with_regressors,
    save_path="diagnostics/regressor_timeseries.png"
)
```

### Correlation Heatmap

Visualize correlations between regressors:

```python
from src.viz import plot_regressor_correlation

fig = plot_regressor_correlation(
    df_with_regressors,
    save_path="diagnostics/correlation_heatmap.png"
)
```

### Regressor vs Returns

Scatter plots showing if regressors predict future returns:

```python
from src.viz import plot_regressor_vs_returns

fig = plot_regressor_vs_returns(
    df_with_regressors,
    save_path="diagnostics/regressor_vs_returns.png"
)
```

### Residuals by Phase

Forecast errors colored by cycle phase:

```python
from src.viz import plot_residuals_by_phase

fig = plot_residuals_by_phase(
    df, forecast,
    save_path="diagnostics/residuals_by_phase.png"
)
```

### Full Diagnostic Report

Generate all plots and summary statistics:

```python
from src.viz import create_diagnostic_report

results = create_diagnostic_report(
    df_with_regressors,
    forecast,
    output_dir="diagnostics/"
)

print(f"Generated: {results['plots']}")
print(f"High correlations: {results['stats']['high_correlations']}")
```

---

## Optimization Module

Hyperparameter tuning for regressor parameters using Optuna.

### Installation

```bash
pip install optuna
```

### Basic Usage

```python
from src.optimization import tune_regressors, print_tuning_report

result = tune_regressors(
    df,
    n_trials=50,
    holdout_days=90,
    metric="mape",  # or "sharpe", "combined"
)

print_tuning_report(result)
```

### Tunable Parameters

| Category | Parameters |
|----------|-----------|
| **CYCLE** | `post_halving_peak_day`, `post_halving_spread`, `pre_halving_window` |
| **DOUBLE_TOP** | `first_top_day`, `second_top_day`, `top_window_spread` |
| **DECAY** | `drawdown_start_day`, `drawdown_peak_day`, `drawdown_spread`, `max_decay_effect` |
| **ENSEMBLE_ADJUST** | `pre_halving_boost`, `bull_run_boost`, `dist_reduction` |

### Export Tuned Parameters

Save best parameters to an env file:

```python
from src.optimization import export_params_to_env

export_params_to_env(result.best_params, ".env.tuned")
```

### RegressorParams Defaults

```python
from src.optimization import RegressorParams

params = RegressorParams(
    # CYCLE
    post_halving_peak_day=240,
    post_halving_spread=150,
    pre_halving_window=365,
    # DOUBLE_TOP
    first_top_day=280,
    second_top_day=520,
    top_window_spread=45,
    # DECAY
    drawdown_start_day=365,
    drawdown_peak_day=550,
    drawdown_spread=150,
    max_decay_effect=0.05,
    # ENSEMBLE_ADJUST
    pre_halving_boost=0.15,
    bull_run_boost=0.12,
    dist_reduction=0.03,
)
```

### Optimization Metrics

| Metric | Description |
|--------|-------------|
| `mape` | Minimize Mean Absolute Percentage Error |
| `sharpe` | Maximize Sharpe ratio from backtest |
| `combined` | Weighted combination: MAPE - 0.5 * Sharpe |
