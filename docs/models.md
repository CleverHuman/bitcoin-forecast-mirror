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
)
```

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
