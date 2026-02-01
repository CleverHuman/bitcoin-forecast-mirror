# Drawdown Prediction Strategies

## Current Approach

Using historical averages:
- `avg_drawdown_days`: days from cycle top to bottom
- `avg_drawdown_pct`: percentage drop from top to bottom
- Prediction: `bottom_date = top_date + avg_drawdown_days`

### Historical Data (from your cycles)
| Cycle | Days to Top | Days Top→Bottom | Drawdown % |
|-------|-------------|-----------------|------------|
| 2016  | 526         | ?               | ?          |
| 2020  | 547         | ?               | ?          |
| 2024  | 534         | ?               | ?          |

---

## Current Prediction Issue

```
Last TOP:           2025-10-05 @ $123,942
Avg drawdown:       65.2% over 286 days
Predicted BOTTOM:   2026-07-18
Predicted price:    $43,122
```

**Problem**: 65% drawdown seems too aggressive for current market structure.

### Previous ATH as Support
| Cycle | Previous ATH | Drawdown Low | Held as Support? |
|-------|-------------|--------------|------------------|
| 2017  | ~$1,200     | $3,200       | Yes (bounced above) |
| 2021  | ~$20,000    | $15,500      | Briefly broke, recovered |
| 2024  | ~$69,000    | ?            | Likely support zone |

**Thesis**: With ETFs and institutional adoption, previous ATH ($69k) should act as strong support. A more realistic bottom might be:
- Conservative: $60-70k (previous ATH zone)
- Moderate: $50-60k (30-40% drawdown from $124k)
- Aggressive: $40-50k (historical avg drawdown)

---

## Ideas for Improvement

### 1. Previous ATH as Floor
- Use previous cycle ATH as minimum support level
- Prediction: `max(predicted_bottom_price, previous_ath * 0.9)`
- Rationale: institutional buyers accumulate at previous ATH

### 2. Logarithmic Drawdown Decay

**Hypothesis**: As price increases, drawdown percentage decreases along a curve.

```
Cycle 1 (2013): ~85% drawdown
Cycle 2 (2017): ~84% drawdown
Cycle 3 (2021): ~77% drawdown
Cycle 4 (2024): ~65%? (if pattern holds, could be lower)
```

**Approach**:
- Calculate drawdowns on log scale: `log(top) - log(bottom)`
- Fit a decay curve across cycles
- Extrapolate for future cycles
- Formula: `drawdown_pct = a * exp(-b * cycle_num) + c`
  - Where `c` is the asymptotic floor (maybe 30-40%?)

**Why log scale?**
- Percentage moves are more meaningful on log scale
- A 50% drop from $100k is different structurally than 50% from $1k
- Log returns are additive and more stationary

### 3. Volatility Score

**Idea**: Track volatility over time to predict drawdown severity.

**Metrics to consider**:
- **Historical volatility**: std dev of daily log returns (30/90/365 day)
- **Realized volatility**: actual vol during run-up phase
- **Volatility decay**: is vol decreasing cycle over cycle?

**Volatility-adjusted drawdown**:
```python
# Higher vol during run-up → expect larger drawdown
vol_run_up = std(log_returns) during run-up phase
vol_ratio = vol_run_up / historical_avg_vol
adjusted_drawdown = base_drawdown * vol_ratio
```

**Volatility as maturity indicator**:
| Cycle | Avg Daily Vol | Drawdown |
|-------|---------------|----------|
| 2013  | ~6%           | 85%      |
| 2017  | ~4%           | 84%      |
| 2021  | ~3.5%         | 77%      |
| 2024  | ~2.5%?        | ?        |

As volatility decreases (market matures), drawdowns should moderate.

### 4. Weighted Averages
- Weight recent cycles more heavily than older ones
- Example: 50% weight on most recent, 30% on second, 20% on third
- Rationale: market structure evolves (ETFs, institutions, etc.)

### 2. Drawdown Phases
Instead of predicting a single bottom date, predict phases:
- **Initial crash**: 30-60 days after top, sharp drop
- **Dead cat bounce**: temporary recovery
- **Capitulation**: final flush to bottom
- **Accumulation zone**: range before next run-up

### 3. Technical Confirmation Signals
Combine timing prediction with:
- RSI < 30 (oversold)
- Price below 200-week MA
- MVRV ratio < 1 (on-chain)
- Puell Multiple < 0.5

### 4. Drawdown Magnitude Correlation
- Does a larger run-up lead to a larger drawdown?
- Does a faster run-up lead to a faster drawdown?
- Analyze correlation between run-up metrics and drawdown metrics

### 5. External Factors
- Macro conditions (Fed policy, liquidity)
- Stock market correlation during drawdowns
- Previous cycle: did macro extend/shorten the drawdown?

### 6. Confidence Intervals
- Instead of single date, provide range: "Bottom likely between X and Y"
- Use standard deviation of historical drawdown_days
- Example: avg 370 days, std 50 days → 68% chance between 320-420 days

---

## Questions to Explore

1. How consistent is `drawdown_days` across cycles?
2. Is the bottom defined correctly? (absolute low vs. "accumulation zone")
3. Should we predict a range instead of a single date?
4. What on-chain or technical indicators confirm we're near the bottom?

---

## Next Steps

- [ ] Calculate std dev of drawdown_days for confidence intervals
- [ ] Analyze correlation between run-up and drawdown metrics
- [ ] Add technical indicator confirmation to bottom prediction
- [ ] Consider multi-phase drawdown model
- [ ] Implement log-scale drawdown calculation
- [ ] Fit exponential decay curve to drawdown percentages
- [ ] Add volatility calculation (30/90/365 day rolling)
- [ ] Track volatility decay across cycles
- [ ] Use previous ATH as price floor in predictions

---

## Proposed: `src/metrics/decay.py`

New module for decay curve fitting and prediction.

```python
"""Exponential decay fitting for drawdown/volatility prediction."""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass

@dataclass
class DecayPrediction:
    predicted_value: float
    floor: float          # asymptotic minimum (c parameter)
    decay_rate: float     # how fast it decays (b parameter)
    r_squared: float      # fit quality
    confidence_low: float
    confidence_high: float

def exp_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * np.exp(-b * x) + c

def fit_decay_curve(
    cycle_nums: list[int],
    values: list[float],
    floor_bounds: tuple = (0.2, 0.5),  # min/max for asymptote
) -> tuple[np.ndarray, float]:
    """Fit exponential decay to historical values.

    Returns: (params [a, b, c], r_squared)
    """
    params, _ = curve_fit(
        exp_decay,
        cycle_nums,
        values,
        p0=[0.5, 0.1, 0.35],
        bounds=([0, 0, floor_bounds[0]], [1, 1, floor_bounds[1]])
    )

    # Calculate R²
    predicted = exp_decay(np.array(cycle_nums), *params)
    ss_res = np.sum((np.array(values) - predicted) ** 2)
    ss_tot = np.sum((np.array(values) - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return params, r_squared

def predict_drawdown(
    cycle_metrics: pd.DataFrame,
    target_cycle: int,
) -> DecayPrediction:
    """Predict drawdown for a future cycle using decay curve."""

    cycle_nums = list(range(1, len(cycle_metrics) + 1))
    drawdowns = (cycle_metrics["drawdown_pct"] / 100).tolist()

    params, r_sq = fit_decay_curve(cycle_nums, drawdowns)
    predicted = exp_decay(target_cycle, *params)

    # Confidence interval from residual std error
    residuals = np.array(drawdowns) - exp_decay(np.array(cycle_nums), *params)
    std_err = np.std(residuals)

    return DecayPrediction(
        predicted_value=predicted,
        floor=params[2],
        decay_rate=params[1],
        r_squared=r_sq,
        confidence_low=max(predicted - 2*std_err, params[2]),
        confidence_high=predicted + 2*std_err,
    )
```

### Usage
```python
from src.metrics.decay import predict_drawdown

pred = predict_drawdown(cycle_metrics, target_cycle=5)
print(f"Predicted drawdown: {pred.predicted_value:.1%}")
print(f"Range: {pred.confidence_low:.1%} - {pred.confidence_high:.1%}")
print(f"Asymptotic floor: {pred.floor:.1%}")
```

