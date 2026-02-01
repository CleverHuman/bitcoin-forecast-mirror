# BTC Forecast Project Notes

## Status: Active Development

---

## What's Working Well

### Cycle Phases ✓
The 6-phase cycle model is working well:
- **Accumulation** (-545 to -180 days): Post-drawdown, buy zone
- **Pre-Halving Run-up** (-180 to 0 days): Price rises into halving
- **Post-Halving Consolidation** (0 to 120 days): Choppy digestion
- **Bull Run** (120 to 365 days): Parabolic phase
- **Distribution** (365 to 545 days): Take profits zone
- **Drawdown** (545+ days): Bear market, accumulation opportunity

### Signal Strategy (Data-Driven)
**Goal**: Buy before halving, sell after halving using HISTORICAL AVERAGES

**How it works:**
Uses `halving_averages` from `compute_halving_averages()`:
- `avg_days_before_halving_to_low` → BUY window (cycle bottom)
- `avg_days_after_halving_to_high` → SELL window (cycle peak)

**BUY Zones:**
- Around historical cycle bottom (avg_days_to_low ± buffer)
- Late drawdown (bear market, DCA)

**HOLD Zones:**
- Pre-halving run-up (<90 days before) - already running
- Post-halving consolidation - wait for peak

**SELL Zones:**
- Around historical cycle peak (avg_days_to_peak ± buffer)
- After peak window (distribution) - MUST SELL

Signal weighting: Cycle (60%) + Technicals (40%)

### Advanced Backtesting
- Stop loss, take profit, trailing stops
- Transaction costs (fees + slippage)
- Full metrics: Sharpe, max drawdown, win rate, profit factor
- Parameter optimization
- Walk-forward validation

---

## Architecture Decisions

### Why Two Forecast Scripts?
1. `forecast.py` - Uses recent data (2022+), better for short-term
2. `forecast_cycle.py` - Uses full history, better for cycle analysis
3. `forecast_signals.py` - Full signals + backtesting

**Reason**: More data dilutes recent signal in Prophet. Recent data = better short-term accuracy.

### Library Structure
```
src/
├── db/          # Databricks connector
├── metrics/     # Halving cycle metrics
└── models/      # Signals, ensemble, backtest
```

---

## TODO / Ideas

- [ ] Add more technical indicators (Bollinger Bands, volume)
- [ ] Implement DCA (Dollar Cost Averaging) strategy in backtest
- [ ] Add portfolio rebalancing
- [ ] Web dashboard for signals
- [ ] Alert system (email/SMS when signal changes)
- [ ] Compare against other ML models (XGBoost, LSTM)

---

## Session Log

### 2026-02-01
- Scaffolded project with Databricks connector
- Created halving metrics module
- Built cycle-aware signal generation
- Implemented advanced backtesting with risk management
- Cycle phases validated as working well
- Fixed plot date formatting issues

---

## Key Findings

### Cycle Metrics (Historical)
*To be updated with actual numbers after running*
- Avg run-up: ~X% over ~Y days
- Avg drawdown: ~X% over ~Y days

### Backtest Results
*To be updated after running*
- Strategy vs Buy & Hold: TBD
- Sharpe Ratio: TBD
- Max Drawdown: TBD
