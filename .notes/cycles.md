# Halving Cycle Requirements

## Original Requirements ✓ IMPLEMENTED

For each halving cycle:
1. Use halving dates ✓
2. Use actual BTC prices ✓
3. Measure:
   - 📈 run-up before halving (low → pre-halving high) ✓
   - 📉 drawdown after halving (high → post-halving low) ✓
   - ⏱️ duration (days up / days down) ✓
4. Average across cycles ✓
5. Use those averages to:
   - parameterise Prophet windows ✓
   - sanity-check Prophet output ✓

This keeps your model data-driven, not narrative-driven.

---

## Implementation

### Location
- `src/metrics/halving.py` - Core metrics computation
- `src/models/cycle_features.py` - Cycle phase encoding
- `src/models/signals.py` - Signal generation using cycles

### Key Functions
```python
from src.metrics import compute_cycle_metrics, compute_halving_averages
from src.models import add_cycle_features, get_cycle_phase
```

### Cycle Phases (Working Well)
| Phase | Days from Halving | Bias |
|-------|------------------|------|
| Accumulation | -545 to -180 | Bullish |
| Pre-Halving Run-up | -180 to 0 | Bullish |
| Post-Halving Consolidation | 0 to 120 | Neutral |
| Bull Run | 120 to 365 | Neutral |
| Distribution | 365 to 545 | Bearish |
| Drawdown | 545+ | Accumulate |

---

## Next Steps

- [ ] Fine-tune phase boundaries based on more historical data
- [ ] Add confidence intervals to cycle predictions
- [ ] Weight recent cycles more heavily than older ones
