# BTC Bitcoin Forecast Documentation

A Python library for Bitcoin price forecasting using Prophet with data from Databricks.

## Modules

| Module | Description |
|--------|-------------|
| [src.db](databricks-connector.md) | Databricks SQL connector for querying data |
| [src.metrics](halving-metrics.md) | Halving cycle metrics and analysis |
| [src.models](models.md) | Cycle-aware models and buy/sell signals |

## Quick Start

### Basic Forecast

```python
from dotenv import load_dotenv
from src.db import DatabricksConnector
from src.metrics import compute_cycle_metrics, compute_halving_averages

load_dotenv()

connector = DatabricksConnector()
df = connector.query("SELECT date as ds, avg_price as y FROM ...")

cycle_metrics = compute_cycle_metrics(df)
averages = compute_halving_averages(cycle_metrics=cycle_metrics)
print(f"Avg run-up: {averages.run_up_pct:.1f}%")
```

### Generate Buy/Sell Signals

```python
from src.models import generate_signals, get_current_signal

df = generate_signals(df, cycle_weight=0.4, technical_weight=0.6)

current = get_current_signal(df)
print(f"Signal: {current['signal']}")
print(f"Recommendation: {current['recommendation']}")
```

### Cycle-Aware Forecast

```python
from src.models import train_simple_ensemble

forecast = train_simple_ensemble(df, periods=365)
# Uses Prophet + cycle position adjustments
```

## Scripts

| Script | Description |
|--------|-------------|
| `forecast.py` | Basic Prophet forecast (recent data) |
| `forecast_cycle.py` | Forecast with halving metrics analysis |
| `forecast_signals.py` | **Full signals** - cycle-aware forecast + buy/sell recommendations |

## Architecture

```
src/
├── db/
│   └── connector.py      # Databricks SQL connector
├── metrics/
│   └── halving.py        # Halving cycle analysis
└── models/
    ├── cycle_features.py # Cycle position encoding
    ├── signals.py        # Buy/sell signal generation
    └── ensemble.py       # Prophet + cycle ensemble
```

## Setup

See the main [README](../README.md) for installation and configuration.
