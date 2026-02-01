# BTC Bitcoin Forecast Documentation

A Python library for Bitcoin price forecasting using Prophet with data from Databricks.

## Modules

| Module | Description |
|--------|-------------|
| [src.db](databricks-connector.md) | Databricks SQL connector for querying data |
| [src.metrics](halving-metrics.md) | Halving cycle metrics and analysis |

## Quick Start

```python
from dotenv import load_dotenv
from src.db import DatabricksConnector
from src.metrics import compute_cycle_metrics, compute_halving_averages

load_dotenv()

# Fetch data from Databricks
connector = DatabricksConnector()
df = connector.query("""
    SELECT date as ds, avg_price as y
    FROM default.bitmex_trade_daily_stats
    WHERE symbol LIKE '%XBTUSD%'
    ORDER BY date
""")

# Compute halving cycle metrics
cycle_metrics = compute_cycle_metrics(df)
averages = compute_halving_averages(cycle_metrics=cycle_metrics)

print(f"Avg run-up: {averages.run_up_pct:.1f}% over {averages.run_up_days:.0f} days")
print(f"Avg drawdown: {averages.drawdown_pct:.1f}% over {averages.drawdown_days:.0f} days")
```

## Scripts

| Script | Description |
|--------|-------------|
| `forecast.py` | Basic BTC forecast with Prophet |
| `forecast_cycle.py` | Forecast with halving cycle metrics integration |

## Setup

See the main [README](../README.md) for installation and configuration.
