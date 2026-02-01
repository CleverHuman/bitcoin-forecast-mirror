# BTC Bitcoin Forecast

Bitcoin price forecasting using Prophet with cycle-aware signals for buy/sell timing.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Databricks credentials:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your Databricks SQL Warehouse details:
   - `DATABRICKS_SERVER_HOSTNAME` - Your workspace hostname
   - `DATABRICKS_HTTP_PATH` - SQL Warehouse HTTP path
   - `DATABRICKS_TOKEN` - Personal Access Token

## Usage

### Scripts

```bash
# Basic forecast (recent data, faster)
python forecast.py

# Forecast with halving cycle analysis
python forecast_cycle.py

# Full signals - cycle-aware forecast + buy/sell recommendations
python forecast_signals.py
```

### Generate Buy/Sell Signals

```python
from dotenv import load_dotenv
from src.db import DatabricksConnector
from src.models import generate_signals, get_current_signal

load_dotenv()

connector = DatabricksConnector()
df = connector.query("SELECT date as ds, avg_price as y FROM ...")

# Generate signals (cycle position + RSI/MACD/MA)
df = generate_signals(df, cycle_weight=0.4, technical_weight=0.6)

# Get current recommendation
current = get_current_signal(df)
print(f"Signal: {current['signal']}")           # strong_buy, buy, hold, sell, strong_sell
print(f"Phase: {current['cycle_phase']}")       # accumulation, pre_halving_runup, etc.
print(f"Recommendation: {current['recommendation']}")
```

### Cycle-Aware Forecast

```python
from src.models import train_simple_ensemble

# Prophet + cycle position adjustments
forecast = train_simple_ensemble(df, periods=365)
```

## How It Works

### Cycle Phases

The 4-year halving cycle is divided into phases:

| Phase | Timing | Signal Bias |
|-------|--------|-------------|
| Accumulation | -545 to -180 days | Bullish |
| Pre-Halving Run-up | -180 to 0 days | Bullish |
| Post-Halving Consolidation | 0 to 120 days | Neutral |
| Bull Run | 120 to 365 days | Neutral |
| Distribution | 365 to 545 days | Bearish |
| Drawdown | 545+ days | Accumulate |

### Signal Scoring

Combines cycle position (40%) with technical indicators (60%):
- RSI: Oversold/overbought
- MACD: Momentum
- MA Crossover: 50/200-day trend

## Documentation

See [docs/](docs/index.md) for detailed API documentation:

- [Databricks Connector](docs/databricks-connector.md) - Query data
- [Halving Metrics](docs/halving-metrics.md) - Cycle analysis
- [Models](docs/models.md) - Signals and ensemble forecasting

## Project Structure

```
bh_bitcoin_forecast/
├── docs/                     # Documentation
├── src/
│   ├── db/
│   │   └── connector.py      # Databricks connector
│   ├── metrics/
│   │   └── halving.py        # Cycle metrics
│   └── models/
│       ├── cycle_features.py # Cycle position encoding
│       ├── signals.py        # Buy/sell signals
│       └── ensemble.py       # Prophet + cycle ensemble
├── forecast.py               # Basic forecast
├── forecast_cycle.py         # With halving metrics
├── forecast_signals.py       # Full signals + recommendations
└── requirements.txt
```
