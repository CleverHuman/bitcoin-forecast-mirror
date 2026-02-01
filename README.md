# BTC Bitcoin Forecast

Bitcoin price forecasting using Prophet with data from Databricks.

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
   - `DATABRICKS_SERVER_HOSTNAME` - Your workspace hostname (e.g., `xxx.cloud.databricks.com`)
   - `DATABRICKS_HTTP_PATH` - SQL Warehouse HTTP path (found in Connection Details)
   - `DATABRICKS_TOKEN` - Personal Access Token (generate in User Settings > Developer)

### Using Databricks serverless SQL

The same connector works with **serverless** and classic SQL warehouses. No code changes are needed.

1. In Databricks: **SQL** > **SQL Warehouses** > **Create SQL warehouse**.
2. Choose **Serverless** (recommended: instant start, elastic scaling, lower cost).
3. Open the warehouse > **Connection details** tab.
4. Copy **Server hostname** and **HTTP path** into your `.env`.
5. Use the same PAT in `DATABRICKS_TOKEN`.

## Usage

### Running the forecast

```bash
# Basic forecast
python forecast.py

# Forecast with halving cycle analysis
python forecast_cycle.py
```

Outputs:
- `historical_data.csv` - Raw historical price data
- `forecasted_data.csv` - Model predictions

### Using the library

```python
from dotenv import load_dotenv
from src.db import DatabricksConnector
from src.metrics import compute_cycle_metrics, compute_halving_averages

load_dotenv()

# Fetch data
connector = DatabricksConnector()
df = connector.query("SELECT date as ds, avg_price as y FROM my_table")

# Analyze halving cycles
metrics = compute_cycle_metrics(df)
averages = compute_halving_averages(cycle_metrics=metrics)
print(f"Avg run-up: {averages.run_up_pct:.1f}%")
```

## Documentation

See the [docs/](docs/index.md) folder for detailed API documentation:

- [Databricks Connector](docs/databricks-connector.md) - Query data from Databricks
- [Halving Metrics](docs/halving-metrics.md) - Analyze BTC halving cycles

## Project Structure

```
bh_bitcoin_forecast/
├── docs/                     # Documentation
│   ├── index.md
│   ├── databricks-connector.md
│   └── halving-metrics.md
├── src/                      # Library code
│   ├── db/
│   │   └── connector.py      # DatabricksConnector
│   └── metrics/
│       └── halving.py        # Halving cycle analysis
├── forecast.py               # Basic forecast script
├── forecast_cycle.py         # Forecast with halving metrics
├── .env.example              # Credentials template
├── .gitignore
├── requirements.txt
└── README.md
```
