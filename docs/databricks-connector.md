# Databricks Connector

The `DatabricksConnector` class provides a simple interface for querying data from Databricks SQL warehouses.

## Configuration

Set these environment variables (or pass to constructor):

```bash
DATABRICKS_SERVER_HOSTNAME=your-workspace.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_TOKEN=your-access-token
```

## Usage

### Basic Query

```python
from dotenv import load_dotenv
from src.db import DatabricksConnector

load_dotenv()

connector = DatabricksConnector()
df = connector.query("SELECT * FROM my_table LIMIT 10")
```

### Parameterized Query

Use named parameters to safely pass values:

```python
df = connector.query_with_params(
    "SELECT * FROM trades WHERE date > :start_date AND symbol = :symbol",
    {"start_date": "2024-01-01", "symbol": "XBTUSD"}
)
```

### Raw Connection Access

For advanced use cases, access the underlying connection:

```python
with connector.connect() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM my_table")
    count = cursor.fetchone()[0]
    cursor.close()
```

### Constructor Parameters

```python
connector = DatabricksConnector(
    server_hostname="xxx.cloud.databricks.com",  # Optional, uses env var
    http_path="/sql/1.0/warehouses/xxx",         # Optional, uses env var
    access_token="dapi...",                       # Optional, uses env var
)
```

## API Reference

### `DatabricksConnector`

#### `__init__(server_hostname, http_path, access_token)`

Initialize the connector. All parameters are optional and default to environment variables.

#### `query(sql_query: str) -> pd.DataFrame`

Execute SQL and return results as a pandas DataFrame.

#### `query_with_params(sql_query: str, params: dict) -> pd.DataFrame`

Execute parameterized SQL with named parameters (`:param_name`).

#### `connect() -> Connection`

Context manager yielding a raw Databricks SQL connection.
