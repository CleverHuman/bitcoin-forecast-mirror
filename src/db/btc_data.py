"""BTC price data fetching from Databricks."""

import pandas as pd

from .connector import DatabricksConnector


def fetch_btc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch BTC trade data from Databricks.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with columns 'ds' (date) and 'y' (avg_price).
    """
    connector = DatabricksConnector()

    sql = f"""
        SELECT date, avg_price
        FROM default.bitmex_trade_daily_stats
        WHERE symbol LIKE '%XBTUSD%'
          AND side = 'Sell'
          AND to_date(date) > '{start_date}'
          AND to_date(date) < '{end_date}'
        ORDER BY to_date(date) DESC
    """

    df = connector.query(sql)
    df = df.rename(columns={"date": "ds", "avg_price": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    return df
