"""Databricks SQL connector for querying data from Databricks SQL warehouses."""

import os
from contextlib import contextmanager
from typing import Generator

import pandas as pd
from databricks import sql
from databricks.sql.client import Connection


class DatabricksConnector:
    """A reusable connector for Databricks SQL warehouses.

    Usage:
        connector = DatabricksConnector()
        df = connector.query("SELECT * FROM my_table LIMIT 10")

        # Or use as context manager for manual connection control:
        with connector.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM my_table")
            results = cursor.fetchall()
    """

    def __init__(
        self,
        server_hostname: str | None = None,
        http_path: str | None = None,
        access_token: str | None = None,
    ):
        """Initialize the connector.

        Args:
            server_hostname: Databricks workspace hostname (e.g., "xxx.cloud.databricks.com")
            http_path: SQL warehouse HTTP path (e.g., "/sql/1.0/warehouses/xxx")
            access_token: Databricks personal access token

        If not provided, values are read from environment variables:
            DATABRICKS_SERVER_HOSTNAME, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
        """
        self.server_hostname = server_hostname or os.environ.get("DATABRICKS_SERVER_HOSTNAME")
        self.http_path = http_path or os.environ.get("DATABRICKS_HTTP_PATH")
        self.access_token = access_token or os.environ.get("DATABRICKS_TOKEN")

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        missing = []
        if not self.server_hostname:
            missing.append("DATABRICKS_SERVER_HOSTNAME")
        if not self.http_path:
            missing.append("DATABRICKS_HTTP_PATH")
        if not self.access_token:
            missing.append("DATABRICKS_TOKEN")

        if missing:
            raise ValueError(
                f"Missing required Databricks configuration: {', '.join(missing)}. "
                "Set these as environment variables or pass them to the constructor."
            )

    @contextmanager
    def connect(self) -> Generator[Connection, None, None]:
        """Create a connection to Databricks SQL warehouse.

        Yields:
            A Databricks SQL connection object.
        """
        conn = sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        )
        try:
            yield conn
        finally:
            conn.close()

    def query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a pandas DataFrame.

        Args:
            sql_query: The SQL query to execute.

        Returns:
            A pandas DataFrame containing the query results.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            df = cursor.fetchall_arrow().to_pandas()
            cursor.close()
            return df

    def query_with_params(self, sql_query: str, params: dict) -> pd.DataFrame:
        """Execute a parameterized SQL query.

        Args:
            sql_query: SQL query with named parameters (e.g., "SELECT * FROM t WHERE id = :id")
            params: Dictionary of parameter names to values.

        Returns:
            A pandas DataFrame containing the query results.
        """
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query, params)
            df = cursor.fetchall_arrow().to_pandas()
            cursor.close()
            return df
