#!/usr/bin/env python3
"""
Standalone script to output halving-cycle metrics as a table.

Shows for each cycle: when lowest/highest prices occur, days before/after
halving, run-up and drawdown percentages and durations.

Usage:
  python -m src.metrics.print_halving_table
  python -m src.metrics.print_halving_table --csv historical_data.csv
  python -m src.metrics.print_halving_table --csv historical_data.csv --output halving_cycle_metrics.csv
"""

import argparse
from datetime import datetime

import pandas as pd

from src.db import DatabricksConnector
from src.metrics import (
    HALVING_DATES,
    compute_cycle_metrics,
    compute_halving_averages,
)


def fetch_btc_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch BTC trade data from Databricks."""
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


def load_csv(path: str) -> pd.DataFrame:
    """Load DataFrame from CSV; expect 'ds'/'y' or 'date'/'avg_price'."""
    df = pd.read_csv(path)
    df = df.rename(columns={"date": "ds", "avg_price": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    if "y" not in df.columns:
        raise ValueError(f"CSV must have price column 'y' or 'avg_price': {path}")
    return df[["ds", "y"]].dropna()


def format_table(cycle_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a display DataFrame with key columns for the table."""
    if cycle_metrics.empty:
        return cycle_metrics

    display = pd.DataFrame()
    display["halving_date"] = cycle_metrics["halving_date"].dt.strftime("%Y-%m-%d")

    # Pre-halving low: date, price, days before halving
    display["pre_low_date"] = cycle_metrics["pre_low_date"].dt.strftime("%Y-%m-%d")
    display["pre_low_price"] = cycle_metrics["pre_low_price"].round(0)
    display["days_before_halving_to_low"] = cycle_metrics["days_before_halving_to_low"]

    # Pre-halving high
    display["pre_high_date"] = cycle_metrics["pre_high_date"].dt.strftime("%Y-%m-%d")
    display["pre_high_price"] = cycle_metrics["pre_high_price"].round(0)
    display["days_before_halving_to_high"] = cycle_metrics["days_before_halving_to_high"]

    # Run-up
    display["run_up_pct"] = cycle_metrics["run_up_pct"].round(1)
    display["run_up_days"] = cycle_metrics["run_up_days"].astype(int)

    # Post-halving high: date, price, days after halving
    display["post_high_date"] = cycle_metrics["post_high_date"].dt.strftime("%Y-%m-%d")
    display["post_high_price"] = cycle_metrics["post_high_price"].round(0)
    display["days_after_halving_to_high"] = cycle_metrics["days_after_halving_to_high"]

    # Post-halving low
    display["post_low_date"] = cycle_metrics["post_low_date"].dt.strftime("%Y-%m-%d")
    display["post_low_price"] = cycle_metrics["post_low_price"].round(0)
    display["days_after_halving_to_low"] = cycle_metrics["days_after_halving_to_low"]

    # Drawdown
    display["drawdown_pct"] = cycle_metrics["drawdown_pct"].round(1)
    display["drawdown_days"] = cycle_metrics["drawdown_days"].astype(int)

    return display


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Output halving-cycle metrics (low/high dates, days before/after halving) as a table."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to CSV with 'ds' and 'y' (or 'date' and 'avg_price'). If omitted, fetch from Databricks.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for Databricks fetch (ignored if --csv is set).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="halving_cycle_metrics.csv",
        help="Path to write the halving cycle table CSV (default: halving_cycle_metrics.csv).",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=90,
        metavar="DAYS",
        help="Minimum days between key prices and halving boundaries to avoid overlap (default: 90).",
    )
    args = parser.parse_args()

    if args.csv:
        print(f"Loading data from {args.csv}...")
        df = load_csv(args.csv)
    else:
        end_date = datetime.now().strftime("%Y-%m-%d")
        print(f"Fetching BTC data from {args.start} to {end_date}...")
        df = fetch_btc_data(args.start, end_date)

    print(f"Loaded {len(df)} rows.\n")

    print("Halving dates used:", [d.strftime("%Y-%m-%d") for d in HALVING_DATES])
    print(f"Buffer: {args.buffer} days (exclude key prices within this range of halving boundaries).")
    print()

    cycle_metrics = compute_cycle_metrics(df, buffer_days=args.buffer)
    if cycle_metrics.empty:
        print("No halving cycles computed (insufficient data).")
        return

    display = format_table(cycle_metrics)
    display.to_csv(args.output, index=False)
    print(f"Wrote halving cycle table to {args.output}")

    print("Per-cycle metrics: lowest/highest prices and days before/after halving")
    print("=" * 120)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 20)
    print(display.to_string(index=False))
    print()

    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print(
        f"Averages (n={averages.n_cycles}): "
        f"run_up {averages.run_up_pct:.1f}% in {averages.run_up_days:.0f} days, "
        f"drawdown {averages.drawdown_pct:.1f}% in {averages.drawdown_days:.0f} days."
    )


if __name__ == "__main__":
    main()
