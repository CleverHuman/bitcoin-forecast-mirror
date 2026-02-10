#!/usr/bin/env python3
"""Compare trading strategies and pick one for paper/live trading.

Run backtests for all strategies, then use the results to decide which to use.
Paper trade the live bot with: python live_trader.py --capital 50000

Strategies compared:
  - Combined: cycle + forecast + technicals (closest to live trader logic)
  - Halving Cycle: cycle + technicals
  - Forecast-Based: simple forecast direction
  - Forecast Momentum: forecast momentum + thresholds
  - Buy & Hold: baseline benchmark
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables before imports
load_dotenv()

from src.db import fetch_btc_data
from src.metrics import compute_cycle_metrics, compute_halving_averages
from src.forecasting import ProphetCycleForecaster
from src.backtesting import (
    BacktestConfig,
    compare_strategies,
    print_comparison,
    print_metrics_report,
    print_trade_log,
    build_trade_log,
)
from src.viz import save_backtest_chart
from src.backtesting.strategies import (
    CombinedStrategy,
    CycleSignalStrategy,
    ForecastBasedStrategy,
    ForecastMomentumStrategy,
    BuyAndHoldStrategy,
)


def main():
    parser = argparse.ArgumentParser(
        description="Backtest strategies to decide which to use for paper/live trading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_strategies.py
  python backtest_strategies.py --start 2020-01-01 --capital 50000
  python backtest_strategies.py --plot          # also save chart to reports/backtest_chart.png
  python backtest_strategies.py --no-save

Then run paper trading:
  python live_trader.py --capital 50000
""",
    )
    parser.add_argument(
        "--start",
        default="2012-01-01",
        help="Backtest start date (default: 2012-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Backtest end date (default: today)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=float(os.getenv("INITIAL_CAPITAL", "10000")),
        help="Initial capital for backtest (default: from .env or 10000)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=float(os.getenv("BACKTEST_POSITION_SIZE", "0.25")),
        help="Fraction of capital per trade (default: from .env or 0.25)",
    )
    parser.add_argument(
        "--max-position",
        type=float,
        default=float(os.getenv("BACKTEST_MAX_POSITION", "1.0")),
        help="Max fraction of capital in BTC (default: from .env or 1.0)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save summary to reports/backtest_summary.csv",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save backtest chart to reports/backtest_chart.png",
    )
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    backtest_start = pd.Timestamp(args.start)

    print("=" * 70)
    print("BTC STRATEGY COMPARISON (NO LOOK-AHEAD BIAS)")
    print("=" * 70)

    # 1. Load ALL data (need historical for training, future for testing)
    print("\n[1/5] Loading BTC data...")
    # Load from 2015 for training history, regardless of backtest start
    df_all = fetch_btc_data(start_date="2015-01-01", end_date=end_date)
    print(f"  Total data: {len(df_all)} days ({df_all['ds'].min().date()} to {df_all['ds'].max().date()})")

    # 2. Split into train (before backtest) and test (backtest period)
    print("\n[2/5] Splitting data (no look-ahead bias)...")
    df_train = df_all[df_all["ds"] < backtest_start].copy()
    df_test = df_all[df_all["ds"] >= backtest_start].copy()
    print(f"  Training data: {len(df_train)} days (up to {args.start})")
    print(f"  Backtest data: {len(df_test)} days ({df_test['ds'].min().date()} to {df_test['ds'].max().date()})")

    if len(df_train) < 365:
        print(f"  WARNING: Only {len(df_train)} days of training data. Consider earlier start date.")

    # 3. Compute cycle metrics using ONLY training data
    print("\n[3/5] Computing cycle metrics (training data only)...")
    cycle_metrics = compute_cycle_metrics(df_train)
    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print(f"  Analyzed {averages.n_cycles} complete halving cycles")
    if averages.n_cycles > 0:
        print(f"  Avg run-up: {averages.run_up_pct:.1f}%")
        print(f"  Avg days to top: {averages.avg_days_to_top:.0f}")

    # 4. Generate forecast using ONLY training data
    print("\n[4/5] Training forecast (on pre-backtest data only)...")
    forecaster = ProphetCycleForecaster(
        halving_averages=averages,
        cycle_metrics=cycle_metrics,
    )
    # Forecast far enough to cover the entire backtest period
    forecast_periods = len(df_test) + 365
    result = forecaster.fit_predict(df_train, periods=forecast_periods)
    forecast = result.forecast
    print(f"  Forecast trained on data before {args.start}")
    print(f"  Forecast covers: {forecast['ds'].min().date()} to {forecast['ds'].max().date()}")

    # Use test data for backtesting
    df = df_test

    # 5. Compare strategies
    print("\n[5/5] Running backtests...")

    config = BacktestConfig(
        initial_capital=args.capital,
        position_size=args.position_size,
        max_position=args.max_position,
        fee_pct=0.1,
        slippage_pct=0.05,
    )
    print(f"  Config: capital=${args.capital:,.0f}, position_size={args.position_size:.0%}, max_position={args.max_position:.0%}")

    strategies = [
        CombinedStrategy(
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
            cycle_weight=0.30,
            forecast_weight=0.40,
            technical_weight=0.30,
        ),
        CycleSignalStrategy(
            cycle_weight=0.6,
            technical_weight=0.4,
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
        ),
        ForecastBasedStrategy(
            forecast_col="yhat_ensemble",
            threshold_pct=5.0,
            lookforward_days=30,
        ),
        ForecastMomentumStrategy(
            forecast_col="yhat_ensemble",
            lookforward_days=30,
            min_upside_pct=5.0,
            min_trade_interval_days=7,
        ),
        BuyAndHoldStrategy(),
    ]

    comparison = compare_strategies(
        df,
        strategies=strategies,
        config=config,
        forecast=forecast,
    )

    # Print results
    print_comparison(comparison)

    # Save summary for records
    if not args.no_save:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        out_path = reports_dir / "backtest_summary.csv"
        comparison.summary.to_csv(out_path, index=False)
        print(f"\nSummary saved to {out_path}")

    # Detailed report and trade log for best by return
    best_name = comparison.summary.iloc[0]["Strategy"]
    best_result = comparison.results[best_name]
    print("\n" + "=" * 70)
    print(f"DETAILED: {best_name}")
    print("=" * 70)
    print_metrics_report(best_result, best_name)

    # When each buy/sell happened and profit per trade
    print_trade_log(best_result, best_name)

    # Save trade log to CSV
    if not args.no_save and best_result.trades:
        log = build_trade_log(best_result.trades)
        trades_df = pd.DataFrame(log)
        trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")
        trades_path = reports_dir / "backtest_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"Trade log saved to {trades_path}")

    # Save backtest chart (equity curves + price + buy/sell markers)
    if args.plot:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        chart_path = save_backtest_chart(
            comparison,
            initial_capital=args.capital,
            path=reports_dir / "backtest_chart.png",
            best_name=best_name,
        )
        print(f"Chart saved to {chart_path}")

    print("\n" + "=" * 70)
    print("NEXT: Paper trade with live prices")
    print("=" * 70)
    print("  python live_trader.py --capital", int(args.capital))
    print("  (Uses forecast + tactical logic; closest backtest analogue: Combined)")


if __name__ == "__main__":
    main()
