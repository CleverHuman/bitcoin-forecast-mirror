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
from datetime import datetime
from pathlib import Path

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
)
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
        default=10000,
        help="Initial capital for backtest (default: 10000)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save summary to reports/backtest_summary.csv",
    )
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    print("=" * 70)
    print("BTC STRATEGY COMPARISON (for paper/live strategy choice)")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading BTC data...")
    df = fetch_btc_data(start_date=args.start, end_date=end_date)
    print(f"  Loaded {len(df)} days of data")
    print(f"  Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")

    # 2. Compute cycle metrics
    print("\n[2/4] Computing cycle metrics...")
    cycle_metrics = compute_cycle_metrics(df)
    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print(f"  Analyzed {averages.n_cycles} complete halving cycles")
    print(f"  Avg run-up: {averages.run_up_pct:.1f}%")
    print(f"  Avg days to top: {averages.avg_days_to_top:.0f}")

    # 3. Generate forecast (optional - needed for ForecastBasedStrategy)
    print("\n[3/4] Generating forecast...")
    forecaster = ProphetCycleForecaster(
        halving_averages=averages,
        cycle_metrics=cycle_metrics,
    )
    result = forecaster.fit_predict(df, periods=365)
    forecast = result.forecast
    print(f"  Forecast generated through {forecast['ds'].max().date()}")

    # 4. Compare strategies
    print("\n[4/4] Running backtests...")

    config = BacktestConfig(
        initial_capital=args.capital,
        position_size=0.25,
        max_position=1.0,
        fee_pct=0.1,
        slippage_pct=0.05,
    )

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

    # Detailed report for best by return
    best_name = comparison.summary.iloc[0]["Strategy"]
    print("\n" + "=" * 70)
    print(f"DETAILED: {best_name}")
    print("=" * 70)
    print_metrics_report(comparison.results[best_name], best_name)

    print("\n" + "=" * 70)
    print("NEXT: Paper trade with live prices")
    print("=" * 70)
    print("  python live_trader.py --capital", int(args.capital))
    print("  (Uses forecast + tactical logic; closest backtest analogue: Combined)")


if __name__ == "__main__":
    main()
