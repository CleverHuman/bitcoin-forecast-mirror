#!/usr/bin/env python3
"""Compare trading strategies using the new backtesting framework.

This script demonstrates:
1. Loading BTC data
2. Running the ProphetCycleForecaster
3. Comparing multiple strategies:
   - CycleSignalStrategy (primary - uses halving cycle + technicals)
   - ForecastBasedStrategy (trades on forecast direction)
   - BuyAndHoldStrategy (baseline benchmark)
"""

from datetime import datetime
from dotenv import load_dotenv

# Load environment variables before imports
load_dotenv()

from src.db import fetch_btc_data
from src.metrics import compute_cycle_metrics, compute_halving_averages
from src.forecasting import ProphetCycleForecaster
from src.backtesting import (
    BacktestRunner,
    BacktestConfig,
    compare_strategies,
    print_comparison,
    print_metrics_report,
)
from src.backtesting.strategies import (
    CycleSignalStrategy,
    ForecastBasedStrategy,
    ForecastMomentumStrategy,
    BuyAndHoldStrategy,
)


def main():
    print("=" * 70)
    print("BTC STRATEGY COMPARISON")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading BTC data...")
    start_date = "2012-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    df = fetch_btc_data(start_date=start_date, end_date=end_date)
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
        initial_capital=10000,
        position_size=0.25,
        max_position=1.0,
        fee_pct=0.1,
        slippage_pct=0.05,
    )

    strategies = [
        CycleSignalStrategy(
            cycle_weight=0.6,
            technical_weight=0.4,
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
        ),
        ForecastMomentumStrategy(
            forecast_col="yhat_ensemble",
            momentum_window=7,
            divergence_threshold_pct=5.0,
            min_trade_interval_days=7,  # ~4 trades per month max
        ),
        ForecastBasedStrategy(
            forecast_col="yhat_ensemble",
            threshold_pct=5.0,
            lookforward_days=30,
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

    # Detailed report for the cycle strategy
    print("\n" + "=" * 70)
    print("DETAILED: Halving Cycle Strategy")
    print("=" * 70)
    cycle_metrics_result = comparison.results["Halving Cycle Strategy"]
    print_metrics_report(cycle_metrics_result, "Halving Cycle Strategy")


if __name__ == "__main__":
    main()
