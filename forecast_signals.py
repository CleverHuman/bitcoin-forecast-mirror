"""BTC forecast with cycle-aware signals for buy/sell timing.

Uses halving cycle position + technical indicators to generate signals.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.db import fetch_btc_data
from src.metrics import (
    compute_cycle_metrics,
    compute_halving_averages,
    print_halving_summary,
)
from src.models import (
    backtest_signals,
    generate_signals,
    get_current_signal,
    train_simple_ensemble,
)
from src.reporting import print_signal_summary
from src.utils import TeeOutput
from src.viz import plot_signals

load_dotenv()

# Forecast horizon in days (set FORECAST_DAYS in .env; default 365)
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "365"))

# Reports directory
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee = TeeOutput()
    sys.stdout = tee

    try:
        start_date = "2015-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        print(f"Run timestamp: {timestamp}")
        print(f"Fetching BTC data from {start_date} to {end_date}...")
        df = fetch_btc_data(start_date, end_date)
        print(f"Loaded {len(df)} rows.")

        print("\nComputing halving cycle metrics...")
        cycle_metrics = compute_cycle_metrics(df)
        averages = compute_halving_averages(cycle_metrics=cycle_metrics)
        print_halving_summary(cycle_metrics, averages)

        print("\nTraining cycle-aware ensemble model with decay adjustment...")
        forecast = train_simple_ensemble(
            df,
            periods=FORECAST_DAYS,
            halving_averages=averages,
            cycle_metrics=cycle_metrics,
        )

        print("Generating buy/sell signals...")
        df_signals = generate_signals(df, cycle_weight=0.4, technical_weight=0.6)
        current = get_current_signal(df_signals, cycle_metrics=cycle_metrics, averages=averages)

        print("Running backtest...")
        backtest = backtest_signals(df_signals, initial_capital=10000)
        print_signal_summary(current, backtest)

        print(f"\nSaving results to {REPORTS_DIR}/...")
        signals_path = REPORTS_DIR / f"signals_{timestamp}.csv"
        forecast_path = REPORTS_DIR / f"forecast_{timestamp}.csv"
        report_path = REPORTS_DIR / f"report_{timestamp}.txt"

        df_signals.to_csv(signals_path, index=False)
        forecast.to_csv(forecast_path, index=False)
        print(f"  Signals:  {signals_path.name}")
        print(f"  Forecast: {forecast_path.name}")
        print(f"  Report:   {report_path.name}")

        print("\nPlotting...")
        plot_signals(df_signals, forecast, cycle_metrics=cycle_metrics)
        print("\nDone!")

    finally:
        sys.stdout = tee.stdout
        with open(REPORTS_DIR / f"report_{timestamp}.txt", "w") as f:
            f.write(tee.getvalue())


if __name__ == "__main__":
    main()
