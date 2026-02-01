"""BTC price forecasting using Prophet with data from Databricks."""

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from matplotlib.ticker import MaxNLocator
from prophet import Prophet

from src.db import DatabricksConnector

from src.metrics import (
    compute_cycle_metrics,
    compute_halving_averages,
    get_prophet_params_from_halving,
    print_halving_summary,
    sanity_check_forecast,
)

# Load environment variables from .env file
load_dotenv()


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


def create_halving_holidays() -> pd.DataFrame:
    """Create a DataFrame of BTC halving dates for Prophet holidays."""
    specific_dates = {
        "ds": ["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-19", "2028-04-11"],
        "holiday": ["btc_halving", "btc_halving", "btc_halving", "btc_halving", "btc_halving"],
    }
    holiday = pd.DataFrame.from_dict(specific_dates)
    holiday["ds"] = pd.to_datetime(holiday["ds"])
    return holiday


def add_moving_averages(df: pd.DataFrame, col: str = "y") -> pd.DataFrame:
    """Add moving averages to a DataFrame.

    Args:
        df: DataFrame with the column to compute moving averages on.
        col: Column name to compute moving averages from.

    Returns:
        DataFrame with added moving average columns.
    """
    df_reversed = df.iloc[::-1].reset_index(drop=True)
    df_reversed[f"{col}_moving_avg_7"] = df_reversed[col].rolling(window=7).mean()
    df_reversed[f"{col}_moving_avg_14"] = df_reversed[col].rolling(window=14).mean()
    df_reversed[f"{col}_moving_avg_28"] = df_reversed[col].rolling(window=28).mean()
    df_reversed[f"{col}_moving_avg_90"] = df_reversed[col].rolling(window=90).mean()

    df[f"{col}_moving_avg_7"] = df_reversed[f"{col}_moving_avg_7"].iloc[::-1].reset_index(drop=True)
    df[f"{col}_moving_avg_14"] = df_reversed[f"{col}_moving_avg_14"].iloc[::-1].reset_index(drop=True)
    df[f"{col}_moving_avg_28"] = df_reversed[f"{col}_moving_avg_28"].iloc[::-1].reset_index(drop=True)
    df[f"{col}_moving_avg_90"] = df_reversed[f"{col}_moving_avg_90"].iloc[::-1].reset_index(drop=True)

    return df


def train_and_forecast(df: pd.DataFrame, periods: int = 240) -> tuple:
    """Train Prophet model and generate forecast.

    Uses same Prophet config as forecast.py so predictions match (not over-smoothed).
    Cycle metrics are used only for sanity-check, not for fitting.

    Args:
        df: DataFrame with 'ds' and 'y' columns.
        periods: Number of days to forecast into the future.

    Returns:
        Tuple of (trained model, forecast DataFrame).
    """
    holiday = create_halving_holidays()

    model = Prophet(
        interval_width=0.95,
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
        holidays=holiday,
        changepoint_prior_scale=0.1,
        changepoint_range=0.8,
        seasonality_prior_scale=10,
        n_changepoints=300,
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq="d", include_history=True)
    forecast = model.predict(future)

    return model, forecast


def plot_forecast(model: Prophet, df: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """Plot the forecast with moving averages and halving events.

    Args:
        model: Trained Prophet model.
        df: Historical data DataFrame with moving averages.
        forecast: Forecast DataFrame with moving averages.
    """
    # Halving event markers
    dates = [datetime.strptime("2024-04-19", "%Y-%m-%d")]
    events = ["4th Halving"]

    # Plot the forecast (sized to fit typical screens)
    fig1 = model.plot(forecast)
    fig1.set_size_inches(14, 8)

    # Plot the forecast components
    fig2 = model.plot_components(forecast)
    fig2.set_size_inches(12, 8)

    ax = fig1.gca()

    # Add vertical lines for each halving date
    for date, event in zip(dates, events):
        ax.axvline(x=date, color="red", linestyle="--", lw=2)
        ax.text(date, ax.get_ylim()[1], event, rotation=45, verticalalignment="bottom")

    # Set major ticks format
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-01"))

    # Rotate labels to avoid overlap
    for label in ax.get_xticklabels():
        label.set_rotation(90)

    # Add more y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=50))

    plt.xticks(rotation=90)
    ax.set_xlim([datetime(2023, 1, 1), forecast["ds"].max()])

    # Plot historical moving averages
    ax.plot(df["ds"], df["y_moving_avg_7"], color="green", label="7-Day Moving Average (Historical)")
    ax.plot(df["ds"], df["y_moving_avg_14"], color="red", label="14-Day Moving Average (Historical)")
    ax.plot(df["ds"], df["y_moving_avg_28"], color="purple", label="28-Day Moving Average (Historical)")
    ax.plot(df["ds"], df["y_moving_avg_90"], color="orange", label="90-Day Moving Average (Historical)")

    # Plot forecasted moving averages
    ax.plot(forecast["ds"], forecast["yhat_moving_avg_7"], color="green", linestyle="--", label="7-Day Moving Average (Forecast)")
    ax.plot(forecast["ds"], forecast["yhat_moving_avg_14"], color="red", linestyle="--", label="14-Day Moving Average (Forecast)")
    ax.plot(forecast["ds"], forecast["yhat_moving_avg_28"], color="purple", linestyle="--", label="28-Day Moving Average (Forecast)")
    ax.plot(forecast["ds"], forecast["yhat_moving_avg_90"], color="orange", linestyle="--", label="90-Day Moving Average (Forecast)")

    ax.legend()
    plt.show()
    plt.close("all")


def main():
    # Configuration: use start_date that covers past halvings for data-driven metrics
    start_date = "2015-01-01"  # Covers 2016, 2020, 2024 halving cycles
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Fetching BTC data from {start_date} to {end_date}...")
    print("(Querying Databricks; this may take a few seconds.)")
    df = fetch_btc_data(start_date, end_date)
    print(f"Loaded {len(df)} rows of data.")

    # Data-driven halving metrics: run-up, drawdown, durations; average across cycles
    print("Computing halving-cycle metrics (run-up, drawdown, duration)...")
    cycle_metrics = compute_cycle_metrics(df)
    averages = compute_halving_averages(cycle_metrics=cycle_metrics)
    print_halving_summary(cycle_metrics, averages)

    # Use same Prophet config as forecast.py (fixed changepoint_range=0.8) so the
    # forecast is not over-smoothed; cycle metrics are still used for sanity-check only.
    prophet_params = get_prophet_params_from_halving(averages)
    if prophet_params:
        print("Suggested Prophet params from halving (for reference):", prophet_params)

    print("Training Prophet model...")
    model, forecast = train_and_forecast(df)

    print("Calculating moving averages...")
    df = add_moving_averages(df, col="y")
    forecast["yhat_moving_avg_7"] = forecast["yhat"].rolling(window=7).mean()
    forecast["yhat_moving_avg_14"] = forecast["yhat"].rolling(window=14).mean()
    forecast["yhat_moving_avg_28"] = forecast["yhat"].rolling(window=28).mean()
    forecast["yhat_moving_avg_90"] = forecast["yhat"].rolling(window=90).mean()

    print("Saving results...")
    df.to_csv("historical_data.csv", index=False)
    forecast.to_csv("forecasted_data.csv", index=False)

    # Sanity-check forecast against historical halving run-up/drawdown
    next_halving = pd.Timestamp("2028-04-11")
    check = sanity_check_forecast(forecast, averages, next_halving, window_days=180)
    print(f"Sanity check (forecast vs historical halving norms): {check['message']}")

    print("Plotting forecast...")
    plot_forecast(model, df, forecast)

    print("Done!")


if __name__ == "__main__":
    main()
