"""Plotting for BTC price, signals, and cycle phases."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics import HALVING_DATES, backtest_predictions
from src.models import add_cycle_features


def _draw_halving_lines(ax, ds_min: pd.Timestamp, ds_max: pd.Timestamp) -> None:
    """Draw vertical lines and labels for halving dates on an axis."""
    for i, h in enumerate(HALVING_DATES):
        if ds_min <= h <= ds_max:
            ax.axvline(x=h, color="red", linestyle="--", alpha=0.7, linewidth=2)
            ax.text(h, ax.get_ylim()[1] * 0.9, f"Halving {i+1}", rotation=90, va="top", fontsize=8)


def _draw_tops_bottoms(ax, cycle_metrics: pd.DataFrame | None) -> None:
    """Draw actual and predicted cycle tops/bottoms on an axis."""
    if cycle_metrics is None or cycle_metrics.empty:
        return

    # Actual tops (filled red triangle pointing down)
    actual_tops = cycle_metrics[["post_high_date", "post_high_price"]].dropna()
    ax.scatter(
        actual_tops["post_high_date"],
        actual_tops["post_high_price"],
        marker="v",
        s=120,
        c="red",
        edgecolors="darkred",
        linewidths=1.5,
        zorder=10,
        label="Actual Top",
    )

    # Actual bottoms (filled green triangle pointing up)
    actual_bottoms = cycle_metrics[["post_low_date", "post_low_price"]].dropna()
    ax.scatter(
        actual_bottoms["post_low_date"],
        actual_bottoms["post_low_price"],
        marker="^",
        s=120,
        c="limegreen",
        edgecolors="darkgreen",
        linewidths=1.5,
        zorder=10,
        label="Actual Bottom",
    )

    # Predicted tops/bottoms from backtesting
    backtest = backtest_predictions(cycle_metrics)
    if not backtest.empty:
        ax.scatter(
            backtest["predicted_top"],
            backtest["actual_top_price"],
            marker="D",
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=9,
            label="Predicted Top",
        )
        ax.scatter(
            backtest["predicted_bottom"],
            backtest["actual_bottom_price"],
            marker="D",
            s=80,
            facecolors="none",
            edgecolors="limegreen",
            linewidths=2,
            zorder=9,
            label="Predicted Bottom",
        )


def _draw_price_forecast(
    ax,
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    last_historical: pd.Timestamp,
    log_scale: bool = False,
) -> None:
    """Draw actual price, forecast, and optional log scale on an axis."""
    ax.plot(df["ds"], df["y"], "k-", label="Actual Price", alpha=0.8, linewidth=1)

    if "yhat_ensemble" in forecast.columns:
        ax.plot(forecast["ds"], forecast["yhat_ensemble"], "b--", label="Ensemble Forecast", alpha=0.7)
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_ensemble_lower"],
            forecast["yhat_ensemble_upper"],
            alpha=0.2,
        )
    else:
        ax.plot(forecast["ds"], forecast["yhat"], "b--", label="Prophet Forecast", alpha=0.7)

    _draw_halving_lines(ax, df["ds"].min(), forecast["ds"].max())
    ax.axvline(x=last_historical, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")


def plot_signals(
    df: pd.DataFrame,
    forecast: pd.DataFrame,
    cycle_metrics: pd.DataFrame | None = None,
) -> None:
    """Plot price with buy/sell signals and cycle phases.

    Args:
        df: Historical price data with signals.
        forecast: Forecasted prices.
        cycle_metrics: Optional cycle metrics for plotting tops/bottoms.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    df = df.sort_values("ds").reset_index(drop=True)
    forecast = forecast.sort_values("ds").reset_index(drop=True)
    last_historical = df["ds"].max()

    # Top: Price with forecast (log scale)
    ax1 = axes[0]
    _draw_price_forecast(ax1, df, forecast, last_historical, log_scale=True)
    _draw_tops_bottoms(ax1, cycle_metrics)
    ax1.set_title("BTC Price with Cycle-Aware Forecast")

    # Middle: Signal score
    ax2 = axes[1]
    colors = ["green" if x > 0 else "red" for x in df["signal_score"]]
    ax2.bar(df["ds"], df["signal_score"], color=colors, alpha=0.6, width=2)
    ax2.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=0.2, color="lightgreen", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5, linewidth=1)
    ax2.axhline(y=-0.2, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=-0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(x=last_historical, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.set_ylabel("Signal Score")
    ax2.set_ylim(-1, 1)
    ax2.set_yticks([-1, -0.5, -0.2, 0, 0.2, 0.5, 1])
    ax2.set_yticklabels(["Strong Sell", "Sell", "", "Hold", "", "Buy", "Strong Buy"], fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Bottom: Cycle phase
    ax3 = axes[2]
    phase_colors = {
        "accumulation": "green",
        "pre_halving_runup": "lightgreen",
        "post_halving_consolidation": "yellow",
        "bull_run": "orange",
        "distribution": "red",
        "drawdown": "purple",
    }
    for phase, color in phase_colors.items():
        mask = df["cycle_phase"] == phase
        if mask.any():
            ax3.fill_between(df["ds"], 0, 1, where=mask, color=color, alpha=0.6, label=phase.replace("_", " ").title())

    forecast_with_phase = add_cycle_features(forecast[["ds"]].copy(), "ds")
    pred = forecast_with_phase[forecast_with_phase["ds"] > last_historical]
    if not pred.empty:
        ax3.axvline(x=last_historical, color="gray", linestyle=":", linewidth=2, alpha=0.9)
        first_pred_label = True
        for phase, color in phase_colors.items():
            mask = pred["cycle_phase"] == phase
            if mask.any():
                lbl = "Predicted phases" if first_pred_label else None
                first_pred_label = False
                ax3.fill_between(
                    pred["ds"], 0, 1, where=mask, color=color, alpha=0.4, hatch="..", label=lbl
                )

    ax3.set_ylabel("Cycle Phase")
    ax3.set_yticks([])
    ax3.set_ylim(0, 1)
    ax3.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.15))
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")
    ax3.set_xlabel("Date")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    plt.show()
    plt.close(fig)

    # Second figure: price only, linear scale
    fig2, ax = plt.subplots(1, 1, figsize=(16, 6))
    _draw_price_forecast(ax, df, forecast, last_historical, log_scale=False)
    _draw_tops_bottoms(ax, cycle_metrics)
    ax.set_xlabel("Date")
    ax.set_title("BTC Price with Forecast (linear scale)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig2.tight_layout()

    plt.show()
    plt.close(fig2)
