"""Prophet forecaster with halving cycle regressors.

This is the main forecaster that uses the 5 configurable regressors:
- CYCLE: sin/cos cycle position, pre/post halving weights
- DOUBLE_TOP: double-top pattern detection
- CYCLE_PHASE: continuous cycle phase encoding
- DECAY: drawdown decay curve adjustment
- ENSEMBLE_ADJUST: bull run boost + drawdown reduction
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import BaseForecaster, ForecastResult
from src.models.cycle_features import add_cycle_features, create_cycle_regressors_for_prophet
from src.models.ensemble import get_regressor_config, validate_regressor_config
from src.metrics import (
    HALVING_DATES,
    create_decay_regressor,
    create_double_top_regressor,
    create_cycle_phase_regressor,
    fit_decay_curve,
)

if TYPE_CHECKING:
    from src.metrics import HalvingAverages


class ProphetCycleForecaster(BaseForecaster):
    """Prophet forecaster with halving cycle awareness.

    Uses the 5 configurable regressors (toggle via .env):
    - REGRESSOR_CYCLE: sin/cos cycle position
    - REGRESSOR_DOUBLE_TOP: double-top pattern detection
    - REGRESSOR_CYCLE_PHASE: continuous phase encoding
    - REGRESSOR_DECAY: drawdown curve adjustment
    - REGRESSOR_ENSEMBLE_ADJUST: timing-based post-processing
    """

    def __init__(
        self,
        halving_averages: "HalvingAverages | None" = None,
        cycle_metrics: pd.DataFrame | None = None,
        prophet_weight: float = 0.7,
        cycle_weight: float = 0.3,
        interval_width: float = 0.95,
        changepoint_prior_scale: float = 0.1,
    ):
        """Initialize the forecaster.

        Args:
            halving_averages: HalvingAverages with historical timing data.
            cycle_metrics: DataFrame from compute_cycle_metrics() for per-halving data.
            prophet_weight: Weight for base Prophet forecast in ensemble.
            cycle_weight: Weight for cycle adjustment in ensemble.
            interval_width: Width of confidence intervals.
            changepoint_prior_scale: Flexibility of trend changes.
        """
        self.halving_averages = halving_averages
        self.cycle_metrics = cycle_metrics
        self.prophet_weight = prophet_weight
        self.cycle_weight = cycle_weight
        self.interval_width = interval_width
        self.changepoint_prior_scale = changepoint_prior_scale

        self._model: Prophet | None = None
        self._df: pd.DataFrame | None = None
        self._decay_params: tuple[float, float, float] | None = None
        self._predicted_drawdown: float = 0.1
        self._regressor_config: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "Prophet (Cycle-Aware)"

    def fit(self, df: pd.DataFrame) -> "ProphetCycleForecaster":
        """Fit Prophet with cycle regressors."""
        self._df = df.copy()

        # Get regressor config from environment
        self._regressor_config = get_regressor_config()
        cfg = self._regressor_config

        # Fit decay curve if we have cycle metrics
        if self.cycle_metrics is not None and len(self.cycle_metrics) >= 2:
            cycle_nums = list(range(1, len(self.cycle_metrics) + 1))
            drawdowns = (self.cycle_metrics["drawdown_pct"] / 100).tolist()
            self._decay_params, _ = fit_decay_curve(cycle_nums, drawdowns)

            from src.metrics import exp_decay
            next_cycle = len(self.cycle_metrics) + 1
            self._predicted_drawdown = float(exp_decay(next_cycle, *self._decay_params))

        # Check which regressors are enabled
        enable_cycle = cfg["cycle"]
        enable_double_top = (
            cfg["double_top"]
            and self.halving_averages is not None
            and self.halving_averages.double_top_frequency > 0
        )
        enable_cycle_phase = cfg["cycle_phase"] and self.halving_averages is not None
        enable_decay = cfg["decay"] and self._decay_params is not None

        # Validate configuration
        validate_regressor_config(cfg, print_warnings=True)

        # Add regressors to training data
        if enable_cycle:
            self._df = create_cycle_regressors_for_prophet(
                self._df, halving_averages=self.halving_averages
            )

        if enable_double_top:
            self._df = create_double_top_regressor(self._df, self.halving_averages)

        if enable_cycle_phase:
            self._df = create_cycle_phase_regressor(
                self._df, self.halving_averages, cycle_metrics=self.cycle_metrics
            )

        if enable_decay:
            self._df = create_decay_regressor(
                self._df, self._decay_params, HALVING_DATES,
                halving_averages=self.halving_averages
            )

        # Build Prophet model
        self._model = Prophet(
            interval_width=self.interval_width,
            growth="linear",
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=self.changepoint_prior_scale,
            changepoint_range=0.8,
            seasonality_prior_scale=10,
            n_changepoints=300,
        )

        # Add 4-year halving cycle seasonality
        self._model.add_seasonality(
            name="halving_cycle",
            period=365.25 * 4,
            fourier_order=3,
        )

        # Add regressors to model
        if enable_cycle:
            self._model.add_regressor("reg_cycle_sin", mode="multiplicative")
            self._model.add_regressor("reg_cycle_cos", mode="multiplicative")
            self._model.add_regressor("reg_pre_halving", mode="multiplicative")
            self._model.add_regressor("reg_post_halving", mode="multiplicative")

        if enable_double_top:
            self._model.add_regressor("double_top_regressor", mode="multiplicative")

        if enable_cycle_phase:
            self._model.add_regressor("cycle_phase_regressor", mode="multiplicative")

        if enable_decay:
            self._model.add_regressor("decay_regressor", mode="multiplicative")

        self._model.fit(self._df)
        return self

    def predict(self, periods: int) -> ForecastResult:
        """Generate forecast with cycle adjustments."""
        if self._model is None:
            raise ValueError("Must call fit() before predict()")

        cfg = self._regressor_config

        # Create future dataframe
        future = self._model.make_future_dataframe(periods=periods, freq="d")

        # Add regressors to future
        enable_cycle = cfg["cycle"]
        enable_double_top = (
            cfg["double_top"]
            and self.halving_averages is not None
            and self.halving_averages.double_top_frequency > 0
        )
        enable_cycle_phase = cfg["cycle_phase"] and self.halving_averages is not None
        enable_decay = cfg["decay"] and self._decay_params is not None

        if enable_cycle:
            future = create_cycle_regressors_for_prophet(
                future, halving_averages=self.halving_averages
            )

        if enable_double_top:
            future = create_double_top_regressor(future, self.halving_averages)

        if enable_cycle_phase:
            future = create_cycle_phase_regressor(
                future, self.halving_averages, cycle_metrics=self.cycle_metrics
            )

        if enable_decay:
            future = create_decay_regressor(
                future, self._decay_params, HALVING_DATES,
                halving_averages=self.halving_averages
            )

        # Generate base forecast
        forecast = self._model.predict(future)

        # Add cycle features
        forecast = add_cycle_features(forecast, date_col="ds")

        # Apply ensemble adjustments if enabled
        adjustment = np.ones(len(forecast))

        if cfg["ensemble_adjust"]:
            adjustment = self._compute_ensemble_adjustment(forecast)

        # Apply weighted ensemble
        forecast["cycle_adjustment"] = adjustment
        forecast["yhat_ensemble"] = (
            forecast["yhat"] * self.prophet_weight
            + forecast["yhat"] * adjustment * self.cycle_weight
        )

        # Adjust confidence intervals
        ratio = forecast["yhat_ensemble"] / forecast["yhat"].replace(0, 1)
        forecast["yhat_ensemble_lower"] = forecast["yhat_lower"] * ratio
        forecast["yhat_ensemble_upper"] = forecast["yhat_upper"] * ratio

        return ForecastResult(
            forecast=forecast,
            model=self._model,
            config={
                "halving_averages": self.halving_averages is not None,
                "cycle_metrics": self.cycle_metrics is not None,
                "prophet_weight": self.prophet_weight,
                "cycle_weight": self.cycle_weight,
                "regressors": self._regressor_config,
                "predicted_drawdown": self._predicted_drawdown,
            },
        )

    def _compute_ensemble_adjustment(self, forecast: pd.DataFrame) -> np.ndarray:
        """Compute data-driven ensemble adjustment multiplier."""
        adjustment = np.ones(len(forecast))

        # Pre-halving boost
        if self.halving_averages is not None and self.halving_averages.run_up_pct > 0:
            pre_boost = min(0.20, self.halving_averages.run_up_pct / 2000)
        else:
            pre_boost = 0.15

        # Bull run timing and boost
        if self.halving_averages is not None and self.halving_averages.avg_days_to_top > 0:
            bull_end_day = int(self.halving_averages.avg_days_to_top)
            bull_start_day = 90
            bull_boost = min(0.18, self.halving_averages.run_up_pct / 2500)
        else:
            bull_start_day = 120
            bull_end_day = 365
            bull_boost = 0.12

        # Distribution reduction
        if self._predicted_drawdown > 0:
            dist_reduction = min(0.08, self._predicted_drawdown / 10)
        else:
            dist_reduction = 0.03

        # Apply pre-halving boost
        pre_mask = forecast["pre_halving_weight"] > 0
        adjustment[pre_mask] += forecast.loc[pre_mask, "pre_halving_weight"] * pre_boost

        # Apply bull run boost
        days_since = forecast["days_since_halving"].fillna(0)
        bull_mask = (days_since > bull_start_day) & (days_since <= bull_end_day)
        if bull_mask.any():
            peak_day = (bull_start_day + bull_end_day) / 2
            spread = (bull_end_day - bull_start_day) / 3
            bull_weight = np.exp(-((days_since[bull_mask] - peak_day) ** 2) / (2 * spread ** 2))
            adjustment[bull_mask] += bull_weight * bull_boost

        # Apply distribution reduction
        dist_start = bull_end_day
        dist_mask = days_since > dist_start
        if dist_mask.any():
            days_into_dist = (days_since[dist_mask] - dist_start).values
            reduction_factor = np.minimum(days_into_dist / 200, 1.0)
            adjustment[dist_mask] -= reduction_factor * dist_reduction

        return adjustment
