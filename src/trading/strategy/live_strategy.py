"""Live trading strategy wrapper using ProphetCycleForecaster."""

import logging
from datetime import datetime, timedelta, date
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.strategies.base import Signal, StrategySignal, classify_signal
from src.forecasting.base import ForecastResult
from src.forecasting.prophet_cycle import ProphetCycleForecaster
from src.metrics.halving import HalvingAverages
from src.models.cycle_features import CyclePhase, get_cycle_phase
from src.trading.config import TradingConfig

logger = logging.getLogger(__name__)


class LiveStrategy:
    """Live trading strategy wrapper using forecast-centric logic.

    This strategy uses ProphetCycleForecaster predictions as the primary
    decision-making engine (60% weight by default), with tactical indicators
    for entry timing (40% weight).

    The strategy:
    1. Gets predicted price N days ahead from the forecast
    2. Calculates expected return percentage
    3. Considers current cycle phase
    4. Adds tactical confirmation (momentum, RSI, price vs forecast deviation)
    5. Returns a weighted signal score

    Attributes:
        forecaster: ProphetCycleForecaster instance
        config: Trading configuration
        halving_averages: Historical halving cycle averages
        forecast_result: Most recent forecast result
        last_forecast_time: When forecast was last refreshed
    """

    def __init__(
        self,
        forecaster: ProphetCycleForecaster,
        config: TradingConfig,
        halving_averages: HalvingAverages | None = None,
    ):
        """Initialize live strategy.

        Args:
            forecaster: ProphetCycleForecaster instance
            config: Trading configuration
            halving_averages: Historical halving averages for cycle phase detection
        """
        self.forecaster = forecaster
        self.config = config
        self.halving_averages = halving_averages
        self.forecast_result: ForecastResult | None = None
        self.last_forecast_time: datetime | None = None
        self._historical_df: pd.DataFrame | None = None

    def refresh_forecast(
        self,
        historical_df: pd.DataFrame,
        periods: int = 365,
    ) -> ForecastResult:
        """Re-run Prophet forecast.

        Should be called daily or when forecast becomes stale.

        Args:
            historical_df: Historical price data with 'ds' and 'y' columns
            periods: Number of days to forecast

        Returns:
            New forecast result
        """
        logger.info("Refreshing forecast...")

        # Fit and predict
        self.forecaster.fit(historical_df)
        self.forecast_result = self.forecaster.predict(periods=periods)
        self.last_forecast_time = datetime.now()
        self._historical_df = historical_df.copy()

        logger.info(
            f"Forecast refreshed: {len(self.forecast_result.forecast)} days, "
            f"next prediction at {self.forecast_result.forecast['ds'].iloc[-1]}"
        )

        return self.forecast_result

    def needs_refresh(self) -> bool:
        """Check if forecast needs to be refreshed.

        Returns:
            True if forecast is stale or missing
        """
        if self.forecast_result is None or self.last_forecast_time is None:
            return True

        hours_since_refresh = (
            datetime.now() - self.last_forecast_time
        ).total_seconds() / 3600

        return hours_since_refresh >= self.config.forecast_refresh_hours

    def get_signal(
        self,
        current_price: float,
        current_date: date | None = None,
        include_details: bool = True,
    ) -> StrategySignal:
        """Generate trading signal.

        Combines forecast-based scoring (60%) with tactical scoring (40%)
        to produce a final signal.

        Args:
            current_price: Current BTC price
            current_date: Current date (defaults to today)
            include_details: Whether to include detailed metadata

        Returns:
            StrategySignal with combined score and reasoning
        """
        current_date = current_date or datetime.now().date()

        if self.forecast_result is None:
            return StrategySignal(
                date=pd.Timestamp(current_date),
                signal=Signal.HOLD,
                score=0.0,
                reason="No forecast available - refresh needed",
                metadata={"error": "no_forecast"},
            )

        # Get component scores
        forecast_score, forecast_details = self._forecast_score(
            current_price, current_date
        )
        tactical_score, tactical_details = self._tactical_score(
            current_price, current_date
        )

        # Weighted combination
        combined_score = (
            self.config.forecast_weight * forecast_score
            + self.config.tactical_weight * tactical_score
        )

        # Clip to valid range
        combined_score = np.clip(combined_score, -1.0, 1.0)

        # Classify signal
        signal = classify_signal(combined_score)

        # Build reason
        direction = "bullish" if combined_score > 0 else "bearish" if combined_score < 0 else "neutral"
        reason = (
            f"{direction.capitalize()} signal: "
            f"forecast={forecast_score:.2f} ({self.config.forecast_weight:.0%}), "
            f"tactical={tactical_score:.2f} ({self.config.tactical_weight:.0%})"
        )

        metadata = None
        if include_details:
            metadata = {
                "forecast_score": forecast_score,
                "tactical_score": tactical_score,
                "forecast_weight": self.config.forecast_weight,
                "tactical_weight": self.config.tactical_weight,
                "forecast_details": forecast_details,
                "tactical_details": tactical_details,
                "current_price": current_price,
                "lookforward_days": self.config.lookforward_days,
            }

        return StrategySignal(
            date=pd.Timestamp(current_date),
            signal=signal,
            score=combined_score,
            reason=reason,
            metadata=metadata,
        )

    def _forecast_score(
        self,
        current_price: float,
        current_date: date,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate score based on forecast direction and magnitude.

        Scoring logic:
        - Strong signals (> +/- 0.7): 10%+ predicted move
        - Moderate signals (0.3-0.7): 5-10% predicted move
        - Weak signals (< 0.3): < 5% predicted move

        Also considers:
        - Confidence interval width (uncertainty discount)
        - Cycle phase alignment

        Args:
            current_price: Current BTC price
            current_date: Current date

        Returns:
            Tuple of (score, details_dict)
        """
        target_date = current_date + timedelta(days=self.config.lookforward_days)
        predicted_price = self._get_predicted_price(target_date)

        if predicted_price is None:
            return 0.0, {"error": "no_prediction_for_date"}

        # Calculate expected return
        expected_return_pct = (predicted_price - current_price) / current_price * 100

        # Base score: scale 10%+ move to +/- 1.0
        # Using 15% as the maximum expected move for full score
        base_score = np.clip(expected_return_pct / 15.0, -1.0, 1.0)

        # Get confidence interval at target date
        lower, upper = self._get_confidence_bounds(target_date)
        if lower is not None and upper is not None:
            interval_width = (upper - lower) / current_price * 100
            # Discount score if confidence interval is very wide (> 30%)
            uncertainty_factor = max(0.5, 1.0 - (interval_width - 15) / 30)
            score = base_score * uncertainty_factor
        else:
            score = base_score
            interval_width = None

        # Cycle phase bonus/penalty
        cycle_phase = self._get_current_cycle_phase(current_date)
        phase_adjustment = self._get_cycle_phase_adjustment(cycle_phase, base_score)
        score = np.clip(score + phase_adjustment, -1.0, 1.0)

        details = {
            "predicted_price": predicted_price,
            "expected_return_pct": expected_return_pct,
            "base_score": base_score,
            "confidence_interval_width": interval_width,
            "cycle_phase": cycle_phase.value if cycle_phase else None,
            "phase_adjustment": phase_adjustment,
            "final_score": score,
        }

        return score, details

    def _tactical_score(
        self,
        current_price: float,
        current_date: date,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate tactical score for entry timing.

        Uses:
        - Price vs forecast deviation (mean reversion opportunity)
        - Short-term momentum (trend confirmation)
        - RSI for overbought/oversold conditions

        Args:
            current_price: Current BTC price
            current_date: Current date

        Returns:
            Tuple of (score, details_dict)
        """
        scores = []
        details = {}

        # 1. Price vs forecast deviation (40% of tactical)
        forecast_price = self._get_predicted_price(current_date)
        if forecast_price is not None:
            deviation_pct = (current_price - forecast_price) / forecast_price * 100
            # If price is below forecast, bullish opportunity
            # If price is above forecast, bearish opportunity
            deviation_score = np.clip(-deviation_pct / 10.0, -1.0, 1.0)
            scores.append(("deviation", deviation_score, 0.4))
            details["deviation_pct"] = deviation_pct
            details["deviation_score"] = deviation_score

        # 2. Short-term momentum (30% of tactical)
        if self._historical_df is not None and len(self._historical_df) >= 10:
            momentum_score = self._calculate_momentum_score()
            scores.append(("momentum", momentum_score, 0.3))
            details["momentum_score"] = momentum_score

        # 3. RSI-based score (30% of tactical)
        if self._historical_df is not None and len(self._historical_df) >= 14:
            rsi = self._calculate_rsi()
            if rsi is not None:
                # RSI < 30: oversold = bullish
                # RSI > 70: overbought = bearish
                if rsi < 30:
                    rsi_score = 0.5 + (30 - rsi) / 60  # 0.5 to 1.0
                elif rsi > 70:
                    rsi_score = -0.5 - (rsi - 70) / 60  # -0.5 to -1.0
                else:
                    rsi_score = 0.0
                scores.append(("rsi", rsi_score, 0.3))
                details["rsi"] = rsi
                details["rsi_score"] = rsi_score

        # Weighted combination
        if not scores:
            return 0.0, details

        total_weight = sum(w for _, _, w in scores)
        tactical_score = sum(s * w for _, s, w in scores) / total_weight

        details["component_scores"] = {name: score for name, score, _ in scores}
        details["final_score"] = tactical_score

        return tactical_score, details

    def _get_predicted_price(self, target_date: date) -> float | None:
        """Get predicted price from forecast for a specific date.

        Args:
            target_date: Date to get prediction for

        Returns:
            Predicted price or None if not available
        """
        if self.forecast_result is None:
            return None

        forecast_df = self.forecast_result.forecast
        target_ts = pd.Timestamp(target_date)

        # Find closest date in forecast
        if "ds" not in forecast_df.columns:
            return None

        forecast_df = forecast_df.copy()
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        matching = forecast_df[forecast_df["ds"].dt.date == target_date]
        if len(matching) > 0:
            # Use yhat_ensemble if available, otherwise yhat
            if "yhat_ensemble" in matching.columns:
                return float(matching.iloc[0]["yhat_ensemble"])
            return float(matching.iloc[0]["yhat"])

        # If exact date not found, interpolate
        before = forecast_df[forecast_df["ds"] < target_ts].tail(1)
        after = forecast_df[forecast_df["ds"] > target_ts].head(1)

        if len(before) > 0 and len(after) > 0:
            col = "yhat_ensemble" if "yhat_ensemble" in forecast_df.columns else "yhat"
            # Linear interpolation
            days_before = (target_ts - before.iloc[0]["ds"]).days
            days_after = (after.iloc[0]["ds"] - target_ts).days
            total_days = days_before + days_after
            weight_after = days_before / total_days
            return float(
                before.iloc[0][col] * (1 - weight_after)
                + after.iloc[0][col] * weight_after
            )

        return None

    def _get_confidence_bounds(
        self,
        target_date: date,
    ) -> tuple[float | None, float | None]:
        """Get confidence interval bounds for a date.

        Args:
            target_date: Date to get bounds for

        Returns:
            Tuple of (lower, upper) bounds or (None, None)
        """
        if self.forecast_result is None:
            return None, None

        forecast_df = self.forecast_result.forecast

        if "yhat_lower" not in forecast_df.columns:
            return None, None

        matching = forecast_df[
            pd.to_datetime(forecast_df["ds"]).dt.date == target_date
        ]

        if len(matching) > 0:
            return (
                float(matching.iloc[0]["yhat_lower"]),
                float(matching.iloc[0]["yhat_upper"]),
            )

        return None, None

    def _get_current_cycle_phase(self, current_date: date) -> CyclePhase | None:
        """Determine current Bitcoin cycle phase.

        Args:
            current_date: Current date

        Returns:
            Current cycle phase or None
        """
        try:
            return get_cycle_phase(pd.Timestamp(current_date))
        except Exception:
            return None

    def _get_cycle_phase_adjustment(
        self,
        phase: CyclePhase | None,
        base_score: float,
    ) -> float:
        """Get adjustment based on cycle phase.

        Amplifies or dampens signals based on where we are in the cycle.

        Args:
            phase: Current cycle phase
            base_score: Base forecast score

        Returns:
            Adjustment to add to score
        """
        if phase is None:
            return 0.0

        # Phase alignments with signal direction
        if phase == CyclePhase.ACCUMULATION:
            # Bullish signals get boost in accumulation
            return 0.1 if base_score > 0 else -0.05

        elif phase == CyclePhase.PRE_HALVING_RUNUP:
            # Strong bullish bias
            return 0.15 if base_score > 0 else -0.1

        elif phase == CyclePhase.BULL_RUN:
            # Bullish but be cautious of tops
            return 0.05 if base_score > 0 else 0.0

        elif phase == CyclePhase.DISTRIBUTION:
            # Be careful - bearish signals get boost
            return -0.1 if base_score > 0 else 0.1

        elif phase == CyclePhase.DRAWDOWN:
            # Bearish signals confirmed
            return -0.05 if base_score > 0 else 0.15

        return 0.0

    def _calculate_momentum_score(self) -> float:
        """Calculate short-term momentum score.

        Uses 5-day vs 20-day price comparison.

        Returns:
            Momentum score (-1 to +1)
        """
        if self._historical_df is None or len(self._historical_df) < 20:
            return 0.0

        df = self._historical_df.copy()
        df = df.sort_values("ds")

        recent_price = df["y"].iloc[-1]
        price_5d_ago = df["y"].iloc[-5] if len(df) >= 5 else recent_price
        price_20d_ago = df["y"].iloc[-20] if len(df) >= 20 else recent_price

        short_momentum = (recent_price - price_5d_ago) / price_5d_ago * 100
        medium_momentum = (recent_price - price_20d_ago) / price_20d_ago * 100

        # Combine short and medium momentum
        combined = (short_momentum * 0.6 + medium_momentum * 0.4)

        # Scale: 10% momentum = +/- 1.0 score
        return np.clip(combined / 10.0, -1.0, 1.0)

    def _calculate_rsi(self, period: int = 14) -> float | None:
        """Calculate RSI.

        Args:
            period: RSI period

        Returns:
            RSI value (0-100) or None
        """
        if self._historical_df is None or len(self._historical_df) < period + 1:
            return None

        df = self._historical_df.copy()
        df = df.sort_values("ds")
        prices = df["y"].values

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_forecast_summary(self) -> dict[str, Any]:
        """Get summary of current forecast state.

        Returns:
            Dictionary with forecast information
        """
        if self.forecast_result is None:
            return {"status": "no_forecast"}

        forecast_df = self.forecast_result.forecast
        col = "yhat_ensemble" if "yhat_ensemble" in forecast_df.columns else "yhat"

        return {
            "status": "active",
            "last_refresh": self.last_forecast_time.isoformat() if self.last_forecast_time else None,
            "refresh_hours": self.config.forecast_refresh_hours,
            "needs_refresh": self.needs_refresh(),
            "forecast_days": len(forecast_df),
            "first_date": str(forecast_df["ds"].iloc[0]),
            "last_date": str(forecast_df["ds"].iloc[-1]),
            "current_prediction": float(forecast_df[col].iloc[0]),
            "end_prediction": float(forecast_df[col].iloc[-1]),
        }
