"""Combined strategy: Cycle timing + Forecast + Technical confirmation.

This is the most sophisticated strategy that combines:
1. Cycle position (where are we in the halving cycle?)
2. Forecast direction (what does the model predict?)
3. Technical confirmation (is price action confirming?)

Signal logic:
- BUY: In buy zone + forecast bullish + technicals oversold
- SELL: In sell zone + forecast bearish + technicals overbought
- Stronger signals when all three agree
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, StrategySignal, classify_signal
from src.models.cycle_features import add_cycle_features, CyclePhase
from src.metrics import HALVING_DATES

if TYPE_CHECKING:
    from src.metrics import HalvingAverages


class CombinedStrategy(BaseStrategy):
    """Combined strategy using cycle + forecast + technicals.

    Three components weighted together:
    1. Cycle Score (30%): Position relative to predicted top/bottom
    2. Forecast Score (40%): Predicted price vs current price
    3. Technical Score (30%): RSI, MACD, price vs MA confirmation

    Signals are strongest when all three components agree.
    """

    def __init__(
        self,
        halving_averages: "HalvingAverages | None" = None,
        cycle_metrics: pd.DataFrame | None = None,
        cycle_weight: float = 0.30,
        forecast_weight: float = 0.40,
        technical_weight: float = 0.30,
        lookforward_days: list[int] | None = None,  # Multi-timeframe
        min_signal_agreement: int = 2,  # How many components must agree
    ):
        """Initialize the strategy.

        Args:
            halving_averages: Historical cycle averages.
            cycle_metrics: Per-cycle metrics for predictions.
            cycle_weight: Weight for cycle position score.
            forecast_weight: Weight for forecast score.
            technical_weight: Weight for technical score.
            lookforward_days: Forecast horizons to check (default: [7, 30, 90]).
            min_signal_agreement: Minimum components that must agree for signal.
        """
        self.halving_averages = halving_averages
        self.cycle_metrics = cycle_metrics
        self.cycle_weight = cycle_weight
        self.forecast_weight = forecast_weight
        self.technical_weight = technical_weight
        self.lookforward_days = lookforward_days or [7, 30, 90]
        self.min_signal_agreement = min_signal_agreement

    @property
    def name(self) -> str:
        return "Combined Strategy"

    @property
    def description(self) -> str:
        return "Cycle timing + Forecast direction + Technical confirmation"

    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate signals combining all three components."""
        self.validate_data(df)
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # Add cycle features
        df = add_cycle_features(df, date_col="ds")

        # Get timing predictions
        avg_days_to_top, avg_days_to_bottom, avg_days_before_low = self._get_timing_params()

        # Get predicted prices
        predicted_bottom_price, predicted_top_price = self._predict_prices()

        # Prepare forecast lookup if available
        forecast_lookups = {}
        if forecast is not None:
            forecast = forecast.copy()
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            forecast_col = "yhat_ensemble" if "yhat_ensemble" in forecast.columns else "yhat"
            forecast_lookups["current"] = forecast.set_index("ds")[forecast_col].to_dict()

            # Compute forecast momentum
            forecast_sorted = forecast.sort_values("ds")
            forecast_sorted["momentum_7d"] = forecast_sorted[forecast_col].pct_change(7) * 100
            forecast_sorted["momentum_30d"] = forecast_sorted[forecast_col].pct_change(30) * 100
            forecast_lookups["momentum_7d"] = forecast_sorted.set_index("ds")["momentum_7d"].to_dict()
            forecast_lookups["momentum_30d"] = forecast_sorted.set_index("ds")["momentum_30d"].to_dict()

        # Compute volatility for adaptive thresholds
        df["volatility"] = df["y"].pct_change().rolling(30).std() * np.sqrt(365) * 100
        df["volatility"] = df["volatility"].fillna(df["volatility"].median())

        # Initialize score columns
        df["cycle_score"] = 0.0
        df["forecast_score"] = 0.0
        df["technical_score"] = 0.0
        df["signal_score"] = 0.0
        df["components_agree"] = 0
        df["signal_reason"] = ""

        # Store for plotting
        df["predicted_bottom_price"] = predicted_bottom_price
        df["predicted_top_price"] = predicted_top_price

        for idx, row in df.iterrows():
            current_date = row["ds"]
            current_price = row["y"]
            days_since = row.get("days_since_halving") or 0
            days_until = row.get("days_until_halving") or 9999
            volatility = row.get("volatility", 50)

            if pd.isna(current_price) or current_price <= 0:
                continue

            # Adaptive threshold based on volatility
            threshold = max(5.0, volatility * 0.3)

            # =================================================================
            # 1. CYCLE SCORE: Where are we in the cycle?
            # =================================================================
            cycle_score, cycle_signal, cycle_reason = self._compute_cycle_score(
                days_since, days_until, current_price,
                avg_days_to_top, avg_days_to_bottom, avg_days_before_low,
                predicted_bottom_price, predicted_top_price
            )
            df.loc[idx, "cycle_score"] = cycle_score

            # =================================================================
            # 2. FORECAST SCORE: What does the model predict?
            # =================================================================
            forecast_score, forecast_signal, forecast_reason = self._compute_forecast_score(
                current_date, current_price, forecast_lookups, threshold
            )
            df.loc[idx, "forecast_score"] = forecast_score

            # Store forecast values for plotting
            if forecast_lookups:
                for horizon in self.lookforward_days:
                    target_date = current_date + pd.Timedelta(days=horizon)
                    pred = forecast_lookups["current"].get(target_date)
                    if pred:
                        df.loc[idx, f"forecast_{horizon}d"] = pred
                        df.loc[idx, f"forecast_{horizon}d_pct"] = (pred / current_price - 1) * 100

            # =================================================================
            # 3. TECHNICAL SCORE: Is price action confirming?
            # =================================================================
            # (Computed after loop for efficiency)

        # Compute technical indicators on full series
        df = self._add_technical_scores(df)

        # Combine scores
        for idx, row in df.iterrows():
            cycle_score = row["cycle_score"]
            forecast_score = row["forecast_score"]
            technical_score = row["technical_score"]

            # Count agreeing components
            signals = []
            if cycle_score > 0.2:
                signals.append(1)  # Bullish
            elif cycle_score < -0.2:
                signals.append(-1)  # Bearish
            else:
                signals.append(0)  # Neutral

            if forecast_score > 0.2:
                signals.append(1)
            elif forecast_score < -0.2:
                signals.append(-1)
            else:
                signals.append(0)

            if technical_score > 0.2:
                signals.append(1)
            elif technical_score < -0.2:
                signals.append(-1)
            else:
                signals.append(0)

            # Count agreement
            bullish_count = signals.count(1)
            bearish_count = signals.count(-1)
            df.loc[idx, "components_agree"] = max(bullish_count, bearish_count)

            # Weighted combination
            raw_score = (
                cycle_score * self.cycle_weight +
                forecast_score * self.forecast_weight +
                technical_score * self.technical_weight
            )

            # Boost if components agree, dampen if they disagree
            if bullish_count >= self.min_signal_agreement:
                final_score = max(raw_score, 0.3) * (1 + 0.2 * (bullish_count - 2))
            elif bearish_count >= self.min_signal_agreement:
                final_score = min(raw_score, -0.3) * (1 + 0.2 * (bearish_count - 2))
            elif bullish_count >= 1 and bearish_count >= 1:
                # Conflicting signals - reduce magnitude
                final_score = raw_score * 0.5
            else:
                final_score = raw_score

            df.loc[idx, "signal_score"] = np.clip(final_score, -1, 1)

            # Generate reason
            reasons = []
            if abs(cycle_score) > 0.2:
                reasons.append(f"cycle={'bullish' if cycle_score > 0 else 'bearish'}")
            if abs(forecast_score) > 0.2:
                reasons.append(f"forecast={'bullish' if forecast_score > 0 else 'bearish'}")
            if abs(technical_score) > 0.2:
                reasons.append(f"technicals={'bullish' if technical_score > 0 else 'bearish'}")

            if reasons:
                agreement = f"({max(bullish_count, bearish_count)}/3 agree)"
                action = "BUY" if final_score > 0.2 else ("SELL" if final_score < -0.2 else "HOLD")
                df.loc[idx, "signal_reason"] = f"{action} - {', '.join(reasons)} {agreement}"
            else:
                df.loc[idx, "signal_reason"] = "HOLD - No clear signal"

        # Classify signals
        df["signal"] = df["signal_score"].apply(lambda s: classify_signal(s).value)

        # Mark buy/sell zones for plotting
        df["buy_zone"] = df["cycle_score"] > 0.3
        df["sell_zone"] = df["cycle_score"] < -0.3

        return df

    def _compute_cycle_score(
        self,
        days_since: float,
        days_until: float,
        current_price: float,
        avg_days_to_top: float,
        avg_days_to_bottom: float,
        avg_days_before_low: float,
        predicted_bottom_price: float | None,
        predicted_top_price: float | None,
    ) -> tuple[float, int, str]:
        """Compute cycle position score."""
        score = 0.0
        signal = 0  # -1=bearish, 0=neutral, 1=bullish
        reason = ""

        # Buy zone: near predicted bottom (before halving)
        buy_zone_center = avg_days_before_low
        dist_from_buy_center = abs(days_until - buy_zone_center)
        if dist_from_buy_center < 150:
            buy_strength = np.exp(-(dist_from_buy_center ** 2) / (2 * 100 ** 2))
            score += 0.5 * buy_strength
            reason = f"buy zone ({int(days_until)}d to halving)"

        # Sell zone: near predicted top (after halving)
        sell_zone_center = avg_days_to_top
        dist_from_sell_center = abs(days_since - sell_zone_center)
        if days_since > 90 and dist_from_sell_center < 150:
            sell_strength = np.exp(-(dist_from_sell_center ** 2) / (2 * 100 ** 2))
            score -= 0.5 * sell_strength
            reason = f"sell zone ({int(days_since)}d since halving)"

        # Price confirmation
        if predicted_bottom_price and current_price > 0:
            price_vs_bottom = current_price / predicted_bottom_price
            if 0.8 <= price_vs_bottom <= 1.2:
                score += 0.2  # Near predicted bottom
                reason += ", near predicted bottom"

        if predicted_top_price and current_price > 0:
            price_vs_top = current_price / predicted_top_price
            if 0.8 <= price_vs_top <= 1.2:
                score -= 0.2  # Near predicted top
                reason += ", near predicted top"

        signal = 1 if score > 0.2 else (-1 if score < -0.2 else 0)
        return np.clip(score, -1, 1), signal, reason

    def _compute_forecast_score(
        self,
        current_date: pd.Timestamp,
        current_price: float,
        forecast_lookups: dict,
        threshold: float,
    ) -> tuple[float, int, str]:
        """Compute forecast-based score using multi-timeframe analysis."""
        if not forecast_lookups:
            return 0.0, 0, "no forecast"

        score = 0.0
        signals = []
        reasons = []

        # Check multiple horizons
        weights = {7: 0.2, 30: 0.5, 90: 0.3}  # Weight longer-term more

        for horizon, weight in weights.items():
            target_date = current_date + pd.Timedelta(days=horizon)
            predicted = forecast_lookups["current"].get(target_date)

            if predicted is None:
                continue

            pct_change = (predicted / current_price - 1) * 100

            if abs(pct_change) >= threshold:
                horizon_score = np.clip(pct_change / 30, -1, 1)  # 30% = max score
                score += horizon_score * weight
                signals.append(1 if pct_change > 0 else -1)
                reasons.append(f"{horizon}d: {pct_change:+.1f}%")

        # Momentum confirmation
        mom_7d = forecast_lookups.get("momentum_7d", {}).get(current_date, 0)
        mom_30d = forecast_lookups.get("momentum_30d", {}).get(current_date, 0)

        if mom_7d > 2 and mom_30d > 0:
            score += 0.15
            reasons.append("rising momentum")
        elif mom_7d < -2 and mom_30d < 0:
            score -= 0.15
            reasons.append("falling momentum")

        signal = 1 if score > 0.2 else (-1 if score < -0.2 else 0)
        reason = ", ".join(reasons) if reasons else "neutral forecast"

        return np.clip(score, -1, 1), signal, reason

    def _add_technical_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator scores to dataframe."""
        prices = df["y"]

        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # RSI score: oversold = bullish, overbought = bearish
        df["rsi_score"] = 0.0
        df.loc[df["rsi"] < 30, "rsi_score"] = 0.5  # Oversold = bullish
        df.loc[df["rsi"] < 20, "rsi_score"] = 0.8  # Very oversold
        df.loc[df["rsi"] > 70, "rsi_score"] = -0.5  # Overbought = bearish
        df.loc[df["rsi"] > 80, "rsi_score"] = -0.8  # Very overbought

        # MACD
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()
        df["macd_histogram"] = macd - macd_signal

        # MACD score: histogram direction
        hist_std = df["macd_histogram"].rolling(50).std().replace(0, 1)
        df["macd_score"] = (df["macd_histogram"] / hist_std).clip(-1, 1) * 0.5

        # Price vs Moving Averages
        df["ma_50"] = prices.rolling(50).mean()
        df["ma_200"] = prices.rolling(200).mean()

        df["ma_score"] = 0.0
        # Price above both MAs = bullish
        df.loc[(prices > df["ma_50"]) & (prices > df["ma_200"]), "ma_score"] = 0.3
        # Golden cross (50 > 200) = more bullish
        df.loc[(df["ma_50"] > df["ma_200"]) & (prices > df["ma_50"]), "ma_score"] = 0.5
        # Price below both MAs = bearish
        df.loc[(prices < df["ma_50"]) & (prices < df["ma_200"]), "ma_score"] = -0.3
        # Death cross (50 < 200) = more bearish
        df.loc[(df["ma_50"] < df["ma_200"]) & (prices < df["ma_50"]), "ma_score"] = -0.5

        # Combined technical score
        df["technical_score"] = (
            df["rsi_score"].fillna(0) * 0.4 +
            df["macd_score"].fillna(0) * 0.3 +
            df["ma_score"].fillna(0) * 0.3
        ).clip(-1, 1)

        return df

    def _get_timing_params(self) -> tuple[float, float, float]:
        """Get timing parameters from cycle_metrics or halving_averages."""
        if self.cycle_metrics is not None and len(self.cycle_metrics) >= 1:
            avg_days_to_top = self.cycle_metrics["days_after_halving_to_high"].mean()
            avg_days_to_bottom = self.cycle_metrics["days_after_halving_to_low"].mean()

            days_before = []
            for _, row in self.cycle_metrics.iterrows():
                halving = row["halving_date"]
                pre_low = row.get("pre_low_date")
                if pd.notna(pre_low):
                    days_before.append((halving - pre_low).days)
            avg_days_before_low = np.mean(days_before) if days_before else 400

            return avg_days_to_top, avg_days_to_bottom, avg_days_before_low

        if self.halving_averages and self.halving_averages.n_cycles > 0:
            return (
                self.halving_averages.avg_days_to_top or 365,
                self.halving_averages.avg_days_to_bottom or 500,
                self.halving_averages.avg_days_before_low or 400,
            )

        return 365, 500, 400

    def _predict_prices(self) -> tuple[float | None, float | None]:
        """Predict bottom and top prices for next cycle.

        Uses decay model for drawdown and recent cycles for run-up estimate.
        Early cycles had 10,000%+ gains which would skew predictions unrealistically.
        """
        if self.cycle_metrics is None or len(self.cycle_metrics) < 2:
            return None, None

        try:
            from src.metrics import predict_drawdown

            last_cycle = self.cycle_metrics.iloc[-1]
            last_top_price = last_cycle["post_high_price"]

            # Predict drawdown using decay model
            prediction = predict_drawdown(self.cycle_metrics)
            predicted_drawdown_pct = prediction.predicted_value

            predicted_bottom = last_top_price * (1 - predicted_drawdown_pct)

            # Use only last 2 cycles for run-up estimate (early cycles too extreme)
            recent_cycles = self.cycle_metrics.tail(2)
            recent_runup = recent_cycles["run_up_pct"].mean() / 100

            # Cap run-up at 500% (5x) to be realistic for mature BTC market
            capped_runup = min(recent_runup, 5.0)

            # Further dampen expectation (diminishing returns)
            dampened_runup = capped_runup * 0.5

            predicted_top = predicted_bottom * (1 + dampened_runup)

            return predicted_bottom, predicted_top

        except Exception:
            return None, None

    def get_current_signal(self, df: pd.DataFrame) -> StrategySignal:
        """Get the most recent signal."""
        if df.empty:
            raise ValueError("Empty DataFrame")

        latest = df.loc[df["ds"].idxmax()]

        return StrategySignal(
            date=latest["ds"],
            signal=Signal(latest.get("signal", Signal.HOLD.value)),
            score=latest.get("signal_score", 0),
            reason=latest.get("signal_reason", ""),
            metadata={
                "cycle_score": latest.get("cycle_score"),
                "forecast_score": latest.get("forecast_score"),
                "technical_score": latest.get("technical_score"),
                "components_agree": latest.get("components_agree"),
                "rsi": latest.get("rsi"),
                "buy_zone": latest.get("buy_zone"),
                "sell_zone": latest.get("sell_zone"),
            },
        )
