"""Halving cycle signal strategy.

This is the primary strategy that uses Bitcoin's 4-year halving cycle
to generate buy/sell signals based on proximity to predicted tops and bottoms.

Strategy:
- BUY near the predicted cycle bottom (late bear market)
- SELL near the predicted cycle top (bull run peak)

Uses cycle_metrics (per-halving historical data) for precise predictions:
- post_high_date/price: When the cycle top occurred
- post_low_date/price: When the cycle bottom occurred
- days_after_halving_to_high: Days from halving to top
- days_after_halving_to_low: Days from halving to bottom
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, StrategySignal, classify_signal
from src.models.cycle_features import add_cycle_features, CyclePhase
from src.metrics import HALVING_DATES

if TYPE_CHECKING:
    from src.metrics import HalvingAverages


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_ma_crossover(prices: pd.Series, short: int = 50, long: int = 200) -> pd.Series:
    """Compute moving average crossover signal (-1 to 1)."""
    ma_short = prices.rolling(window=short).mean()
    ma_long = prices.rolling(window=long).mean()
    crossover = (ma_short - ma_long) / ma_long
    return crossover.clip(-0.5, 0.5) * 2


class CycleSignalStrategy(BaseStrategy):
    """Strategy based on Bitcoin's 4-year halving cycle.

    Core logic:
    - BUY when near predicted cycle bottom (late bear market, before halving)
    - SELL when near predicted cycle top (bull run, after halving)

    Uses historical averages to predict:
    - Cycle top: ~avg_days_to_top days after halving
    - Cycle bottom: ~avg_days_to_bottom days after halving (or before next halving)

    Technical indicators (RSI, MACD, MA) provide additional confirmation.
    """

    def __init__(
        self,
        cycle_weight: float = 0.6,
        technical_weight: float = 0.4,
        include_technicals: bool = True,
        halving_averages: "HalvingAverages | None" = None,
        cycle_metrics: pd.DataFrame | None = None,
        buy_window_days: int = 120,   # Window around predicted bottom
        sell_window_days: int = 120,  # Window around predicted top
    ):
        """Initialize the strategy.

        Args:
            cycle_weight: Weight for cycle-based signals (0-1).
            technical_weight: Weight for technical indicators (0-1).
            include_technicals: Whether to include RSI/MACD/MA.
            halving_averages: Historical averages for timing predictions.
            cycle_metrics: Per-cycle metrics for more precise predictions.
            buy_window_days: Days before/after predicted bottom to signal buy.
            sell_window_days: Days before/after predicted top to signal sell.
        """
        self.cycle_weight = cycle_weight
        self.technical_weight = technical_weight
        self.include_technicals = include_technicals
        self.halving_averages = halving_averages
        self.cycle_metrics = cycle_metrics
        self.buy_window_days = buy_window_days
        self.sell_window_days = sell_window_days

    @property
    def name(self) -> str:
        return "Halving Cycle Strategy"

    @property
    def description(self) -> str:
        return "Buy near predicted bottom, sell near predicted top"

    def generate_signals(
        self,
        df: pd.DataFrame,
        forecast: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate signals based on proximity to predicted tops/bottoms."""
        self.validate_data(df)
        df = df.copy()

        # Add cycle features
        df = add_cycle_features(df, date_col="ds")

        # Get timing predictions from cycle_metrics (per-halving) or halving_averages
        avg_days_to_top, avg_days_to_bottom, avg_days_before_low = self._get_timing_params()

        # If we have cycle_metrics, compute predicted top/bottom prices too
        predicted_bottom_price = None
        predicted_top_price = None
        if self.cycle_metrics is not None and len(self.cycle_metrics) >= 2:
            predicted_bottom_price, predicted_top_price = self._predict_prices()

        # Initialize cycle score
        df["cycle_score"] = 0.0

        # For each row, compute proximity to predicted top and bottom
        days_since = df["days_since_halving"].fillna(0)
        days_until = df["days_until_halving"].fillna(9999)

        # =================================================================
        # BUY SIGNAL: Near predicted cycle bottom
        # =================================================================
        # Bottom typically occurs avg_days_before_low days before halving
        # (or avg_days_to_bottom days after the PREVIOUS halving)

        buy_zone_center = avg_days_before_low
        buy_zone_start = buy_zone_center + self.buy_window_days
        buy_zone_end = max(buy_zone_center - self.buy_window_days, 60)

        # Gaussian-weighted buy signal
        in_buy_zone = (days_until >= buy_zone_end) & (days_until <= buy_zone_start)
        if in_buy_zone.any():
            dist_from_bottom = np.abs(days_until[in_buy_zone] - buy_zone_center)
            spread = self.buy_window_days
            buy_weight = np.exp(-(dist_from_bottom ** 2) / (2 * spread ** 2))
            df.loc[in_buy_zone, "cycle_score"] = 0.5 + 0.5 * buy_weight

        df["buy_zone"] = in_buy_zone
        df["days_to_predicted_bottom"] = np.abs(days_until - buy_zone_center)

        # Also buy in late drawdown (catches bottom if later than typical)
        late_drawdown = (days_since > avg_days_to_bottom - 100) & (days_since < avg_days_to_bottom + 200)
        if late_drawdown.any():
            dist_from_late_bottom = np.abs(days_since[late_drawdown] - avg_days_to_bottom)
            late_buy_weight = np.exp(-(dist_from_late_bottom ** 2) / (2 * 100 ** 2))
            mask = late_drawdown & ~in_buy_zone
            if mask.any():
                df.loc[mask, "cycle_score"] = np.maximum(
                    df.loc[mask, "cycle_score"],
                    0.4 + 0.3 * late_buy_weight[late_drawdown][~in_buy_zone[late_drawdown]]
                )

        # Price-based buy boost: if price is near predicted bottom price
        if predicted_bottom_price is not None and "y" in df.columns:
            price_ratio = df["y"] / predicted_bottom_price
            # Boost buy signal when price is within 20% of predicted bottom
            near_bottom_price = (price_ratio >= 0.8) & (price_ratio <= 1.2)
            df.loc[near_bottom_price & in_buy_zone, "cycle_score"] += 0.2

        # =================================================================
        # SELL SIGNAL: Near predicted cycle top
        # =================================================================
        sell_zone_center = avg_days_to_top
        sell_zone_start = max(sell_zone_center - self.sell_window_days, 90)
        sell_zone_end = sell_zone_center + self.sell_window_days

        in_sell_zone = (days_since >= sell_zone_start) & (days_since <= sell_zone_end)
        if in_sell_zone.any():
            dist_from_top = np.abs(days_since[in_sell_zone] - sell_zone_center)
            spread = self.sell_window_days
            sell_weight = np.exp(-(dist_from_top ** 2) / (2 * spread ** 2))
            df.loc[in_sell_zone, "cycle_score"] = -0.5 - 0.5 * sell_weight

        df["sell_zone"] = in_sell_zone
        df["days_to_predicted_top"] = np.abs(days_since - sell_zone_center)

        # Late distribution: past sell zone
        late_distribution = days_since > sell_zone_end
        if late_distribution.any():
            days_past_zone = days_since[late_distribution] - sell_zone_end
            late_sell_score = np.minimum(-0.3 - days_past_zone / 500, -0.7)
            df.loc[late_distribution, "cycle_score"] = late_sell_score

        # Price-based sell boost: if price is near predicted top price
        if predicted_top_price is not None and "y" in df.columns:
            price_ratio = df["y"] / predicted_top_price
            # Boost sell signal when price is within 20% of predicted top
            near_top_price = (price_ratio >= 0.8) & (price_ratio <= 1.2)
            df.loc[near_top_price & in_sell_zone, "cycle_score"] -= 0.2

        # =================================================================
        # NEUTRAL ZONES
        # =================================================================
        consolidation = (days_since > 0) & (days_since < sell_zone_start)
        df.loc[consolidation, "cycle_score"] = df.loc[consolidation, "cycle_score"].clip(-0.2, 0.2)

        pre_halving_runup = (days_until > 0) & (days_until < buy_zone_end)
        df.loc[pre_halving_runup, "cycle_score"] = df.loc[pre_halving_runup, "cycle_score"].clip(-0.1, 0.1)

        # =================================================================
        # TECHNICAL INDICATORS (confirmation)
        # =================================================================
        if self.include_technicals and "y" in df.columns:
            prices = df["y"]

            # RSI
            rsi = compute_rsi(prices)
            df["rsi"] = rsi
            df["rsi_score"] = (50 - rsi) / 50  # Oversold = positive

            # MACD
            macd_line, signal_line, histogram = compute_macd(prices)
            df["macd"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_histogram"] = histogram
            hist_std = histogram.rolling(window=50).std().replace(0, 1)
            df["macd_score"] = (histogram / hist_std).clip(-2, 2) / 2

            # MA Crossover
            df["ma_crossover"] = compute_ma_crossover(prices)
            df["ma_score"] = df["ma_crossover"]

            # Combined technical score
            df["technical_score"] = (
                df["rsi_score"].fillna(0) * 0.3
                + df["macd_score"].fillna(0) * 0.4
                + df["ma_score"].fillna(0) * 0.3
            )
        else:
            df["technical_score"] = 0

        # =================================================================
        # COMBINE CYCLE + TECHNICALS
        # =================================================================
        total = self.cycle_weight + self.technical_weight
        cycle_w = self.cycle_weight / total
        tech_w = self.technical_weight / total

        df["signal_score"] = (
            df["cycle_score"] * cycle_w + df["technical_score"] * tech_w
        ).clip(-1, 1)

        # Classify signals
        df["signal"] = df["signal_score"].apply(lambda s: classify_signal(s).value)

        # Generate reasons
        df["signal_reason"] = df.apply(self._get_reason, axis=1)

        return df

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
                "cycle_phase": latest.get("cycle_phase"),
                "days_until_halving": latest.get("days_until_halving"),
                "days_since_halving": latest.get("days_since_halving"),
                "buy_zone": latest.get("buy_zone"),
                "sell_zone": latest.get("sell_zone"),
                "days_to_predicted_bottom": latest.get("days_to_predicted_bottom"),
                "days_to_predicted_top": latest.get("days_to_predicted_top"),
                "rsi": latest.get("rsi"),
            },
        )

    def _get_timing_params(self) -> tuple[float, float, float]:
        """Get timing parameters from cycle_metrics or halving_averages.

        Returns:
            Tuple of (avg_days_to_top, avg_days_to_bottom, avg_days_before_low)
        """
        # Prefer cycle_metrics for more accurate per-cycle data
        if self.cycle_metrics is not None and len(self.cycle_metrics) >= 1:
            # Use average of historical cycles
            avg_days_to_top = self.cycle_metrics["days_after_halving_to_high"].mean()
            avg_days_to_bottom = self.cycle_metrics["days_after_halving_to_low"].mean()

            # Days before halving to the low (compute from halving_date - pre_low_date)
            days_before = []
            for _, row in self.cycle_metrics.iterrows():
                halving = row["halving_date"]
                pre_low = row.get("pre_low_date")
                if pd.notna(pre_low):
                    days_before.append((halving - pre_low).days)
            avg_days_before_low = np.mean(days_before) if days_before else 400

            return avg_days_to_top, avg_days_to_bottom, avg_days_before_low

        # Fall back to halving_averages
        if self.halving_averages and self.halving_averages.n_cycles > 0:
            return (
                self.halving_averages.avg_days_to_top or 365,
                self.halving_averages.avg_days_to_bottom or 500,
                self.halving_averages.avg_days_before_low or 400,
            )

        # Defaults
        return 365, 500, 400

    def _predict_prices(self) -> tuple[float | None, float | None]:
        """Predict bottom and top prices for next cycle using historical patterns.

        Uses decay model for drawdown, run-up patterns for gains.

        Returns:
            Tuple of (predicted_bottom_price, predicted_top_price)
        """
        if self.cycle_metrics is None or len(self.cycle_metrics) < 2:
            return None, None

        try:
            from src.metrics import predict_drawdown

            # Get last cycle's top as reference
            last_cycle = self.cycle_metrics.iloc[-1]
            last_top_price = last_cycle["post_high_price"]

            # Predict drawdown for next cycle
            prediction = predict_drawdown(self.cycle_metrics)
            predicted_drawdown_pct = prediction.predicted_value  # As decimal (e.g., 0.5 for 50%)

            # Predicted bottom = last top * (1 - drawdown)
            predicted_bottom = last_top_price * (1 - predicted_drawdown_pct)

            # Predicted top = predicted bottom * (1 + avg run-up)
            avg_runup = self.cycle_metrics["run_up_pct"].mean() / 100  # As decimal
            # Dampen run-up expectation (diminishing returns hypothesis)
            dampened_runup = avg_runup * 0.7
            predicted_top = predicted_bottom * (1 + dampened_runup)

            return predicted_bottom, predicted_top

        except Exception:
            return None, None

    def _get_reason(self, row: pd.Series) -> str:
        """Generate human-readable signal reason."""
        buy_zone = row.get("buy_zone", False)
        sell_zone = row.get("sell_zone", False)
        days_until = row.get("days_until_halving")
        days_since = row.get("days_since_halving")
        days_to_bottom = row.get("days_to_predicted_bottom", 9999)
        days_to_top = row.get("days_to_predicted_top", 9999)
        score = row.get("signal_score", 0)

        if buy_zone and score > 0.3:
            if days_to_bottom < 30:
                return f"STRONG BUY - Near predicted cycle bottom ({int(days_to_bottom)}d away)"
            return f"BUY - In accumulation zone ({int(days_until)}d to halving)"

        if sell_zone and score < -0.3:
            if days_to_top < 30:
                return f"STRONG SELL - Near predicted cycle top ({int(days_to_top)}d away)"
            return f"SELL - In distribution zone ({int(days_since)}d since halving)"

        if days_since and days_since > 0:
            if days_since < 90:
                return f"HOLD - Post-halving consolidation ({int(days_since)}d)"
            if sell_zone:
                return f"TRIM - Approaching predicted top ({int(days_to_top)}d away)"
            if days_since > 500:
                return f"ACCUMULATE - Late bear market, approaching bottom"

        if days_until and days_until > 0:
            if days_until < 60:
                return f"HOLD - Halving imminent ({int(days_until)}d)"
            if buy_zone:
                return f"ACCUMULATE - Approaching predicted bottom ({int(days_to_bottom)}d away)"

        return "HOLD"
