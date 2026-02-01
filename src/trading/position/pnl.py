"""PnL calculation and equity tracking."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EquityPoint:
    """Single point in equity curve.

    Attributes:
        timestamp: When this measurement was taken
        equity: Total account equity
        cash: Cash balance
        position_value: Value of open positions
        unrealized_pnl: Unrealized P&L
        realized_pnl: Cumulative realized P&L
    """

    timestamp: datetime
    equity: float
    cash: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "cash": self.cash,
            "position_value": self.position_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }


class PnLTracker:
    """Tracks profit/loss and equity over time.

    Maintains equity curve history and calculates various
    performance metrics like drawdown, daily/weekly P&L.

    Attributes:
        initial_capital: Starting capital
        equity_curve: List of equity points over time
        peak_equity: Highest equity achieved
        daily_pnl_history: Daily P&L tracking
        weekly_pnl_history: Weekly P&L tracking
    """

    def __init__(self, initial_capital: float):
        """Initialize P&L tracker.

        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.equity_curve: list[EquityPoint] = []
        self.peak_equity = initial_capital
        self.realized_pnl = 0.0

        # Daily/weekly tracking
        self._daily_start_equity: float | None = None
        self._weekly_start_equity: float | None = None
        self._day_start: datetime | None = None
        self._week_start: datetime | None = None

        # Record initial point
        self.record_equity(
            equity=initial_capital,
            cash=initial_capital,
            position_value=0.0,
            unrealized_pnl=0.0,
        )

    def record_equity(
        self,
        equity: float,
        cash: float,
        position_value: float,
        unrealized_pnl: float,
    ) -> EquityPoint:
        """Record current equity state.

        Args:
            equity: Total equity value
            cash: Cash balance
            position_value: Value of positions
            unrealized_pnl: Unrealized P&L

        Returns:
            Created equity point
        """
        now = datetime.now()

        # Check for new day/week
        self._check_period_rollover(now, equity)

        point = EquityPoint(
            timestamp=now,
            equity=equity,
            cash=cash,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
        )

        self.equity_curve.append(point)

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        return point

    def _check_period_rollover(self, now: datetime, current_equity: float) -> None:
        """Check if we've entered a new day or week.

        Args:
            now: Current timestamp
            current_equity: Current equity for starting new period
        """
        today = now.date()

        # Daily rollover
        if self._day_start is None or self._day_start.date() != today:
            self._day_start = now
            self._daily_start_equity = current_equity

        # Weekly rollover (Monday start)
        week_start = today - timedelta(days=today.weekday())
        if self._week_start is None or self._week_start.date() < week_start:
            self._week_start = now
            self._weekly_start_equity = current_equity

    def record_realized_pnl(self, pnl: float) -> None:
        """Record realized P&L from a closed position.

        Args:
            pnl: Realized P&L amount
        """
        self.realized_pnl += pnl
        logger.debug(f"Recorded realized P&L: {pnl:.2f}, total: {self.realized_pnl:.2f}")

    @property
    def current_equity(self) -> float:
        """Get most recent equity value."""
        if not self.equity_curve:
            return self.initial_capital
        return self.equity_curve[-1].equity

    @property
    def current_drawdown(self) -> float:
        """Calculate current drawdown from peak.

        Returns:
            Drawdown as decimal (0.05 = 5% drawdown)
        """
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown as percentage."""
        return self.current_drawdown * 100

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve.

        Returns:
            Maximum drawdown as decimal
        """
        if len(self.equity_curve) < 2:
            return 0.0

        peak = self.equity_curve[0].equity
        max_dd = 0.0

        for point in self.equity_curve:
            if point.equity > peak:
                peak = point.equity
            dd = (peak - point.equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    @property
    def daily_pnl(self) -> float:
        """Get P&L for current day.

        Returns:
            Daily P&L in currency units
        """
        if self._daily_start_equity is None:
            return 0.0
        return self.current_equity - self._daily_start_equity

    @property
    def daily_pnl_pct(self) -> float:
        """Get daily P&L as percentage.

        Returns:
            Daily P&L as percentage
        """
        if self._daily_start_equity is None or self._daily_start_equity == 0:
            return 0.0
        return (self.daily_pnl / self._daily_start_equity) * 100

    @property
    def weekly_pnl(self) -> float:
        """Get P&L for current week.

        Returns:
            Weekly P&L in currency units
        """
        if self._weekly_start_equity is None:
            return 0.0
        return self.current_equity - self._weekly_start_equity

    @property
    def weekly_pnl_pct(self) -> float:
        """Get weekly P&L as percentage.

        Returns:
            Weekly P&L as percentage
        """
        if self._weekly_start_equity is None or self._weekly_start_equity == 0:
            return 0.0
        return (self.weekly_pnl / self._weekly_start_equity) * 100

    @property
    def total_return(self) -> float:
        """Get total return since inception.

        Returns:
            Total return in currency units
        """
        return self.current_equity - self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Get total return as percentage.

        Returns:
            Total return as percentage
        """
        if self.initial_capital == 0:
            return 0.0
        return (self.total_return / self.initial_capital) * 100

    def get_equity_curve_df(self) -> "pd.DataFrame":
        """Get equity curve as pandas DataFrame.

        Returns:
            DataFrame with equity history
        """
        import pandas as pd

        if not self.equity_curve:
            return pd.DataFrame()

        data = [p.to_dict() for p in self.equity_curve]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive P&L metrics.

        Returns:
            Dictionary of performance metrics
        """
        return {
            "initial_capital": self.initial_capital,
            "current_equity": self.current_equity,
            "total_return": self.total_return,
            "total_return_pct": self.total_return_pct,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.equity_curve[-1].unrealized_pnl if self.equity_curve else 0,
            "peak_equity": self.peak_equity,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown * 100,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "weekly_pnl": self.weekly_pnl,
            "weekly_pnl_pct": self.weekly_pnl_pct,
            "equity_points": len(self.equity_curve),
        }

    def reset_daily(self) -> None:
        """Reset daily P&L tracking (call at day start)."""
        self._daily_start_equity = self.current_equity
        self._day_start = datetime.now()

    def reset_weekly(self) -> None:
        """Reset weekly P&L tracking (call at week start)."""
        self._weekly_start_equity = self.current_equity
        self._week_start = datetime.now()
