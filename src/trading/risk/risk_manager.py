"""Risk management for live trading."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from src.trading.config import TradingConfig
from src.trading.position.position_tracker import Position, PositionTracker
from src.trading.position.pnl import PnLTracker

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level indicators."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskCheckResult:
    """Result of a risk check.

    Attributes:
        allowed: Whether the action is allowed
        reason: Explanation for the decision
        risk_level: Current risk level
        limits_triggered: List of limits that were triggered
    """

    allowed: bool
    reason: str
    risk_level: RiskLevel = RiskLevel.NORMAL
    limits_triggered: list[str] = field(default_factory=list)


class RiskManager:
    """Manages trading risk limits and controls.

    Enforces:
    - Position size limits
    - Total exposure limits
    - Daily/weekly loss limits
    - Maximum drawdown (kill switch)
    - Per-position stop losses

    Attributes:
        config: Trading configuration
        positions: Position tracker reference
        pnl: P&L tracker reference
        kill_switch_active: Whether emergency stop is engaged
        trading_paused: Whether trading is paused
        pause_until: When trading pause expires
    """

    def __init__(
        self,
        config: TradingConfig,
        position_tracker: PositionTracker,
        pnl_tracker: PnLTracker,
    ):
        """Initialize risk manager.

        Args:
            config: Trading configuration
            position_tracker: Position tracker instance
            pnl_tracker: P&L tracker instance
        """
        self.config = config
        self.positions = position_tracker
        self.pnl = pnl_tracker

        self.kill_switch_active = False
        self.kill_switch_reason: str | None = None
        self.kill_switch_time: datetime | None = None

        self.trading_paused = False
        self.pause_reason: str | None = None
        self.pause_until: datetime | None = None

        self._last_trade_time: datetime | None = None

    def check_pre_trade(
        self,
        side: str,
        order_size_usd: float,
        signal_score: float,
    ) -> RiskCheckResult:
        """Perform pre-trade risk checks.

        Args:
            side: "BUY" or "SELL"
            order_size_usd: Proposed order size in USD
            signal_score: Signal strength (-1 to +1)

        Returns:
            RiskCheckResult with decision and explanation
        """
        limits_triggered = []

        # Check kill switch
        if self.kill_switch_active:
            return RiskCheckResult(
                allowed=False,
                reason=f"Kill switch active: {self.kill_switch_reason}",
                risk_level=RiskLevel.CRITICAL,
                limits_triggered=["kill_switch"],
            )

        # Check trading pause
        if self.trading_paused:
            if self.pause_until and datetime.now() < self.pause_until:
                remaining = self.pause_until - datetime.now()
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Trading paused: {self.pause_reason} (resumes in {remaining})",
                    risk_level=RiskLevel.HIGH,
                    limits_triggered=["trading_pause"],
                )
            else:
                # Pause expired
                self._resume_trading()

        # Check minimum trade interval
        if self._last_trade_time:
            min_interval = timedelta(hours=self.config.min_trade_interval_hours)
            if datetime.now() - self._last_trade_time < min_interval:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Minimum trade interval not met ({self.config.min_trade_interval_hours}h)",
                    risk_level=RiskLevel.NORMAL,
                    limits_triggered=["trade_interval"],
                )

        # Check minimum signal score
        if abs(signal_score) < self.config.min_signal_score:
            return RiskCheckResult(
                allowed=False,
                reason=f"Signal score {signal_score:.2f} below minimum {self.config.min_signal_score}",
                risk_level=RiskLevel.NORMAL,
                limits_triggered=["min_signal"],
            )

        # Check minimum trade size
        if order_size_usd < self.config.min_trade_usd:
            return RiskCheckResult(
                allowed=False,
                reason=f"Order size ${order_size_usd:.2f} below minimum ${self.config.min_trade_usd}",
                risk_level=RiskLevel.NORMAL,
                limits_triggered=["min_trade_size"],
            )

        # For BUY orders, check position limits
        if side == "BUY":
            # Check position size limit
            max_position = self.config.initial_capital * self.config.max_position_pct
            if order_size_usd > max_position:
                limits_triggered.append("position_size")
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Order ${order_size_usd:.2f} exceeds max position ${max_position:.2f}",
                    risk_level=RiskLevel.ELEVATED,
                    limits_triggered=limits_triggered,
                )

            # Check total exposure limit
            current_exposure = float(self.positions.get_total_exposure())
            max_exposure = self.config.initial_capital * self.config.max_total_exposure_pct
            new_exposure = current_exposure + order_size_usd

            if new_exposure > max_exposure:
                limits_triggered.append("total_exposure")
                return RiskCheckResult(
                    allowed=False,
                    reason=(
                        f"Total exposure ${new_exposure:.2f} would exceed "
                        f"max ${max_exposure:.2f}"
                    ),
                    risk_level=RiskLevel.ELEVATED,
                    limits_triggered=limits_triggered,
                )

        # Check daily loss limit
        if self.pnl.daily_pnl_pct <= -self.config.daily_loss_limit_pct * 100:
            limits_triggered.append("daily_loss")
            self._pause_trading(
                reason="Daily loss limit hit",
                duration_hours=24,
            )
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily loss {self.pnl.daily_pnl_pct:.2f}% exceeds limit",
                risk_level=RiskLevel.HIGH,
                limits_triggered=limits_triggered,
            )

        # Check weekly loss limit
        if self.pnl.weekly_pnl_pct <= -self.config.weekly_loss_limit_pct * 100:
            limits_triggered.append("weekly_loss")
            self._pause_trading(
                reason="Weekly loss limit hit",
                duration_hours=168,  # 7 days
            )
            return RiskCheckResult(
                allowed=False,
                reason=f"Weekly loss {self.pnl.weekly_pnl_pct:.2f}% exceeds limit",
                risk_level=RiskLevel.HIGH,
                limits_triggered=limits_triggered,
            )

        # Determine risk level based on metrics
        risk_level = self._assess_risk_level()

        return RiskCheckResult(
            allowed=True,
            reason="All risk checks passed",
            risk_level=risk_level,
            limits_triggered=[],
        )

    def _assess_risk_level(self) -> RiskLevel:
        """Assess current overall risk level.

        Returns:
            Current risk level
        """
        drawdown = self.pnl.current_drawdown

        # Critical: approaching kill switch
        if drawdown >= self.config.max_drawdown_pct * 0.8:
            return RiskLevel.CRITICAL

        # High: significant drawdown or losses
        if drawdown >= self.config.max_drawdown_pct * 0.5:
            return RiskLevel.HIGH

        # Elevated: moderate concerns
        if (
            drawdown >= self.config.max_drawdown_pct * 0.25
            or self.pnl.daily_pnl_pct <= -self.config.daily_loss_limit_pct * 50
        ):
            return RiskLevel.ELEVATED

        return RiskLevel.NORMAL

    def update_pnl(self, current_equity: float, cash: float, position_value: float) -> None:
        """Update P&L tracking and check limits.

        Should be called regularly (e.g., hourly) to track equity
        and check for limit breaches.

        Args:
            current_equity: Total current equity
            cash: Current cash balance
            position_value: Current value of positions
        """
        unrealized_pnl = position_value - float(self.positions.get_total_exposure())

        self.pnl.record_equity(
            equity=current_equity,
            cash=cash,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
        )

        # Check max drawdown for kill switch
        if self.pnl.current_drawdown >= self.config.max_drawdown_pct:
            self.activate_kill_switch(
                f"Max drawdown {self.pnl.current_drawdown_pct:.2f}% "
                f"exceeds limit {self.config.max_drawdown_pct * 100:.1f}%"
            )

    def check_stop_loss(self, position: Position, current_price: Decimal) -> bool:
        """Check if a position has hit its stop loss.

        Args:
            position: Position to check
            current_price: Current market price

        Returns:
            True if stop loss triggered
        """
        if position.stop_loss_price is None:
            return False

        triggered = current_price <= position.stop_loss_price

        if triggered:
            logger.warning(
                f"Stop loss triggered for {position.position_id}: "
                f"price {current_price} <= stop {position.stop_loss_price}"
            )

        return triggered

    def get_positions_at_stop_loss(self, current_price: Decimal) -> list[Position]:
        """Get all positions that have hit stop loss.

        Args:
            current_price: Current market price

        Returns:
            List of positions at stop loss
        """
        return [
            p for p in self.positions.open_positions.values()
            if self.check_stop_loss(p, current_price)
        ]

    def activate_kill_switch(self, reason: str) -> None:
        """Activate emergency stop.

        Kill switch closes all positions and prevents any further trading
        until manually reset.

        Args:
            reason: Reason for activating kill switch
        """
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_time = datetime.now()

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Manually deactivate kill switch.

        Should only be called after reviewing the situation.
        """
        logger.warning("Kill switch deactivated manually")
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_time = None

    def _pause_trading(self, reason: str, duration_hours: int) -> None:
        """Pause trading for specified duration.

        Args:
            reason: Reason for pause
            duration_hours: Hours to pause
        """
        self.trading_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.now() + timedelta(hours=duration_hours)

        logger.warning(f"Trading paused: {reason} (until {self.pause_until})")

    def _resume_trading(self) -> None:
        """Resume trading after pause expires."""
        logger.info(f"Trading resumed after pause: {self.pause_reason}")
        self.trading_paused = False
        self.pause_reason = None
        self.pause_until = None

    def record_trade(self) -> None:
        """Record that a trade was executed (for interval tracking)."""
        self._last_trade_time = datetime.now()

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive risk status.

        Returns:
            Dictionary with risk metrics and status
        """
        return {
            "risk_level": self._assess_risk_level().value,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "trading_paused": self.trading_paused,
            "pause_reason": self.pause_reason,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
            "current_drawdown_pct": self.pnl.current_drawdown_pct,
            "max_drawdown_limit_pct": self.config.max_drawdown_pct * 100,
            "daily_pnl_pct": self.pnl.daily_pnl_pct,
            "daily_loss_limit_pct": self.config.daily_loss_limit_pct * 100,
            "weekly_pnl_pct": self.pnl.weekly_pnl_pct,
            "weekly_loss_limit_pct": self.config.weekly_loss_limit_pct * 100,
            "total_exposure": float(self.positions.get_total_exposure()),
            "max_exposure": self.config.max_total_exposure_usd,
            "exposure_pct": (
                float(self.positions.get_total_exposure())
                / self.config.initial_capital
                * 100
            ),
            "open_positions": len(self.positions.open_positions),
            "last_trade_time": (
                self._last_trade_time.isoformat() if self._last_trade_time else None
            ),
        }

    def get_risk_report(self) -> str:
        """Generate human-readable risk report.

        Returns:
            Formatted risk report string
        """
        status = self.get_status()
        risk_level = status["risk_level"].upper()

        lines = [
            f"=== RISK REPORT ===",
            f"Risk Level: {risk_level}",
            "",
            f"Kill Switch: {'ACTIVE - ' + status['kill_switch_reason'] if status['kill_switch_active'] else 'Inactive'}",
            f"Trading: {'PAUSED - ' + status['pause_reason'] if status['trading_paused'] else 'Active'}",
            "",
            f"Drawdown: {status['current_drawdown_pct']:.2f}% (limit: {status['max_drawdown_limit_pct']:.1f}%)",
            f"Daily P&L: {status['daily_pnl_pct']:.2f}% (limit: -{status['daily_loss_limit_pct']:.1f}%)",
            f"Weekly P&L: {status['weekly_pnl_pct']:.2f}% (limit: -{status['weekly_loss_limit_pct']:.1f}%)",
            "",
            f"Exposure: ${status['total_exposure']:.2f} ({status['exposure_pct']:.1f}%)",
            f"Max Exposure: ${status['max_exposure']:.2f}",
            f"Open Positions: {status['open_positions']}",
        ]

        return "\n".join(lines)
