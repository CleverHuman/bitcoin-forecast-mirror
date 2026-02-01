"""Telegram alert integration for trading notifications."""

import asyncio
import logging
from datetime import datetime
from typing import Any

from src.trading.config import TradingConfig
from src.trading.orders.order import Order, OrderState
from src.trading.position.position_tracker import Position

logger = logging.getLogger(__name__)


class TelegramAlerter:
    """Sends trading alerts via Telegram.

    Sends notifications for:
    - Trade executions
    - Stop loss triggers
    - Risk limit breaches
    - Daily/weekly summaries
    - System errors

    Attributes:
        config: Trading configuration
        bot: Telegram bot instance (if available)
        enabled: Whether alerting is enabled
    """

    def __init__(self, config: TradingConfig):
        """Initialize Telegram alerter.

        Args:
            config: Trading configuration with Telegram credentials
        """
        self.config = config
        self.bot = None
        self.enabled = False
        self._message_queue: list[str] = []
        self._last_error_time: datetime | None = None

        self._initialize_bot()

    def _initialize_bot(self) -> None:
        """Initialize Telegram bot if credentials are available."""
        if not self.config.telegram_token or not self.config.telegram_chat_id:
            logger.warning("Telegram credentials not configured - alerts disabled")
            return

        try:
            from telegram import Bot
            self.bot = Bot(token=self.config.telegram_token)
            self.enabled = True
            logger.info("Telegram alerter initialized")
        except ImportError:
            logger.warning("python-telegram-bot not installed - alerts disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message via Telegram.

        Args:
            message: Message text (supports HTML formatting)
            parse_mode: Message parse mode (HTML or Markdown)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled or not self.bot:
            logger.debug(f"Alert (not sent): {message[:100]}...")
            return False

        try:
            await self.bot.send_message(
                chat_id=self.config.telegram_chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            self._message_queue.append(message)
            return False

    async def send_trade_alert(
        self,
        order: Order,
        signal_score: float | None = None,
        reason: str | None = None,
    ) -> bool:
        """Send alert for a trade execution.

        Args:
            order: Executed order
            signal_score: Signal score that triggered the trade
            reason: Reason for the trade

        Returns:
            True if sent
        """
        if order.state != OrderState.FILLED:
            return False

        emoji = "\U0001f7e2" if order.side.value == "BUY" else "\U0001f534"  # Green/Red circle

        message = f"""
{emoji} <b>TRADE EXECUTED</b>

<b>Symbol:</b> {order.symbol}
<b>Side:</b> {order.side.value}
<b>Quantity:</b> {order.filled_qty}
<b>Price:</b> ${order.avg_fill_price:,.2f}
<b>Value:</b> ${float(order.total_value):,.2f}
<b>Fees:</b> ${float(order.total_commission):,.4f}

<b>Signal Score:</b> {signal_score:.2f if signal_score else 'N/A'}
<b>Reason:</b> {reason or 'N/A'}

<i>Order ID: {order.order_id[:8]}...</i>
"""
        return await self.send_message(message.strip())

    async def send_stop_loss_alert(
        self,
        position: Position,
        exit_price: float,
        pnl: float,
    ) -> bool:
        """Send alert when stop loss is triggered.

        Args:
            position: Position that was stopped out
            exit_price: Exit price
            pnl: Realized P&L

        Returns:
            True if sent
        """
        pnl_emoji = "\U0001f4b8" if pnl < 0 else "\U0001f4b0"  # Money with wings / Money bag

        message = f"""
\U0001f6a8 <b>STOP LOSS TRIGGERED</b> {pnl_emoji}

<b>Position:</b> {position.position_id}
<b>Symbol:</b> {position.symbol}
<b>Entry Price:</b> ${float(position.entry_price):,.2f}
<b>Exit Price:</b> ${exit_price:,.2f}
<b>Quantity:</b> {position.qty}

<b>P&L:</b> ${pnl:,.2f} ({pnl / float(position.entry_value) * 100:.2f}%)

<i>Stop was set at ${float(position.stop_loss_price):,.2f}</i>
"""
        return await self.send_message(message.strip())

    async def send_risk_alert(
        self,
        alert_type: str,
        details: dict[str, Any],
    ) -> bool:
        """Send alert for risk limit breach.

        Args:
            alert_type: Type of risk alert (e.g., "kill_switch", "daily_loss")
            details: Details about the breach

        Returns:
            True if sent
        """
        emoji = "\U0001f6a8" if "kill" in alert_type.lower() else "\U000026a0"  # Siren / Warning

        message = f"""
{emoji} <b>RISK ALERT: {alert_type.upper()}</b>

"""
        for key, value in details.items():
            if isinstance(value, float):
                message += f"<b>{key}:</b> {value:.2f}\n"
            else:
                message += f"<b>{key}:</b> {value}\n"

        return await self.send_message(message.strip())

    async def send_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades_today: int,
        open_positions: int,
        drawdown_pct: float,
    ) -> bool:
        """Send daily trading summary.

        Args:
            equity: Current equity
            daily_pnl: Day's P&L
            daily_pnl_pct: Day's P&L as percentage
            trades_today: Number of trades today
            open_positions: Number of open positions
            drawdown_pct: Current drawdown percentage

        Returns:
            True if sent
        """
        pnl_emoji = "\U0001f4c8" if daily_pnl >= 0 else "\U0001f4c9"  # Chart up/down

        message = f"""
{pnl_emoji} <b>DAILY SUMMARY</b>

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}

<b>Equity:</b> ${equity:,.2f}
<b>Day P&L:</b> ${daily_pnl:,.2f} ({daily_pnl_pct:+.2f}%)
<b>Drawdown:</b> {drawdown_pct:.2f}%

<b>Trades Today:</b> {trades_today}
<b>Open Positions:</b> {open_positions}

<i>Paper Trading: {'Yes' if self.config.paper_trading else 'No'}</i>
"""
        return await self.send_message(message.strip())

    async def send_error_alert(
        self,
        error: Exception,
        context: str | None = None,
    ) -> bool:
        """Send alert for system error.

        Rate-limits error alerts to prevent spam.

        Args:
            error: The exception that occurred
            context: Context about what was happening

        Returns:
            True if sent
        """
        # Rate limit: only send one error alert per minute
        now = datetime.now()
        if self._last_error_time:
            seconds_since_last = (now - self._last_error_time).total_seconds()
            if seconds_since_last < 60:
                logger.debug(f"Error alert rate-limited: {error}")
                return False

        self._last_error_time = now

        message = f"""
\U0001f525 <b>SYSTEM ERROR</b>

<b>Time:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}
<b>Context:</b> {context or 'Unknown'}
<b>Error:</b> {type(error).__name__}
<b>Message:</b> {str(error)[:200]}

<i>Check logs for details</i>
"""
        return await self.send_message(message.strip())

    async def send_startup_message(self) -> bool:
        """Send message when trading system starts.

        Returns:
            True if sent
        """
        mode = "PAPER" if self.config.paper_trading else "LIVE"
        emoji = "\U0001f4dd" if self.config.paper_trading else "\U0001f4b5"  # Memo / Dollar

        message = f"""
{emoji} <b>TRADING SYSTEM STARTED</b>

<b>Mode:</b> {mode} TRADING
<b>Symbol:</b> {self.config.symbol}
<b>Initial Capital:</b> ${self.config.initial_capital:,.2f}

<b>Risk Limits:</b>
- Max Position: {self.config.max_position_pct * 100:.0f}%
- Max Exposure: {self.config.max_total_exposure_pct * 100:.0f}%
- Stop Loss: {self.config.stop_loss_pct * 100:.0f}%
- Max Drawdown: {self.config.max_drawdown_pct * 100:.0f}%

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return await self.send_message(message.strip())

    async def send_shutdown_message(self, reason: str = "normal") -> bool:
        """Send message when trading system stops.

        Args:
            reason: Reason for shutdown

        Returns:
            True if sent
        """
        message = f"""
\U0001f6d1 <b>TRADING SYSTEM STOPPED</b>

<b>Reason:</b> {reason}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return await self.send_message(message.strip())

    async def flush_queue(self) -> int:
        """Attempt to send any queued messages.

        Returns:
            Number of messages sent
        """
        if not self._message_queue:
            return 0

        sent = 0
        remaining = []

        for message in self._message_queue:
            if await self.send_message(message):
                sent += 1
            else:
                remaining.append(message)
            await asyncio.sleep(0.5)  # Rate limit

        self._message_queue = remaining
        return sent

    def is_enabled(self) -> bool:
        """Check if alerter is enabled and configured.

        Returns:
            True if alerts will be sent
        """
        return self.enabled and self.bot is not None

    def get_status(self) -> dict[str, Any]:
        """Get alerter status.

        Returns:
            Status dictionary
        """
        return {
            "enabled": self.enabled,
            "configured": bool(self.config.telegram_token and self.config.telegram_chat_id),
            "queued_messages": len(self._message_queue),
            "last_error_time": (
                self._last_error_time.isoformat() if self._last_error_time else None
            ),
        }
