"""Position tracking for managing open positions."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.trading.config import TradingConfig
from src.trading.orders.order import Order, OrderSide, OrderState

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position.

    Attributes:
        position_id: Unique position identifier
        symbol: Trading pair symbol
        side: LONG or SHORT (BUY side = LONG for spot)
        qty: Position quantity
        entry_price: Average entry price
        entry_time: When position was opened
        entry_orders: Orders that opened this position
        current_price: Last known price
        stop_loss_price: Stop loss trigger price
        take_profit_price: Take profit trigger price
        metadata: Additional position data
    """

    position_id: str
    symbol: str
    side: str  # "LONG" or "SHORT"
    qty: Decimal
    entry_price: Decimal
    entry_time: datetime
    entry_orders: list[str] = field(default_factory=list)
    current_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entry_value(self) -> Decimal:
        """Total value at entry."""
        return self.qty * self.entry_price

    @property
    def current_value(self) -> Decimal | None:
        """Current position value."""
        if self.current_price is None:
            return None
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> Decimal | None:
        """Unrealized profit/loss."""
        if self.current_price is None:
            return None
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.current_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float | None:
        """Unrealized P&L as percentage."""
        if self.current_price is None or self.entry_price == 0:
            return None
        pnl = self.unrealized_pnl
        if pnl is None:
            return None
        return float(pnl / self.entry_value) * 100

    def update_price(self, price: Decimal) -> None:
        """Update current price for this position."""
        self.current_price = price

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": str(self.qty),
            "entry_price": str(self.entry_price),
            "entry_time": self.entry_time.isoformat(),
            "entry_value": str(self.entry_value),
            "current_price": str(self.current_price) if self.current_price else None,
            "current_value": str(self.current_value) if self.current_value else None,
            "unrealized_pnl": str(self.unrealized_pnl) if self.unrealized_pnl else None,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "stop_loss_price": str(self.stop_loss_price) if self.stop_loss_price else None,
            "take_profit_price": str(self.take_profit_price) if self.take_profit_price else None,
        }


class PositionTracker:
    """Tracks open and closed positions.

    Manages position lifecycle from entry to exit, calculating
    P&L and maintaining position history.

    Attributes:
        config: Trading configuration
        open_positions: Currently open positions
        closed_positions: Historical closed positions
    """

    def __init__(self, config: TradingConfig):
        """Initialize position tracker.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.open_positions: dict[str, Position] = {}
        self.closed_positions: list[dict[str, Any]] = []
        self._position_counter = 0

    def _generate_position_id(self) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"POS-{self._position_counter:06d}"

    def open_position(
        self,
        order: Order,
        stop_loss_pct: float | None = None,
    ) -> Position | None:
        """Open a new position from a filled order.

        Args:
            order: Filled buy order
            stop_loss_pct: Optional stop loss percentage (default from config)

        Returns:
            Created position or None if order not filled
        """
        if order.state != OrderState.FILLED:
            logger.warning(f"Cannot open position from unfilled order: {order.order_id}")
            return None

        if order.side != OrderSide.BUY:
            logger.warning("Can only open LONG positions from BUY orders (spot trading)")
            return None

        stop_loss_pct = stop_loss_pct or self.config.stop_loss_pct
        stop_loss_price = order.avg_fill_price * Decimal(str(1 - stop_loss_pct))

        position = Position(
            position_id=self._generate_position_id(),
            symbol=order.symbol,
            side="LONG",
            qty=order.filled_qty,
            entry_price=order.avg_fill_price,
            entry_time=datetime.now(),
            entry_orders=[order.order_id],
            current_price=order.avg_fill_price,
            stop_loss_price=stop_loss_price,
            metadata={"signal": order.metadata.get("signal")},
        )

        self.open_positions[position.position_id] = position

        logger.info(
            f"Position opened: {position.position_id} - "
            f"{position.side} {position.qty} {position.symbol} @ {position.entry_price} "
            f"(SL: {stop_loss_price})"
        )

        return position

    def add_to_position(
        self,
        position_id: str,
        order: Order,
    ) -> Position | None:
        """Add to an existing position.

        Args:
            position_id: Position to add to
            order: Filled buy order

        Returns:
            Updated position or None if not found/invalid
        """
        if position_id not in self.open_positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        if order.state != OrderState.FILLED or order.side != OrderSide.BUY:
            logger.warning("Can only add to position with filled BUY order")
            return None

        position = self.open_positions[position_id]

        # Calculate new average entry price
        old_value = position.qty * position.entry_price
        new_value = order.filled_qty * order.avg_fill_price
        new_qty = position.qty + order.filled_qty
        new_avg_price = (old_value + new_value) / new_qty

        position.qty = new_qty
        position.entry_price = new_avg_price
        position.entry_orders.append(order.order_id)

        # Update stop loss based on new entry
        position.stop_loss_price = new_avg_price * Decimal(str(1 - self.config.stop_loss_pct))

        logger.info(
            f"Position increased: {position_id} - "
            f"now {position.qty} @ avg {position.entry_price}"
        )

        return position

    def close_position(
        self,
        position_id: str,
        exit_order: Order,
        reason: str = "manual",
    ) -> dict[str, Any] | None:
        """Close an open position.

        Args:
            position_id: Position to close
            exit_order: Filled sell order
            reason: Reason for closing (e.g., "stop_loss", "take_profit", "signal")

        Returns:
            Closed position summary or None if not found
        """
        if position_id not in self.open_positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        if exit_order.state != OrderState.FILLED or exit_order.side != OrderSide.SELL:
            logger.warning("Can only close position with filled SELL order")
            return None

        position = self.open_positions[position_id]

        # Calculate P&L
        exit_price = exit_order.avg_fill_price
        pnl = (exit_price - position.entry_price) * position.qty
        pnl_pct = float(pnl / position.entry_value) * 100

        # Calculate fees (from entry orders + exit order)
        total_fees = exit_order.total_commission

        closed_position = {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "side": position.side,
            "qty": float(position.qty),
            "entry_price": float(position.entry_price),
            "exit_price": float(exit_price),
            "entry_time": position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "pnl": float(pnl),
            "pnl_pct": pnl_pct,
            "fees": float(total_fees),
            "net_pnl": float(pnl - total_fees),
            "reason": reason,
            "entry_orders": position.entry_orders,
            "exit_order": exit_order.order_id,
            "metadata": position.metadata,
        }

        self.closed_positions.append(closed_position)
        del self.open_positions[position_id]

        logger.info(
            f"Position closed: {position_id} - "
            f"PnL: {pnl:.2f} ({pnl_pct:.2f}%) - Reason: {reason}"
        )

        return closed_position

    def partial_close(
        self,
        position_id: str,
        exit_order: Order,
        reason: str = "partial",
    ) -> dict[str, Any] | None:
        """Partially close a position.

        Args:
            position_id: Position to partially close
            exit_order: Filled sell order (qty < position qty)
            reason: Reason for partial close

        Returns:
            Partial close summary or None if invalid
        """
        if position_id not in self.open_positions:
            logger.warning(f"Position not found: {position_id}")
            return None

        position = self.open_positions[position_id]

        if exit_order.filled_qty >= position.qty:
            # Full close
            return self.close_position(position_id, exit_order, reason)

        # Calculate P&L for closed portion
        exit_price = exit_order.avg_fill_price
        closed_qty = exit_order.filled_qty
        pnl = (exit_price - position.entry_price) * closed_qty

        partial_result = {
            "position_id": position_id,
            "closed_qty": float(closed_qty),
            "remaining_qty": float(position.qty - closed_qty),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "reason": reason,
        }

        # Update position
        position.qty -= closed_qty

        logger.info(
            f"Position partially closed: {position_id} - "
            f"Closed {closed_qty}, remaining {position.qty}"
        )

        return partial_result

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update current prices for all positions.

        Args:
            prices: Dictionary of symbol -> price
        """
        for position in self.open_positions.values():
            if position.symbol in prices:
                position.update_price(prices[position.symbol])

    def get_total_exposure(self) -> Decimal:
        """Get total exposure in quote currency (USD).

        Returns:
            Sum of all position entry values
        """
        return sum(
            (p.entry_value for p in self.open_positions.values()),
            Decimal("0"),
        )

    def get_total_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L across all positions.

        Returns:
            Sum of unrealized P&L (0 if prices not updated)
        """
        total = Decimal("0")
        for position in self.open_positions.values():
            pnl = position.unrealized_pnl
            if pnl is not None:
                total += pnl
        return total

    def get_equity(self, current_price: Decimal, cash_balance: Decimal) -> Decimal:
        """Calculate total equity (cash + positions at current price).

        Args:
            current_price: Current price of the traded asset
            cash_balance: Current cash balance

        Returns:
            Total equity value
        """
        # Update position prices
        self.update_prices({self.config.symbol: current_price})

        position_value = sum(
            (p.current_value or Decimal("0") for p in self.open_positions.values()),
            Decimal("0"),
        )

        return cash_balance + position_value

    def get_positions_at_stop_loss(self, current_price: Decimal) -> list[Position]:
        """Get positions that have hit their stop loss.

        Args:
            current_price: Current market price

        Returns:
            List of positions at or below stop loss
        """
        triggered = []
        for position in self.open_positions.values():
            if position.stop_loss_price and current_price <= position.stop_loss_price:
                triggered.append(position)
        return triggered

    def get_summary(self) -> dict[str, Any]:
        """Get summary of position tracking state.

        Returns:
            Dictionary with position statistics
        """
        closed = self.closed_positions
        profitable = [p for p in closed if p["pnl"] > 0]
        losing = [p for p in closed if p["pnl"] < 0]

        total_pnl = sum(p["pnl"] for p in closed)
        total_fees = sum(p["fees"] for p in closed)

        return {
            "open_positions": len(self.open_positions),
            "total_exposure": float(self.get_total_exposure()),
            "unrealized_pnl": float(self.get_total_unrealized_pnl()),
            "closed_positions": len(closed),
            "profitable_trades": len(profitable),
            "losing_trades": len(losing),
            "win_rate": len(profitable) / len(closed) * 100 if closed else 0,
            "total_realized_pnl": total_pnl,
            "total_fees": total_fees,
            "net_realized_pnl": total_pnl - total_fees,
        }
