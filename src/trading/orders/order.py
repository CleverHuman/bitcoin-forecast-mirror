"""Order and Fill dataclasses for trading operations."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
import uuid


class OrderSide(Enum):
    """Order side (buy or sell)."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderState(Enum):
    """Order lifecycle state."""

    PENDING = "PENDING"  # Created but not sent
    OPEN = "OPEN"  # Sent to exchange, not filled
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Some quantity filled
    FILLED = "FILLED"  # Fully executed
    CANCELLED = "CANCELLED"  # Cancelled by user
    REJECTED = "REJECTED"  # Rejected by exchange
    EXPIRED = "EXPIRED"  # Time limit exceeded
    FAILED = "FAILED"  # Execution error


@dataclass
class Fill:
    """Represents a partial or full order fill.

    Attributes:
        fill_id: Unique fill identifier
        order_id: Parent order ID
        price: Execution price
        qty: Filled quantity
        commission: Fee charged for this fill
        commission_asset: Asset the fee was charged in
        timestamp: When the fill occurred
    """

    fill_id: str
    order_id: str
    price: Decimal
    qty: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: datetime

    @property
    def value(self) -> Decimal:
        """Total value of this fill (price * qty)."""
        return self.price * self.qty


@dataclass
class Order:
    """Represents a trading order.

    Attributes:
        order_id: Unique order identifier
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        side: Buy or sell
        order_type: Market or limit
        qty: Requested quantity
        price: Limit price (None for market orders)
        state: Current order state
        created_at: When the order was created
        updated_at: When the order was last updated
        filled_qty: Quantity that has been filled
        avg_fill_price: Volume-weighted average fill price
        fills: List of individual fills
        client_order_id: Client-specified order ID
        exchange_order_id: Exchange-assigned order ID
        error_message: Error message if order failed
        metadata: Additional order metadata
    """

    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    qty: Decimal = Decimal("0")
    price: Decimal | None = None
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    fills: list[Fill] = field(default_factory=list)
    client_order_id: str | None = None
    exchange_order_id: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if order has reached a terminal state."""
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        )

    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be filled or cancelled)."""
        return self.state in (
            OrderState.PENDING,
            OrderState.OPEN,
            OrderState.PARTIALLY_FILLED,
        )

    @property
    def unfilled_qty(self) -> Decimal:
        """Remaining quantity to be filled."""
        return self.qty - self.filled_qty

    @property
    def fill_pct(self) -> float:
        """Percentage of order that has been filled."""
        if self.qty == 0:
            return 0.0
        return float(self.filled_qty / self.qty) * 100

    @property
    def total_value(self) -> Decimal:
        """Total value of filled portion."""
        if self.avg_fill_price is None:
            return Decimal("0")
        return self.filled_qty * self.avg_fill_price

    @property
    def total_commission(self) -> Decimal:
        """Total commission paid across all fills."""
        return sum((f.commission for f in self.fills), Decimal("0"))

    def add_fill(self, fill: Fill) -> None:
        """Add a fill to this order and update state.

        Args:
            fill: The fill to add
        """
        self.fills.append(fill)
        self.filled_qty += fill.qty
        self.updated_at = datetime.now()

        # Update average fill price
        total_value = sum(f.price * f.qty for f in self.fills)
        total_qty = sum(f.qty for f in self.fills)
        if total_qty > 0:
            self.avg_fill_price = total_value / total_qty

        # Update state
        if self.filled_qty >= self.qty:
            self.state = OrderState.FILLED
        elif self.filled_qty > 0:
            self.state = OrderState.PARTIALLY_FILLED

    def to_dict(self) -> dict[str, Any]:
        """Convert order to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": str(self.qty),
            "price": str(self.price) if self.price else None,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_qty": str(self.filled_qty),
            "avg_fill_price": str(self.avg_fill_price) if self.avg_fill_price else None,
            "fill_count": len(self.fills),
            "total_commission": str(self.total_commission),
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "error_message": self.error_message,
        }
