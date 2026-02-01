"""Order manager for handling order lifecycle and execution."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.trading.config import TradingConfig
from src.trading.exchange.base import BaseExchange
from src.trading.orders.order import (
    Order,
    OrderSide,
    OrderState,
    OrderType,
)

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order placement, tracking, and lifecycle.

    Handles:
    - Order placement with retry logic
    - Order status tracking
    - Position sizing calculations
    - Fill tracking and aggregation

    Attributes:
        exchange: Exchange implementation to use
        config: Trading configuration
        pending_orders: Orders awaiting confirmation
        active_orders: Orders that are open on the exchange
        completed_orders: Historical completed orders
    """

    def __init__(
        self,
        exchange: BaseExchange,
        config: TradingConfig,
    ):
        """Initialize order manager.

        Args:
            exchange: Exchange implementation
            config: Trading configuration
        """
        self.exchange = exchange
        self.config = config
        self.pending_orders: dict[str, Order] = {}
        self.active_orders: dict[str, Order] = {}
        self.completed_orders: list[Order] = []
        self._max_retries = 3
        self._retry_delay_seconds = 1.0

    async def buy(
        self,
        qty: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        """Place a buy order.

        Args:
            qty: Quantity to buy (in base asset)
            order_type: MARKET or LIMIT
            limit_price: Required for LIMIT orders
            metadata: Optional metadata to attach to order

        Returns:
            Executed or placed order

        Raises:
            ValueError: If limit order missing price
        """
        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValueError("Limit price required for limit orders")

        return await self._place_order(
            side=OrderSide.BUY,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            metadata=metadata,
        )

    async def sell(
        self,
        qty: Decimal,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Decimal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        """Place a sell order.

        Args:
            qty: Quantity to sell (in base asset)
            order_type: MARKET or LIMIT
            limit_price: Required for LIMIT orders
            metadata: Optional metadata to attach to order

        Returns:
            Executed or placed order

        Raises:
            ValueError: If limit order missing price
        """
        if order_type == OrderType.LIMIT and limit_price is None:
            raise ValueError("Limit price required for limit orders")

        return await self._place_order(
            side=OrderSide.SELL,
            qty=qty,
            order_type=order_type,
            limit_price=limit_price,
            metadata=metadata,
        )

    async def _place_order(
        self,
        side: OrderSide,
        qty: Decimal,
        order_type: OrderType,
        limit_price: Decimal | None,
        metadata: dict[str, Any] | None,
    ) -> Order:
        """Internal order placement with retry logic.

        Args:
            side: BUY or SELL
            qty: Quantity
            order_type: MARKET or LIMIT
            limit_price: Limit price if applicable
            metadata: Order metadata

        Returns:
            Placed or executed order
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                if order_type == OrderType.MARKET:
                    order = await self.exchange.place_market_order(
                        symbol=self.config.symbol,
                        side=side.value,
                        qty=qty,
                    )
                else:
                    order = await self.exchange.place_limit_order(
                        symbol=self.config.symbol,
                        side=side.value,
                        price=limit_price,
                        qty=qty,
                    )

                if metadata:
                    order.metadata.update(metadata)

                # Track order
                self._track_order(order)

                logger.info(
                    f"Order placed: {order.order_id} - {side.value} {qty} "
                    f"{self.config.symbol} ({order.state.value})"
                )

                return order

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Order attempt {attempt + 1}/{self._max_retries} failed: {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay_seconds * (attempt + 1))

        # All retries failed
        order = Order(
            symbol=self.config.symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=limit_price,
            state=OrderState.FAILED,
            error_message=f"Failed after {self._max_retries} attempts: {last_error}",
        )
        if metadata:
            order.metadata.update(metadata)

        self.completed_orders.append(order)
        logger.error(f"Order failed permanently: {order.error_message}")
        return order

    def _track_order(self, order: Order) -> None:
        """Track order in appropriate collection.

        Args:
            order: Order to track
        """
        if order.is_complete:
            self.completed_orders.append(order)
            # Remove from active if present
            self.pending_orders.pop(order.order_id, None)
            self.active_orders.pop(order.order_id, None)
        elif order.is_active:
            if order.state == OrderState.PENDING:
                self.pending_orders[order.order_id] = order
            else:
                self.active_orders[order.order_id] = order
                self.pending_orders.pop(order.order_id, None)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False otherwise
        """
        order = self.active_orders.get(order_id) or self.pending_orders.get(order_id)
        if not order:
            logger.warning(f"Order not found for cancellation: {order_id}")
            return False

        try:
            success = await self.exchange.cancel_order(order_id, self.config.symbol)
            if success:
                order.state = OrderState.CANCELLED
                order.updated_at = datetime.now()
                self._track_order(order)
                logger.info(f"Order cancelled: {order_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all active orders.

        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        all_active = list(self.active_orders.keys()) + list(self.pending_orders.keys())

        for order_id in all_active:
            if await self.cancel_order(order_id):
                cancelled += 1

        logger.info(f"Cancelled {cancelled} orders")
        return cancelled

    async def refresh_order_status(self, order_id: str) -> Order | None:
        """Refresh order status from exchange.

        Args:
            order_id: Order ID to refresh

        Returns:
            Updated order or None if not found
        """
        try:
            order = await self.exchange.get_order(order_id, self.config.symbol)
            if order:
                self._track_order(order)
            return order
        except Exception as e:
            logger.error(f"Failed to refresh order {order_id}: {e}")
            return None

    async def refresh_all_active(self) -> None:
        """Refresh status of all active orders."""
        for order_id in list(self.active_orders.keys()):
            await self.refresh_order_status(order_id)

    def calculate_position_size(
        self,
        current_price: Decimal,
        capital_fraction: float | None = None,
    ) -> Decimal:
        """Calculate position size based on configuration.

        Args:
            current_price: Current asset price
            capital_fraction: Override for position size fraction (default: max_position_pct)

        Returns:
            Quantity to trade
        """
        fraction = capital_fraction or self.config.max_position_pct
        position_value = Decimal(str(self.config.initial_capital * fraction))
        qty = position_value / current_price

        # Round to reasonable precision (8 decimal places for BTC)
        qty = qty.quantize(Decimal("0.00000001"))

        return qty

    def get_order_history(
        self,
        side: OrderSide | None = None,
        state: OrderState | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get filtered order history.

        Args:
            side: Filter by side
            state: Filter by state
            limit: Maximum orders to return

        Returns:
            List of matching orders (most recent first)
        """
        orders = list(self.completed_orders)

        if side:
            orders = [o for o in orders if o.side == side]
        if state:
            orders = [o for o in orders if o.state == state]

        # Sort by created_at descending
        orders.sort(key=lambda o: o.created_at, reverse=True)

        return orders[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get order manager statistics.

        Returns:
            Dictionary of statistics
        """
        completed = self.completed_orders
        filled = [o for o in completed if o.state == OrderState.FILLED]
        failed = [o for o in completed if o.state in (OrderState.FAILED, OrderState.REJECTED)]

        total_bought = sum(
            o.filled_qty for o in filled if o.side == OrderSide.BUY
        )
        total_sold = sum(
            o.filled_qty for o in filled if o.side == OrderSide.SELL
        )
        total_fees = sum(o.total_commission for o in filled)

        return {
            "pending_orders": len(self.pending_orders),
            "active_orders": len(self.active_orders),
            "completed_orders": len(completed),
            "filled_orders": len(filled),
            "failed_orders": len(failed),
            "total_bought": float(total_bought),
            "total_sold": float(total_sold),
            "total_fees": float(total_fees),
        }
