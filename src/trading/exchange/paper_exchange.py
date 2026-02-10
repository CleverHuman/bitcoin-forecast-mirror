"""Paper trading exchange implementation for testing and simulation."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Callable
import uuid

from src.trading.config import TradingConfig
from src.trading.exchange.base import BaseExchange
from src.trading.orders.order import (
    Order,
    OrderSide,
    OrderState,
    OrderType,
    Fill,
)

logger = logging.getLogger(__name__)


class PaperExchange(BaseExchange):
    """Simulated exchange for paper trading.

    This implementation maintains virtual balances and simulates order
    execution with configurable fees and slippage. Useful for testing
    strategies without risking real capital.

    Attributes:
        config: Trading configuration
        balances: Virtual asset balances
        orders: Order history
        price_callbacks: Registered price update callbacks
        current_prices: Current simulated prices
    """

    def __init__(
        self,
        config: TradingConfig,
        initial_prices: dict[str, Decimal] | None = None,
    ):
        """Initialize paper exchange.

        Args:
            config: Trading configuration
            initial_prices: Optional initial prices for symbols
        """
        self.config = config
        self.balances: dict[str, Decimal] = {
            config.quote_asset: Decimal(str(config.initial_capital)),
            config.base_asset: Decimal("0"),
        }
        self.orders: dict[str, Order] = {}
        self.price_callbacks: dict[str, list[Callable[[str, Decimal], None]]] = {}
        self.current_prices: dict[str, Decimal] = initial_prices or {}
        self._running = False

    async def get_price(self, symbol: str) -> Decimal:
        """Get current simulated price.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price

        Raises:
            ValueError: If price not set for symbol
        """
        if symbol not in self.current_prices:
            raise ValueError(f"No price set for {symbol}")
        return self.current_prices[symbol]

    def set_price(self, symbol: str, price: Decimal) -> None:
        """Set current price for a symbol (for simulation).

        Args:
            symbol: Trading pair symbol
            price: New price
        """
        self.current_prices[symbol] = price
        # Notify callbacks
        if symbol in self.price_callbacks:
            for callback in self.price_callbacks[symbol]:
                try:
                    callback(symbol, price)
                except Exception as e:
                    logger.error(f"Error in price callback: {e}")

    async def get_balance(self, asset: str) -> Decimal:
        """Get available balance for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Available balance (returns 0 if asset not tracked)
        """
        return self.balances.get(asset, Decimal("0"))

    async def get_total_balance(self, asset: str) -> Decimal:
        """Get total balance for an asset.

        For paper trading, available and total are the same.

        Args:
            asset: Asset symbol

        Returns:
            Total balance
        """
        return await self.get_balance(asset)

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
    ) -> Order:
        """Place and immediately execute a simulated market order.

        Applies slippage and fees based on configuration.

        Args:
            symbol: Trading pair symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade

        Returns:
            Executed order with fill details
        """
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.MARKET,
            qty=qty,
            state=OrderState.PENDING,
        )

        try:
            # Get execution price with slippage
            base_price = await self.get_price(symbol)
            slippage_mult = Decimal("1") + Decimal(str(self.config.slippage_pct / 100))
            if order.side == OrderSide.BUY:
                exec_price = base_price * slippage_mult
            else:
                exec_price = base_price / slippage_mult

            # Calculate commission
            trade_value = qty * exec_price
            commission = trade_value * Decimal(str(self.config.fee_pct / 100))

            # Check sufficient balance
            if order.side == OrderSide.BUY:
                required = trade_value + commission
                available = await self.get_balance(self.config.quote_asset)
                if available < required:
                    order.state = OrderState.REJECTED
                    order.error_message = (
                        f"Insufficient {self.config.quote_asset}: "
                        f"need {required}, have {available}"
                    )
                    self.orders[order.order_id] = order
                    return order

                # Execute: debit quote asset, credit base asset
                self.balances[self.config.quote_asset] -= required
                self.balances[self.config.base_asset] += qty
            else:
                available = await self.get_balance(self.config.base_asset)
                if available < qty:
                    order.state = OrderState.REJECTED
                    order.error_message = (
                        f"Insufficient {self.config.base_asset}: "
                        f"need {qty}, have {available}"
                    )
                    self.orders[order.order_id] = order
                    return order

                # Execute: debit base asset, credit quote asset (minus fee)
                self.balances[self.config.base_asset] -= qty
                self.balances[self.config.quote_asset] += trade_value - commission

            # Create fill
            fill = Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                price=exec_price,
                qty=qty,
                commission=commission,
                commission_asset=self.config.quote_asset,
                timestamp=datetime.now(),
            )

            order.add_fill(fill)
            order.state = OrderState.FILLED
            order.exchange_order_id = order.order_id  # Same for paper

            logger.info(
                f"Paper order executed: {side} {qty} {symbol} @ {exec_price} "
                f"(commission: {commission})"
            )

        except Exception as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            logger.error(f"Paper order failed: {e}")

        self.orders[order.order_id] = order
        return order

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        qty: Decimal,
    ) -> Order:
        """Place a limit order (simulated).

        For paper trading, limit orders are stored but need to be
        manually triggered when price moves. This implementation
        immediately executes if the price is favorable.

        Args:
            symbol: Trading pair symbol
            side: "BUY" or "SELL"
            price: Limit price
            qty: Quantity to trade

        Returns:
            Order with OPEN or FILLED state
        """
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.LIMIT,
            qty=qty,
            price=price,
            state=OrderState.OPEN,
        )

        try:
            current_price = await self.get_price(symbol)

            # Check if order should fill immediately
            should_fill = (
                (order.side == OrderSide.BUY and current_price <= price)
                or (order.side == OrderSide.SELL and current_price >= price)
            )

            if should_fill:
                # Execute at limit price
                trade_value = qty * price
                commission = trade_value * Decimal(str(self.config.fee_pct / 100))

                if order.side == OrderSide.BUY:
                    required = trade_value + commission
                    available = await self.get_balance(self.config.quote_asset)
                    if available < required:
                        order.state = OrderState.REJECTED
                        order.error_message = f"Insufficient {self.config.quote_asset}"
                        self.orders[order.order_id] = order
                        return order

                    self.balances[self.config.quote_asset] -= required
                    self.balances[self.config.base_asset] += qty
                else:
                    available = await self.get_balance(self.config.base_asset)
                    if available < qty:
                        order.state = OrderState.REJECTED
                        order.error_message = f"Insufficient {self.config.base_asset}"
                        self.orders[order.order_id] = order
                        return order

                    self.balances[self.config.base_asset] -= qty
                    self.balances[self.config.quote_asset] += trade_value - commission

                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    price=price,
                    qty=qty,
                    commission=commission,
                    commission_asset=self.config.quote_asset,
                    timestamp=datetime.now(),
                )
                order.add_fill(fill)

                logger.info(f"Paper limit order filled: {side} {qty} {symbol} @ {price}")
            else:
                logger.info(
                    f"Paper limit order placed: {side} {qty} {symbol} @ {price} "
                    f"(current: {current_price})"
                )

        except Exception as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            logger.error(f"Paper limit order failed: {e}")

        order.exchange_order_id = order.order_id
        self.orders[order.order_id] = order
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol (unused for paper)

        Returns:
            True if cancelled, False otherwise
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if not order.is_active:
            return False

        order.state = OrderState.CANCELLED
        order.updated_at = datetime.now()
        logger.info(f"Paper order cancelled: {order_id}")
        return True

    async def get_order_status(self, order_id: str, symbol: str) -> OrderState:
        """Get current status of an order.

        Args:
            order_id: Order ID to check
            symbol: Trading pair symbol (unused for paper)

        Returns:
            Current OrderState

        Raises:
            ValueError: If order not found
        """
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")
        return self.orders[order_id].state

    async def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get full order details.

        Args:
            order_id: Order ID to fetch
            symbol: Trading pair symbol (unused for paper)

        Returns:
            Order object or None if not found
        """
        return self.orders.get(order_id)

    def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, Decimal], None],
    ) -> None:
        """Subscribe to price updates.

        For paper trading, callbacks are invoked when set_price is called.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with (symbol, price)
        """
        if symbol not in self.price_callbacks:
            self.price_callbacks[symbol] = []
        self.price_callbacks[symbol].append(callback)
        logger.debug(f"Subscribed to {symbol} price updates")

    def unsubscribe_price(self, symbol: str) -> None:
        """Unsubscribe from price updates.

        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.price_callbacks:
            self.price_callbacks[symbol].clear()
        logger.debug(f"Unsubscribed from {symbol} price updates")

    async def close(self) -> None:
        """Clean up resources."""
        self._running = False
        self.price_callbacks.clear()
        logger.info("Paper exchange closed")

    def get_portfolio_value(self, prices: dict[str, Decimal] | None = None) -> Decimal:
        """Calculate total portfolio value in quote asset.

        Args:
            prices: Optional price overrides (uses current_prices if not provided)

        Returns:
            Total portfolio value in quote asset
        """
        prices = prices or self.current_prices
        total = self.balances.get(self.config.quote_asset, Decimal("0"))

        base_balance = self.balances.get(self.config.base_asset, Decimal("0"))
        if base_balance > 0 and self.config.symbol in prices:
            total += base_balance * prices[self.config.symbol]

        return total

    def get_position_summary(self) -> dict:
        """Get summary of current positions.

        Returns:
            Dictionary with balance and position information
        """
        return {
            "quote_balance": float(self.balances.get(self.config.quote_asset, 0)),
            "base_balance": float(self.balances.get(self.config.base_asset, 0)),
            "current_price": float(self.current_prices.get(self.config.symbol, 0)),
            "portfolio_value": float(self.get_portfolio_value()),
            "open_orders": len([o for o in self.orders.values() if o.is_active]),
            "total_orders": len(self.orders),
        }
