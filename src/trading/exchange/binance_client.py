"""Binance REST API client for live trading."""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable
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


class BinanceClient(BaseExchange):
    """Binance exchange client for live trading.

    Uses the python-binance library for REST API calls.
    Supports both spot trading operations.

    Attributes:
        config: Trading configuration
        client: Binance Client instance
        _price_callbacks: Registered price update callbacks
    """

    def __init__(self, config: TradingConfig):
        """Initialize Binance client.

        Args:
            config: Trading configuration with API credentials

        Raises:
            ImportError: If python-binance is not installed
            ValueError: If API credentials are missing
        """
        self.config = config

        if not config.binance_api_key or not config.binance_api_secret:
            raise ValueError("Binance API credentials required for live trading")

        try:
            from binance import AsyncClient
            self._AsyncClient = AsyncClient
        except ImportError:
            raise ImportError(
                "python-binance is required for live trading. "
                "Install with: pip install python-binance"
            )

        self._client = None
        self._price_callbacks: dict[str, list[Callable[[str, Decimal], None]]] = {}
        self._ws_manager = None

    async def _ensure_client(self) -> None:
        """Ensure client is initialized."""
        if self._client is None:
            self._client = await self._AsyncClient.create(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
            )

    async def get_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")

        Returns:
            Current price as Decimal
        """
        await self._ensure_client()

        ticker = await self._client.get_symbol_ticker(symbol=symbol)
        return Decimal(ticker["price"])

    async def get_balance(self, asset: str) -> Decimal:
        """Get available balance for an asset.

        Args:
            asset: Asset symbol (e.g., "BTC", "USDT")

        Returns:
            Available (free) balance as Decimal
        """
        await self._ensure_client()

        account = await self._client.get_account()
        for balance in account["balances"]:
            if balance["asset"] == asset:
                return Decimal(balance["free"])
        return Decimal("0")

    async def get_total_balance(self, asset: str) -> Decimal:
        """Get total balance for an asset (free + locked).

        Args:
            asset: Asset symbol

        Returns:
            Total balance as Decimal
        """
        await self._ensure_client()

        account = await self._client.get_account()
        for balance in account["balances"]:
            if balance["asset"] == asset:
                free = Decimal(balance["free"])
                locked = Decimal(balance["locked"])
                return free + locked
        return Decimal("0")

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
    ) -> Order:
        """Place a market order on Binance.

        Args:
            symbol: Trading pair symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade (in base asset)

        Returns:
            Order object with execution details
        """
        await self._ensure_client()

        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.MARKET,
            qty=qty,
            state=OrderState.PENDING,
        )

        try:
            # Place order
            result = await self._client.create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=str(qty),
            )

            order.exchange_order_id = str(result["orderId"])
            order.client_order_id = result.get("clientOrderId")

            # Process fills
            for fill_data in result.get("fills", []):
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    price=Decimal(fill_data["price"]),
                    qty=Decimal(fill_data["qty"]),
                    commission=Decimal(fill_data["commission"]),
                    commission_asset=fill_data["commissionAsset"],
                    timestamp=datetime.now(),
                )
                order.add_fill(fill)

            # Map Binance status to our OrderState
            order.state = self._map_order_status(result["status"])

            logger.info(
                f"Binance order executed: {side} {qty} {symbol} @ {order.avg_fill_price}"
            )

        except Exception as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            logger.error(f"Binance order failed: {e}")

        return order

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        qty: Decimal,
    ) -> Order:
        """Place a limit order on Binance.

        Args:
            symbol: Trading pair symbol
            side: "BUY" or "SELL"
            price: Limit price
            qty: Quantity to trade

        Returns:
            Order object with OPEN or FILLED state
        """
        await self._ensure_client()

        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.LIMIT,
            qty=qty,
            price=price,
            state=OrderState.PENDING,
        )

        try:
            result = await self._client.create_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                timeInForce="GTC",
                quantity=str(qty),
                price=str(price),
            )

            order.exchange_order_id = str(result["orderId"])
            order.client_order_id = result.get("clientOrderId")
            order.state = self._map_order_status(result["status"])

            # Process any immediate fills
            for fill_data in result.get("fills", []):
                fill = Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    price=Decimal(fill_data["price"]),
                    qty=Decimal(fill_data["qty"]),
                    commission=Decimal(fill_data["commission"]),
                    commission_asset=fill_data["commissionAsset"],
                    timestamp=datetime.now(),
                )
                order.add_fill(fill)

            logger.info(f"Binance limit order placed: {side} {qty} {symbol} @ {price}")

        except Exception as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            logger.error(f"Binance limit order failed: {e}")

        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading pair symbol

        Returns:
            True if cancelled successfully
        """
        await self._ensure_client()

        try:
            await self._client.cancel_order(
                symbol=symbol,
                orderId=int(order_id),
            )
            logger.info(f"Binance order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderState:
        """Get current status of an order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            Current OrderState
        """
        await self._ensure_client()

        try:
            result = await self._client.get_order(
                symbol=symbol,
                orderId=int(order_id),
            )
            return self._map_order_status(result["status"])
        except Exception as e:
            logger.error(f"Failed to get order status {order_id}: {e}")
            raise

    async def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get full order details from Binance.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol

        Returns:
            Order object or None if not found
        """
        await self._ensure_client()

        try:
            result = await self._client.get_order(
                symbol=symbol,
                orderId=int(order_id),
            )

            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=result["symbol"],
                side=OrderSide(result["side"]),
                order_type=OrderType(result["type"]),
                qty=Decimal(result["origQty"]),
                price=Decimal(result["price"]) if result["price"] != "0.00000000" else None,
                state=self._map_order_status(result["status"]),
                exchange_order_id=str(result["orderId"]),
                filled_qty=Decimal(result["executedQty"]),
            )

            # Get trades to build fills
            trades = await self._client.get_my_trades(symbol=symbol, orderId=int(order_id))
            for trade in trades:
                fill = Fill(
                    fill_id=str(trade["id"]),
                    order_id=order.order_id,
                    price=Decimal(trade["price"]),
                    qty=Decimal(trade["qty"]),
                    commission=Decimal(trade["commission"]),
                    commission_asset=trade["commissionAsset"],
                    timestamp=datetime.fromtimestamp(trade["time"] / 1000),
                )
                order.fills.append(fill)

            # Recalculate average price from fills
            if order.fills:
                total_value = sum(f.price * f.qty for f in order.fills)
                total_qty = sum(f.qty for f in order.fills)
                order.avg_fill_price = total_value / total_qty

            return order

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, Decimal], None],
    ) -> None:
        """Subscribe to real-time price updates via WebSocket.

        Note: For full WebSocket support, use BinanceWebSocket class.
        This method provides a simple callback registration.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with (symbol, price)
        """
        if symbol not in self._price_callbacks:
            self._price_callbacks[symbol] = []
        self._price_callbacks[symbol].append(callback)
        logger.info(f"Registered price callback for {symbol}")

    def unsubscribe_price(self, symbol: str) -> None:
        """Unsubscribe from price updates.

        Args:
            symbol: Trading pair symbol
        """
        if symbol in self._price_callbacks:
            self._price_callbacks[symbol].clear()

    def _map_order_status(self, binance_status: str) -> OrderState:
        """Map Binance order status to our OrderState.

        Args:
            binance_status: Binance order status string

        Returns:
            Corresponding OrderState
        """
        status_map = {
            "NEW": OrderState.OPEN,
            "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
            "FILLED": OrderState.FILLED,
            "CANCELED": OrderState.CANCELLED,
            "PENDING_CANCEL": OrderState.OPEN,
            "REJECTED": OrderState.REJECTED,
            "EXPIRED": OrderState.EXPIRED,
        }
        return status_map.get(binance_status, OrderState.PENDING)

    async def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """Get exchange trading rules and symbol info.

        Args:
            symbol: Optional symbol to filter

        Returns:
            Exchange info dictionary
        """
        await self._ensure_client()

        info = await self._client.get_exchange_info()

        if symbol:
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    return s
            return {}

        return info

    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Get recent trades for a symbol.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch (max 1000)

        Returns:
            List of trade dictionaries
        """
        await self._ensure_client()

        trades = await self._client.get_recent_trades(symbol=symbol, limit=limit)
        return trades

    async def close(self) -> None:
        """Close client connections."""
        if self._client:
            await self._client.close_connection()
            self._client = None
        self._price_callbacks.clear()
        logger.info("Binance client closed")
