"""Base exchange interface for trading operations."""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Callable

from src.trading.orders.order import Order, OrderState


class BaseExchange(ABC):
    """Abstract base class for exchange implementations.

    This interface defines the contract that all exchange implementations
    (paper trading, Binance, etc.) must follow.
    """

    @abstractmethod
    async def get_price(self, symbol: str) -> Decimal:
        """Get current price for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")

        Returns:
            Current price as Decimal
        """
        pass

    @abstractmethod
    async def get_balance(self, asset: str) -> Decimal:
        """Get available balance for an asset.

        Args:
            asset: Asset symbol (e.g., "BTC", "USDT")

        Returns:
            Available balance as Decimal
        """
        pass

    @abstractmethod
    async def get_total_balance(self, asset: str) -> Decimal:
        """Get total balance for an asset (including locked).

        Args:
            asset: Asset symbol (e.g., "BTC", "USDT")

        Returns:
            Total balance as Decimal
        """
        pass

    @abstractmethod
    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
    ) -> Order:
        """Place a market order.

        Args:
            symbol: Trading pair symbol
            side: "BUY" or "SELL"
            qty: Quantity to trade (in base asset)

        Returns:
            Order object with execution details
        """
        pass

    @abstractmethod
    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        qty: Decimal,
    ) -> Order:
        """Place a limit order.

        Args:
            symbol: Trading pair symbol
            side: "BUY" or "SELL"
            price: Limit price
            qty: Quantity to trade (in base asset)

        Returns:
            Order object with pending status
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol

        Returns:
            True if cancelled successfully, False otherwise
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> OrderState:
        """Get current status of an order.

        Args:
            order_id: Order ID to check
            symbol: Trading pair symbol

        Returns:
            Current OrderState
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order | None:
        """Get full order details.

        Args:
            order_id: Order ID to fetch
            symbol: Trading pair symbol

        Returns:
            Order object or None if not found
        """
        pass

    @abstractmethod
    def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, Decimal], None],
    ) -> None:
        """Subscribe to real-time price updates.

        Args:
            symbol: Trading pair symbol
            callback: Function to call with (symbol, price) on each update
        """
        pass

    @abstractmethod
    def unsubscribe_price(self, symbol: str) -> None:
        """Unsubscribe from price updates.

        Args:
            symbol: Trading pair symbol
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close exchange connections and clean up resources."""
        pass
