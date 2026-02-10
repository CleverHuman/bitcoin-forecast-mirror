"""Bybit REST API client for live trading (v5, spot)."""

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


def _map_order_status(bybit_status: str) -> OrderState:
    """Map Bybit order status to our OrderState."""
    status_map = {
        "New": OrderState.OPEN,
        "PartiallyFilled": OrderState.PARTIALLY_FILLED,
        "Filled": OrderState.FILLED,
        "Cancelled": OrderState.CANCELLED,
        "Rejected": OrderState.REJECTED,
        "Deactivated": OrderState.EXPIRED,
    }
    return status_map.get(bybit_status, OrderState.PENDING)


class BybitClient(BaseExchange):
    """Bybit exchange client for live trading (v5 API, spot).

    Uses pybit for REST. Use system-generated API keys from Bybit (public + private).
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        if not config.bybit_api_key or not config.bybit_api_secret:
            raise ValueError("Bybit API credentials required for live trading")
        try:
            from pybit.unified_trading import HTTP
            self._HTTP = HTTP
        except ImportError:
            raise ImportError(
                "pybit is required for Bybit. Install with: pip install pybit"
            )
        self._session: Any = None
        self._price_callbacks: dict[str, list[Callable[[str, Decimal], None]]] = {}

    def _ensure_session(self) -> None:
        if self._session is None:
            self._session = self._HTTP(
                api_key=self.config.bybit_api_key,
                api_secret=self.config.bybit_api_secret,
                testnet=self.config.bybit_testnet,
            )

    async def get_price(self, symbol: str) -> Decimal:
        def _get() -> Decimal:
            self._ensure_session()
            r = self._session.get_tickers(category="spot", symbol=symbol)
            if r.get("retCode") != 0:
                raise RuntimeError(r.get("retMsg", "Unknown error"))
            lst = (r.get("result") or {}).get("list") or []
            if not lst:
                raise RuntimeError("No ticker data")
            return Decimal(lst[0]["lastPrice"])
        return await asyncio.to_thread(_get)

    async def get_balance(self, asset: str) -> Decimal:
        def _get() -> Decimal:
            self._ensure_session()
            r = self._session.get_wallet_balance(accountType="UNIFIED", coin=asset)
            if r.get("retCode") != 0:
                raise RuntimeError(r.get("retMsg", "Unknown error"))
            for acc in (r.get("result") or {}).get("list") or []:
                for c in acc.get("coin") or []:
                    if c.get("coin") == asset:
                        return Decimal(c.get("walletBalance") or "0")
            return Decimal("0")
        return await asyncio.to_thread(_get)

    async def get_total_balance(self, asset: str) -> Decimal:
        return await self.get_balance(asset)

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: Decimal,
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.MARKET,
            qty=qty,
            state=OrderState.PENDING,
        )

        def _place() -> dict:
            self._ensure_session()
            return self._session.place_order(
                category="spot",
                symbol=symbol,
                side="Buy" if side == "BUY" else "Sell",
                orderType="Market",
                qty=str(qty),
                timeInForce="IOC",
                isLeverage=0,
            )

        try:
            result = await asyncio.to_thread(_place)
            if result.get("retCode") != 0:
                order.state = OrderState.FAILED
                order.error_message = result.get("retMsg", "Unknown error")
                return order
            order.exchange_order_id = result.get("result", {}).get("orderId", "")
            order.state = OrderState.FILLED
            price = await self.get_price(symbol)
            order.add_fill(Fill(
                fill_id=str(uuid.uuid4()),
                order_id=order.order_id,
                price=price,
                qty=qty,
                commission=Decimal("0"),
                commission_asset=self.config.quote_asset,
                timestamp=datetime.now(),
            ))
            logger.info(f"Bybit order executed: {side} {qty} {symbol} @ {price}")
        except Exception as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            logger.error(f"Bybit order failed: {e}")
        return order

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: Decimal,
        qty: Decimal,
    ) -> Order:
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType.LIMIT,
            qty=qty,
            price=price,
            state=OrderState.PENDING,
        )

        def _place() -> dict:
            self._ensure_session()
            return self._session.place_order(
                category="spot",
                symbol=symbol,
                side="Buy" if side == "BUY" else "Sell",
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="GTC",
                isLeverage=0,
            )

        try:
            result = await asyncio.to_thread(_place)
            if result.get("retCode") != 0:
                order.state = OrderState.FAILED
                order.error_message = result.get("retMsg", "Unknown error")
                return order
            order.exchange_order_id = result.get("result", {}).get("orderId", "")
            order.state = OrderState.OPEN
            logger.info(f"Bybit limit order placed: {side} {qty} {symbol} @ {price}")
        except Exception as e:
            order.state = OrderState.FAILED
            order.error_message = str(e)
            logger.error(f"Bybit limit order failed: {e}")
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        def _cancel() -> dict:
            self._ensure_session()
            return self._session.cancel_order(
                category="spot",
                orderId=order_id,
            )

        try:
            result = await asyncio.to_thread(_cancel)
            return result.get("retCode") == 0
        except Exception as e:
            logger.error(f"Failed to cancel Bybit order {order_id}: {e}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderState:
        def _get() -> dict:
            self._ensure_session()
            return self._session.get_order_history(
                category="spot",
                orderId=order_id,
            )

        try:
            result = await asyncio.to_thread(_get)
            if result.get("retCode") != 0:
                raise RuntimeError(result.get("retMsg", "Unknown error"))
            lst = (result.get("result") or {}).get("list") or []
            if not lst:
                raise ValueError(f"Order not found: {order_id}")
            return _map_order_status(lst[0].get("orderStatus", ""))
        except Exception as e:
            logger.error(f"Failed to get Bybit order status {order_id}: {e}")
            raise

    async def get_order(self, order_id: str, symbol: str) -> Order | None:
        def _get() -> dict:
            self._ensure_session()
            return self._session.get_order_history(
                category="spot",
                orderId=order_id,
            )

        try:
            result = await asyncio.to_thread(_get)
            if result.get("retCode") != 0 or not (result.get("result") or {}).get("list"):
                return None
            o = (result["result"]["list"])[0]
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=o.get("symbol", symbol),
                side=OrderSide.BUY if o.get("side") == "Buy" else OrderSide.SELL,
                order_type=OrderType.LIMIT if o.get("orderType") == "Limit" else OrderType.MARKET,
                qty=Decimal(o.get("qty", "0")),
                price=Decimal(o["price"]) if o.get("price") else None,
                state=_map_order_status(o.get("orderStatus", "")),
                exchange_order_id=o.get("orderId", ""),
                filled_qty=Decimal(o.get("cumExecQty", "0")),
            )
            if order.filled_qty and order.filled_qty > 0:
                avg = o.get("avgPrice") or o.get("price") or "0"
                order.add_fill(Fill(
                    fill_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    price=Decimal(avg),
                    qty=order.filled_qty,
                    commission=Decimal("0"),
                    commission_asset=self.config.quote_asset,
                    timestamp=datetime.now(),
                ))
            return order
        except Exception as e:
            logger.error(f"Failed to get Bybit order {order_id}: {e}")
            return None

    def subscribe_price(
        self,
        symbol: str,
        callback: Callable[[str, Decimal], None],
    ) -> None:
        if symbol not in self._price_callbacks:
            self._price_callbacks[symbol] = []
        self._price_callbacks[symbol].append(callback)

    def unsubscribe_price(self, symbol: str) -> None:
        self._price_callbacks.pop(symbol, None)

    async def close(self) -> None:
        self._session = None
        self._price_callbacks.clear()
        logger.info("Bybit client closed")
