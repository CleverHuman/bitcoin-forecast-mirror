"""Binance WebSocket streams for real-time data."""

import asyncio
import logging
from decimal import Decimal
from typing import Any, Callable

from src.trading.config import TradingConfig

logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """Binance WebSocket manager for real-time price streams.

    Handles WebSocket connections to Binance for:
    - Real-time price updates
    - Trade streams
    - Order book updates

    Attributes:
        config: Trading configuration
        callbacks: Registered callback functions
        running: Whether the stream is active
    """

    def __init__(self, config: TradingConfig):
        """Initialize WebSocket manager.

        Args:
            config: Trading configuration
        """
        self.config = config
        self._callbacks: dict[str, list[Callable]] = {
            "price": [],
            "trade": [],
            "kline": [],
        }
        self._running = False
        self._ws_client = None
        self._reconnect_delay = 5
        self._max_reconnect_delay = 300

    async def start(self) -> None:
        """Start WebSocket connections."""
        if self._running:
            logger.warning("WebSocket already running")
            return

        try:
            from binance import AsyncClient, BinanceSocketManager
            self._AsyncClient = AsyncClient
            self._BinanceSocketManager = BinanceSocketManager
        except ImportError:
            raise ImportError(
                "python-binance is required for WebSocket streams. "
                "Install with: pip install python-binance"
            )

        self._running = True
        asyncio.create_task(self._run_stream())
        logger.info("WebSocket stream started")

    async def _run_stream(self) -> None:
        """Main WebSocket stream loop with reconnection."""
        reconnect_delay = self._reconnect_delay

        while self._running:
            try:
                # Create client
                client = await self._AsyncClient.create(
                    api_key=self.config.binance_api_key,
                    api_secret=self.config.binance_api_secret,
                )

                # Create socket manager
                bm = self._BinanceSocketManager(client)

                # Start mini ticker stream for price updates
                symbol_lower = self.config.symbol.lower()
                ts = bm.symbol_ticker_socket(symbol_lower)

                async with ts as stream:
                    logger.info(f"Connected to {self.config.symbol} ticker stream")
                    reconnect_delay = self._reconnect_delay  # Reset on success

                    while self._running:
                        try:
                            msg = await asyncio.wait_for(
                                stream.recv(),
                                timeout=30.0,
                            )
                            await self._handle_ticker_message(msg)
                        except asyncio.TimeoutError:
                            # Ping to keep alive
                            continue

            except asyncio.CancelledError:
                logger.info("WebSocket stream cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self._running:
                    logger.info(f"Reconnecting in {reconnect_delay}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2,
                        self._max_reconnect_delay,
                    )

            finally:
                try:
                    if client:
                        await client.close_connection()
                except Exception:
                    pass

    async def _handle_ticker_message(self, msg: dict[str, Any]) -> None:
        """Handle incoming ticker message.

        Args:
            msg: Ticker message from Binance
        """
        if "e" not in msg:
            return

        event_type = msg["e"]

        if event_type == "24hrMiniTicker":
            # Mini ticker with current price
            symbol = msg["s"]
            price = Decimal(msg["c"])  # Current close price

            for callback in self._callbacks["price"]:
                try:
                    callback(symbol, price)
                except Exception as e:
                    logger.error(f"Error in price callback: {e}")

        elif event_type == "trade":
            # Individual trade
            trade_data = {
                "symbol": msg["s"],
                "price": Decimal(msg["p"]),
                "quantity": Decimal(msg["q"]),
                "time": msg["T"],
                "is_buyer_maker": msg["m"],
            }

            for callback in self._callbacks["trade"]:
                try:
                    callback(trade_data)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")

    async def start_trade_stream(self) -> None:
        """Start individual trade stream for more detailed data."""
        if not self._running:
            await self.start()

        try:
            client = await self._AsyncClient.create(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
            )
            bm = self._BinanceSocketManager(client)

            symbol_lower = self.config.symbol.lower()
            ts = bm.trade_socket(symbol_lower)

            async with ts as stream:
                logger.info(f"Connected to {self.config.symbol} trade stream")
                while self._running:
                    try:
                        msg = await asyncio.wait_for(stream.recv(), timeout=30.0)
                        await self._handle_ticker_message(msg)
                    except asyncio.TimeoutError:
                        continue

        except Exception as e:
            logger.error(f"Trade stream error: {e}")
        finally:
            try:
                await client.close_connection()
            except Exception:
                pass

    async def start_kline_stream(self, interval: str = "1h") -> None:
        """Start kline/candlestick stream.

        Args:
            interval: Kline interval (e.g., "1m", "5m", "1h", "1d")
        """
        if not self._running:
            await self.start()

        try:
            client = await self._AsyncClient.create(
                api_key=self.config.binance_api_key,
                api_secret=self.config.binance_api_secret,
            )
            bm = self._BinanceSocketManager(client)

            symbol_lower = self.config.symbol.lower()
            ts = bm.kline_socket(symbol_lower, interval=interval)

            async with ts as stream:
                logger.info(f"Connected to {self.config.symbol} kline stream ({interval})")
                while self._running:
                    try:
                        msg = await asyncio.wait_for(stream.recv(), timeout=60.0)
                        await self._handle_kline_message(msg)
                    except asyncio.TimeoutError:
                        continue

        except Exception as e:
            logger.error(f"Kline stream error: {e}")
        finally:
            try:
                await client.close_connection()
            except Exception:
                pass

    async def _handle_kline_message(self, msg: dict[str, Any]) -> None:
        """Handle incoming kline message.

        Args:
            msg: Kline message from Binance
        """
        if "e" not in msg or msg["e"] != "kline":
            return

        kline = msg["k"]
        kline_data = {
            "symbol": kline["s"],
            "interval": kline["i"],
            "open_time": kline["t"],
            "close_time": kline["T"],
            "open": Decimal(kline["o"]),
            "high": Decimal(kline["h"]),
            "low": Decimal(kline["l"]),
            "close": Decimal(kline["c"]),
            "volume": Decimal(kline["v"]),
            "is_closed": kline["x"],
        }

        for callback in self._callbacks["kline"]:
            try:
                callback(kline_data)
            except Exception as e:
                logger.error(f"Error in kline callback: {e}")

    def on_price(self, callback: Callable[[str, Decimal], None]) -> None:
        """Register callback for price updates.

        Args:
            callback: Function(symbol, price) to call on price update
        """
        self._callbacks["price"].append(callback)

    def on_trade(self, callback: Callable[[dict], None]) -> None:
        """Register callback for trade updates.

        Args:
            callback: Function(trade_data) to call on trade
        """
        self._callbacks["trade"].append(callback)

    def on_kline(self, callback: Callable[[dict], None]) -> None:
        """Register callback for kline updates.

        Args:
            callback: Function(kline_data) to call on kline update
        """
        self._callbacks["kline"].append(callback)

    def remove_callback(self, callback_type: str, callback: Callable) -> bool:
        """Remove a registered callback.

        Args:
            callback_type: Type of callback ("price", "trade", "kline")
            callback: Callback to remove

        Returns:
            True if removed, False if not found
        """
        if callback_type in self._callbacks:
            try:
                self._callbacks[callback_type].remove(callback)
                return True
            except ValueError:
                return False
        return False

    async def stop(self) -> None:
        """Stop WebSocket connections."""
        self._running = False
        self._callbacks = {"price": [], "trade": [], "kline": []}
        logger.info("WebSocket stream stopped")

    @property
    def is_running(self) -> bool:
        """Check if WebSocket is running."""
        return self._running


class BinanceUserDataStream:
    """Binance user data stream for account updates.

    Handles real-time updates for:
    - Account balance changes
    - Order updates
    - Trade execution notifications
    """

    def __init__(self, config: TradingConfig):
        """Initialize user data stream.

        Args:
            config: Trading configuration
        """
        self.config = config
        self._callbacks: dict[str, list[Callable]] = {
            "balance": [],
            "order": [],
            "execution": [],
        }
        self._running = False
        self._listen_key: str | None = None

    async def start(self) -> None:
        """Start user data stream."""
        if self._running:
            return

        try:
            from binance import AsyncClient, BinanceSocketManager
        except ImportError:
            raise ImportError("python-binance required for user data stream")

        self._running = True

        client = await AsyncClient.create(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
        )

        try:
            # Start user socket
            bm = BinanceSocketManager(client)
            us = bm.user_socket()

            async with us as stream:
                logger.info("Connected to user data stream")

                # Start keepalive task
                keepalive_task = asyncio.create_task(
                    self._keepalive_loop(client)
                )

                try:
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(stream.recv(), timeout=60.0)
                            await self._handle_user_message(msg)
                        except asyncio.TimeoutError:
                            continue
                finally:
                    keepalive_task.cancel()

        finally:
            await client.close_connection()

    async def _keepalive_loop(self, client) -> None:
        """Send periodic keepalive to maintain stream.

        Args:
            client: Binance client
        """
        while self._running:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                await client.stream_keepalive(self._listen_key)
                logger.debug("User stream keepalive sent")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Keepalive error: {e}")

    async def _handle_user_message(self, msg: dict[str, Any]) -> None:
        """Handle incoming user data message.

        Args:
            msg: User data message
        """
        event_type = msg.get("e")

        if event_type == "outboundAccountPosition":
            # Balance update
            balances = {
                b["a"]: {
                    "free": Decimal(b["f"]),
                    "locked": Decimal(b["l"]),
                }
                for b in msg.get("B", [])
            }

            for callback in self._callbacks["balance"]:
                try:
                    callback(balances)
                except Exception as e:
                    logger.error(f"Error in balance callback: {e}")

        elif event_type == "executionReport":
            # Order update
            order_update = {
                "symbol": msg["s"],
                "order_id": msg["i"],
                "client_order_id": msg["c"],
                "side": msg["S"],
                "order_type": msg["o"],
                "status": msg["X"],
                "quantity": Decimal(msg["q"]),
                "price": Decimal(msg["p"]),
                "filled_qty": Decimal(msg["z"]),
                "last_fill_qty": Decimal(msg["l"]),
                "last_fill_price": Decimal(msg["L"]),
                "commission": Decimal(msg["n"]) if msg["n"] else Decimal("0"),
                "commission_asset": msg.get("N"),
            }

            for callback in self._callbacks["order"]:
                try:
                    callback(order_update)
                except Exception as e:
                    logger.error(f"Error in order callback: {e}")

            # Also trigger execution callback for fills
            if order_update["last_fill_qty"] > 0:
                for callback in self._callbacks["execution"]:
                    try:
                        callback(order_update)
                    except Exception as e:
                        logger.error(f"Error in execution callback: {e}")

    def on_balance_update(self, callback: Callable[[dict], None]) -> None:
        """Register callback for balance updates."""
        self._callbacks["balance"].append(callback)

    def on_order_update(self, callback: Callable[[dict], None]) -> None:
        """Register callback for order updates."""
        self._callbacks["order"].append(callback)

    def on_execution(self, callback: Callable[[dict], None]) -> None:
        """Register callback for trade executions."""
        self._callbacks["execution"].append(callback)

    async def stop(self) -> None:
        """Stop user data stream."""
        self._running = False
        self._callbacks = {"balance": [], "order": [], "execution": []}
        logger.info("User data stream stopped")
