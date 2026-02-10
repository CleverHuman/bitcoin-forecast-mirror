"""Bybit WebSocket streams for real-time price (v5 public spot)."""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Any, Callable

import aiohttp

from src.trading.config import TradingConfig

logger = logging.getLogger(__name__)

BYBIT_WS_SPOT_MAINNET = "wss://stream.bybit.com/v5/public/spot"
BYBIT_WS_SPOT_TESTNET = "wss://stream-testnet.bybit.com/v5/public/spot"


class BybitWebSocket:
    """Bybit WebSocket manager for real-time spot price.

    Public stream: no API key required. Use for paper or live price feed.
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self._callbacks: list[Callable[[str, Decimal], None]] = []
        self._running = False
        self._reconnect_delay = 5
        self._max_reconnect_delay = 300

    async def start(self) -> None:
        if self._running:
            logger.warning("Bybit WebSocket already running")
            return
        self._running = True
        asyncio.create_task(self._run_stream())
        logger.info("Bybit WebSocket price stream started")

    async def _run_stream(self) -> None:
        reconnect_delay = self._reconnect_delay
        symbol = self.config.symbol
        topic = f"tickers.{symbol}"
        ws_url = BYBIT_WS_SPOT_TESTNET if self.config.bybit_testnet else BYBIT_WS_SPOT_MAINNET

        while self._running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        ws_url,
                        heartbeat=30,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as ws:
                        await ws.send_str(
                            json.dumps({"op": "subscribe", "args": [topic]})
                        )
                        logger.info(f"Connected to Bybit {symbol} ticker stream")
                        reconnect_delay = self._reconnect_delay

                        async for msg in ws:
                            if not self._running:
                                break
                            if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                                data = msg.data if isinstance(msg.data, str) else msg.data.decode()
                                try:
                                    obj = json.loads(data)
                                except json.JSONDecodeError:
                                    continue
                                if obj.get("topic") == topic and "data" in obj:
                                    d = obj["data"]
                                    last = d.get("lastPrice") if isinstance(d, dict) else (d[0].get("lastPrice") if d else None)
                                    if last is not None:
                                        price = Decimal(str(last))
                                        for cb in self._callbacks:
                                            try:
                                                cb(symbol, price)
                                            except Exception as e:
                                                logger.error("Bybit WS price callback error: %s", e)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bybit WebSocket error: %s", e)
                if self._running:
                    logger.info("Reconnecting in %ss...", reconnect_delay)
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2,
                        self._max_reconnect_delay,
                    )

        self._running = False

    def on_price(self, callback: Callable[[str, Decimal], None]) -> None:
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[str, Decimal], None]) -> bool:
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            return False

    async def stop(self) -> None:
        self._running = False
        self._callbacks.clear()
        logger.info("Bybit WebSocket price stream stopped")

    @property
    def is_running(self) -> bool:
        return self._running
