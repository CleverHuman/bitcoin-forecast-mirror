"""Fetch spot price from Binance public API (no auth required)."""

import logging
from decimal import Decimal

import aiohttp

logger = logging.getLogger(__name__)

BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


async def fetch_spot_price(symbol: str) -> Decimal | None:
    """Fetch current spot price for a symbol from Binance public API.

    No API key required. Use for paper trading with live prices.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").

    Returns:
        Current price as Decimal, or None on error.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                BINANCE_TICKER_URL,
                params={"symbol": symbol},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Binance ticker returned status %s for %s",
                        resp.status,
                        symbol,
                    )
                    return None
                data = await resp.json()
                return Decimal(data["price"])
    except Exception as e:
        logger.warning("Failed to fetch Binance price for %s: %s", symbol, e)
        return None
