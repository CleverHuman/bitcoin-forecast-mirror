"""Fetch spot price from Bybit public API (no auth required)."""

import logging
from decimal import Decimal

import aiohttp

logger = logging.getLogger(__name__)

BYBIT_TICKERS_MAINNET = "https://api.bybit.com/v5/market/tickers"
BYBIT_TICKERS_TESTNET = "https://api-testnet.bybit.com/v5/market/tickers"


async def fetch_spot_price(symbol: str, testnet: bool = False) -> Decimal | None:
    """Fetch current spot price for a symbol from Bybit public API.

    No API key required. Use for paper trading with live prices.

    Args:
        symbol: Trading pair (e.g. "BTCUSDT").
        testnet: If True, use Bybit testnet (demo) API.

    Returns:
        Current price as Decimal, or None on error.
    """
    url = BYBIT_TICKERS_TESTNET if testnet else BYBIT_TICKERS_MAINNET
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params={"category": "spot", "symbol": symbol},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Bybit tickers returned status %s for %s",
                        resp.status,
                        symbol,
                    )
                    return None
                data = await resp.json()
                if data.get("retCode") != 0:
                    logger.warning(
                        "Bybit tickers retCode %s: %s",
                        data.get("retCode"),
                        data.get("retMsg"),
                    )
                    return None
                lst = data.get("result", {}).get("list") or []
                if not lst:
                    return None
                return Decimal(lst[0]["lastPrice"])
    except Exception as e:
        logger.warning("Failed to fetch Bybit price for %s: %s", symbol, e)
        return None
