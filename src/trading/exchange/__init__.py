"""Exchange integration module."""

from src.trading.exchange.base import BaseExchange
from src.trading.exchange.paper_exchange import PaperExchange

__all__ = [
    "BaseExchange",
    "PaperExchange",
]

# Lazy imports for optional dependencies
def get_binance_client():
    """Get BinanceClient class (requires python-binance)."""
    from src.trading.exchange.binance_client import BinanceClient
    return BinanceClient

def get_binance_websocket():
    """Get BinanceWebSocket class (requires python-binance)."""
    from src.trading.exchange.binance_ws import BinanceWebSocket
    return BinanceWebSocket

def get_bybit_client():
    """Get BybitClient class (requires pybit)."""
    from src.trading.exchange.bybit_client import BybitClient
    return BybitClient

def get_bybit_websocket():
    """Get BybitWebSocket class (no auth for public spot stream)."""
    from src.trading.exchange.bybit_ws import BybitWebSocket
    return BybitWebSocket
