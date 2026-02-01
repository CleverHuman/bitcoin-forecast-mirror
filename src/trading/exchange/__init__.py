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
