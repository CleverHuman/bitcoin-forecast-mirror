"""
Live trading module for Bitcoin forecasting system.

This module provides infrastructure for hybrid live trading using
ProphetCycleForecaster predictions as the core decision engine.

Key Components:
- TradingConfig: Configuration for trading parameters and risk limits
- LiveStrategy: Forecast-centric trading strategy wrapper
- OrderManager: Order placement and lifecycle management
- RiskManager: Risk controls and kill switch
- PositionTracker: Position state and P&L tracking
- DataManager: Real-time and historical data integration
- TelegramAlerter: Trade notifications via Telegram

Exchanges:
- PaperExchange: Simulated exchange for testing
- BinanceClient: Live trading via Binance REST API
- BinanceWebSocket: Real-time price streams

Usage:
    from src.trading import TradingConfig
    from src.trading.exchange import PaperExchange
    from src.trading.strategy import LiveStrategy

    config = TradingConfig.from_env()
    exchange = PaperExchange(config)
    # ... setup other components ...

For live trading, run:
    python live_trader.py --paper  # Paper trading (default)
    python live_trader.py --live   # Live trading (requires API keys)
"""

from src.trading.config import TradingConfig

__all__ = ["TradingConfig"]
