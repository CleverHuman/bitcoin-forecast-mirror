"""Trading configuration dataclass."""

from dataclasses import dataclass, field
from typing import Any
import os


@dataclass
class TradingConfig:
    """Configuration for the live trading system.

    Attributes:
        paper_trading: If True, use simulated exchange (no real trades)
        paper_use_live_prices: In paper mode, fetch live price from Binance public API (no key)
        use_ws_price: Use WebSocket for live price (real-time); if False, REST poll each iteration
        initial_capital: Starting capital in USD
        max_position_pct: Maximum position size as fraction of capital (0.10 = 10%, 2.0 = 2x leverage)
        max_total_exposure_pct: Maximum total exposure as fraction of capital (0.70 = 70%, 2.0 = 2x)
        daily_loss_limit_pct: Daily loss limit as % of capital (triggers pause)
        weekly_loss_limit_pct: Weekly loss limit as % of capital (triggers pause)
        max_drawdown_pct: Maximum drawdown from peak (triggers kill switch)
        stop_loss_pct: Per-position stop loss percentage
        min_trade_usd: Minimum trade size in USD
        fee_pct: Exchange fee percentage
        slippage_pct: Expected slippage percentage
        forecast_weight: Weight for forecast signals (vs tactical)
        lookforward_days: Days ahead for forecast signal calculation
        min_signal_score: Minimum signal score to trigger a trade
        min_trade_interval_hours: Minimum hours between trades
        forecast_refresh_hours: Hours between forecast refreshes
        symbol: Trading pair symbol
        base_asset: Base asset (e.g., BTC)
        quote_asset: Quote asset (e.g., USDT)
        exchange: Exchange to use: "binance" or "bybit"
        bybit_testnet: If True, use Bybit testnet (demo account)
        binance_api_key: Binance API key (for live trading)
        binance_api_secret: Binance API secret (for live trading)
        bybit_api_key: Bybit API key (system-generated public key)
        bybit_api_secret: Bybit API secret (system-generated private key)
        telegram_token: Telegram bot token for alerts
        telegram_chat_id: Telegram chat ID for alerts
    """

    # Mode
    paper_trading: bool = True
    paper_use_live_prices: bool = True  # In paper mode, fetch live price from Binance (no API key)
    use_ws_price: bool = True  # Use WebSocket for live price (real-time); if False, use REST each iteration

    # Capital & Position Sizing
    initial_capital: float = 25000.0
    max_position_pct: float = 0.10
    max_total_exposure_pct: float = 0.70

    # Risk Limits
    daily_loss_limit_pct: float = 0.02
    weekly_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    stop_loss_pct: float = 0.08

    # Execution
    min_trade_usd: float = 500.0
    fee_pct: float = 0.10
    slippage_pct: float = 0.05

    # Strategy
    forecast_weight: float = 0.60
    lookforward_days: int = 30
    min_signal_score: float = 0.3
    min_trade_interval_hours: int = 4

    # Forecast refresh
    forecast_refresh_hours: int = 24

    # Trading pair
    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"

    # Exchange selection (binance | bybit)
    exchange: str = "binance"
    # Bybit demo/testnet: use testnet.bybit.com (demo account). Mainnet if False.
    bybit_testnet: bool = False

    # API credentials (loaded from environment)
    binance_api_key: str = ""
    binance_api_secret: str = ""
    bybit_api_key: str = ""      # System-generated API key (public)
    bybit_api_secret: str = ""   # System-generated API secret (private)
    telegram_token: str = ""
    telegram_chat_id: str = ""

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "TradingConfig":
        """Create config from environment variables.

        Environment variables (all optional, defaults used if not set):
            PAPER_TRADING: "true" or "false"
            PAPER_USE_LIVE_PRICES: "true" or "false" (paper mode: use live price from Binance)
            USE_WS_PRICE: "true" or "false" (use WebSocket for price; no API key needed for public stream)
            INITIAL_CAPITAL: Starting capital in USD
            MAX_POSITION_PCT: Max position size as decimal (e.g., 0.10)
            MAX_TOTAL_EXPOSURE_PCT: Max total exposure as decimal
            DAILY_LOSS_LIMIT_PCT: Daily loss limit as decimal
            WEEKLY_LOSS_LIMIT_PCT: Weekly loss limit as decimal
            MAX_DRAWDOWN_PCT: Max drawdown as decimal
            STOP_LOSS_PCT: Per-position stop loss as decimal
            MIN_TRADE_USD: Minimum trade size
            FEE_PCT: Exchange fee as decimal
            SLIPPAGE_PCT: Expected slippage as decimal
            FORECAST_WEIGHT: Weight for forecast vs tactical (0-1)
            LOOKFORWARD_DAYS: Days ahead for forecast
            MIN_SIGNAL_SCORE: Minimum signal score (0-1)
            MIN_TRADE_INTERVAL_HOURS: Hours between trades
            FORECAST_REFRESH_HOURS: Hours between forecast refreshes
            TRADING_SYMBOL: Trading pair (default: BTCUSDT)
            EXCHANGE: "binance" or "bybit"
            BYBIT_TESTNET: "true" to use Bybit demo/testnet (testnet.bybit.com)
            BINANCE_API_KEY: Binance API key
            BINANCE_API_SECRET: Binance API secret
            BYBIT_API_KEY: Bybit API key (system-generated)
            BYBIT_API_SECRET: Bybit API secret (system-generated)
            TELEGRAM_TOKEN: Telegram bot token
            TELEGRAM_CHAT_ID: Telegram chat ID
            LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        return cls(
            paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
            paper_use_live_prices=os.getenv("PAPER_USE_LIVE_PRICES", "true").lower() == "true",
            use_ws_price=os.getenv("USE_WS_PRICE", "true").lower() == "true",
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "25000")),
            max_position_pct=float(os.getenv("MAX_POSITION_PCT", "0.10")),
            max_total_exposure_pct=float(os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.70")),
            daily_loss_limit_pct=float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02")),
            weekly_loss_limit_pct=float(os.getenv("WEEKLY_LOSS_LIMIT_PCT", "0.05")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.15")),
            stop_loss_pct=float(os.getenv("STOP_LOSS_PCT", "0.08")),
            min_trade_usd=float(os.getenv("MIN_TRADE_USD", "500")),
            fee_pct=float(os.getenv("FEE_PCT", "0.10")),
            slippage_pct=float(os.getenv("SLIPPAGE_PCT", "0.05")),
            forecast_weight=float(os.getenv("FORECAST_WEIGHT", "0.60")),
            lookforward_days=int(os.getenv("LOOKFORWARD_DAYS", "30")),
            min_signal_score=float(os.getenv("MIN_SIGNAL_SCORE", "0.3")),
            min_trade_interval_hours=int(os.getenv("MIN_TRADE_INTERVAL_HOURS", "4")),
            forecast_refresh_hours=int(os.getenv("FORECAST_REFRESH_HOURS", "24")),
            symbol=os.getenv("TRADING_SYMBOL", "BTCUSDT"),
            base_asset=os.getenv("BASE_ASSET", "BTC"),
            quote_asset=os.getenv("QUOTE_ASSET", "USDT"),
            exchange=os.getenv("EXCHANGE", "binance").lower(),
            bybit_testnet=os.getenv("BYBIT_TESTNET", "false").lower() == "true",
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            bybit_api_key=os.getenv("BYBIT_API_KEY", ""),
            bybit_api_secret=os.getenv("BYBIT_API_SECRET", ""),
            telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors (empty if valid)."""
        errors = []

        if self.initial_capital <= 0:
            errors.append("initial_capital must be positive")

        # Allow > 1 for leverage (e.g. 2.0 = 2x)
        if not 0 < self.max_position_pct <= 10:
            errors.append("max_position_pct must be between 0 and 10 (use > 1 for leverage)")

        if not 0 < self.max_total_exposure_pct <= 10:
            errors.append("max_total_exposure_pct must be between 0 and 10 (use > 1 for leverage)")

        if not 0 < self.daily_loss_limit_pct <= 1:
            errors.append("daily_loss_limit_pct must be between 0 and 1")

        if not 0 < self.weekly_loss_limit_pct <= 1:
            errors.append("weekly_loss_limit_pct must be between 0 and 1")

        if not 0 < self.max_drawdown_pct <= 1:
            errors.append("max_drawdown_pct must be between 0 and 1")

        if not 0 < self.stop_loss_pct <= 1:
            errors.append("stop_loss_pct must be between 0 and 1")

        if self.min_trade_usd <= 0:
            errors.append("min_trade_usd must be positive")

        if not 0 <= self.forecast_weight <= 1:
            errors.append("forecast_weight must be between 0 and 1")

        if self.lookforward_days <= 0:
            errors.append("lookforward_days must be positive")

        if not 0 <= self.min_signal_score <= 1:
            errors.append("min_signal_score must be between 0 and 1")

        if not self.paper_trading:
            if self.exchange == "binance":
                if not self.binance_api_key:
                    errors.append("binance_api_key required for live trading")
                if not self.binance_api_secret:
                    errors.append("binance_api_secret required for live trading")
            elif self.exchange == "bybit":
                if not self.bybit_api_key:
                    errors.append("bybit_api_key required for live trading")
                if not self.bybit_api_secret:
                    errors.append("bybit_api_secret required for live trading")
            else:
                errors.append("EXCHANGE must be 'binance' or 'bybit'")

        return errors

    @property
    def tactical_weight(self) -> float:
        """Weight for tactical signals (complement of forecast_weight)."""
        return 1.0 - self.forecast_weight

    @property
    def max_position_usd(self) -> float:
        """Maximum position size in USD."""
        return self.initial_capital * self.max_position_pct

    @property
    def max_total_exposure_usd(self) -> float:
        """Maximum total exposure in USD."""
        return self.initial_capital * self.max_total_exposure_pct
