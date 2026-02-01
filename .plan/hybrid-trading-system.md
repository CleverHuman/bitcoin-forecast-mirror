# Hybrid Live Trading System - Implementation Plan

## Overview

Build a medium-frequency trading bot centered around your daily ProphetCycleForecaster predictions. The system uses forecasts for directional bias and tactical signals for entry timing.

**Exchange**: Bybit
**Capital**: $500k (phased: $25k → $100k → $250k → $400k)
**Target**: 10-50 trades/month, 4h-2wk holding periods

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LIVE TRADING SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │    Bybit     │     │  Historical  │     │   Prophet   │  │
│  │  WebSocket   │     │  Databricks  │     │  Forecaster │  │
│  └──────┬───────┘     └──────┬───────┘     └──────┬──────┘  │
│         │                    │                    │         │
│         └────────────┬───────┴────────────────────┘         │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              DATA MANAGER                                │ │
│  │   (Real-time prices + historical + forecast merge)       │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           HYBRID STRATEGY ENGINE                         │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │ Forecast Layer (60%): Daily yhat_ensemble          │ │ │
│  │  │   - Predicted price N days ahead                   │ │ │
│  │  │   - Cycle phase (accumulation/bull/bear)           │ │ │
│  │  │   - Confidence intervals                           │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  │  ┌────────────────────────────────────────────────────┐ │ │
│  │  │ Tactical Layer (40%): Entry timing                 │ │ │
│  │  │   - Momentum confirmation                          │ │ │
│  │  │   - RSI/MACD for overbought/oversold               │ │ │
│  │  │   - Price vs forecast deviation                    │ │ │
│  │  └────────────────────────────────────────────────────┘ │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              RISK MANAGER                                │ │
│  │   Pre-trade checks | Stop-loss | Kill switch            │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              ORDER MANAGER                               │ │
│  │   Order placement | Fill tracking | Retries             │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │         POSITION & PnL TRACKER                           │ │
│  │   Entry price | Current PnL | Equity curve              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## New Module Structure

```
src/trading/                          # NEW MODULE
├── __init__.py
├── config.py                         # TradingConfig dataclass
│
├── exchange/                         # Exchange integration
│   ├── __init__.py
│   ├── base.py                       # BaseExchange interface
│   ├── bybit_client.py               # Bybit REST API (V5)
│   ├── bybit_ws.py                   # Bybit WebSocket streams
│   └── paper_exchange.py             # Simulated exchange for paper trading
│
├── orders/                           # Order management
│   ├── __init__.py
│   ├── order.py                      # Order, OrderState, Fill dataclasses
│   └── order_manager.py              # Order lifecycle, retries
│
├── position/                         # Position tracking
│   ├── __init__.py
│   ├── position_tracker.py           # Position state, entry prices
│   └── pnl.py                        # PnL calculation, equity curve
│
├── risk/                             # Risk management
│   ├── __init__.py
│   ├── risk_manager.py               # Pre-trade checks, limits
│   └── kill_switch.py                # Emergency stop functionality
│
├── strategy/                         # Live strategy wrapper
│   ├── __init__.py
│   └── live_strategy.py              # Wraps existing strategies for live use
│
├── alerts/                           # Notifications
│   ├── __init__.py
│   └── telegram.py                   # Telegram bot alerts
│
└── data/                             # Real-time data management
    ├── __init__.py
    └── data_manager.py               # Merges historical + real-time + forecast

live_trader.py                        # Main entry point
```

---

## Key Components

### 1. TradingConfig (`src/trading/config.py`)

```python
@dataclass
class TradingConfig:
    # Mode
    paper_trading: bool = True

    # Capital & Position Sizing
    initial_capital: float = 25000      # Start with $25k
    max_position_pct: float = 0.10      # 10% max per trade ($50k at full scale)
    max_total_exposure_pct: float = 0.70  # 70% max in positions

    # Risk Limits
    daily_loss_limit_pct: float = 0.02   # 2% daily loss = pause
    weekly_loss_limit_pct: float = 0.05  # 5% weekly loss = pause
    max_drawdown_pct: float = 0.15       # 15% drawdown = full stop
    stop_loss_pct: float = 0.08          # 8% stop loss per position

    # Execution
    min_trade_usd: float = 500
    fee_pct: float = 0.055               # 0.055% Bybit taker fee
    slippage_pct: float = 0.05           # 0.05% slippage estimate

    # Strategy
    forecast_weight: float = 0.60        # 60% forecast, 40% tactical
    lookforward_days: int = 30           # Forecast horizon for signals
    min_signal_score: float = 0.3        # Minimum score to act
    min_trade_interval_hours: int = 4    # Prevent overtrading

    # Forecast refresh
    forecast_refresh_hours: int = 24     # Re-run forecast daily

    # Bybit specific
    bybit_category: str = "spot"         # "spot" or "linear" for perpetuals
```

### 2. BaseExchange Interface (`src/trading/exchange/base.py`)

```python
class BaseExchange(ABC):
    @abstractmethod
    async def get_price(self, symbol: str) -> Decimal

    @abstractmethod
    async def get_balance(self, asset: str) -> Decimal

    @abstractmethod
    async def place_market_order(self, symbol: str, side: str, qty: Decimal) -> Order

    @abstractmethod
    async def place_limit_order(self, symbol: str, side: str, price: Decimal, qty: Decimal) -> Order

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderState

    @abstractmethod
    def subscribe_price(self, symbol: str, callback: Callable) -> None
```

### 3. Bybit Client (`src/trading/exchange/bybit_client.py`)

```python
class BybitClient(BaseExchange):
    """Bybit V5 API integration for spot and linear perpetuals"""

    def __init__(self, config: TradingConfig):
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        self.category = config.bybit_category  # "spot" or "linear"

        # Bybit V5 endpoints
        self.base_url = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"

    async def get_price(self, symbol: str = "BTCUSDT") -> Decimal:
        """Get current mark price"""

    async def place_market_order(self, symbol: str, side: str, qty: Decimal) -> Order:
        """Place market order via V5 API"""
        # POST /v5/order/create

    async def get_wallet_balance(self) -> dict:
        """Get unified account balance"""
        # GET /v5/account/wallet-balance
```

### 4. RiskManager (`src/trading/risk/risk_manager.py`)

```python
class RiskManager:
    def __init__(self, config: TradingConfig, position_tracker: PositionTracker):
        self.config = config
        self.positions = position_tracker
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_equity = config.initial_capital
        self.kill_switch_active = False

    def check_pre_trade(self, signal: StrategySignal, order_size_usd: float) -> tuple[bool, str]:
        """Returns (allowed, reason)"""
        # Check kill switch
        # Check drawdown limits
        # Check daily/weekly loss limits
        # Check position limits
        # Check minimum trade size

    def update_pnl(self, current_equity: float) -> None:
        """Update PnL tracking, check limits"""

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss triggered"""

    def activate_kill_switch(self, reason: str) -> None:
        """Emergency stop - close all positions, stop trading"""
```

### 5. LiveStrategy Wrapper (`src/trading/strategy/live_strategy.py`)

```python
class LiveStrategy:
    """Wraps existing strategies for live trading with forecast-centric logic"""

    def __init__(
        self,
        forecaster: ProphetCycleForecaster,
        config: TradingConfig,
        halving_averages: HalvingAverages,
    ):
        self.forecaster = forecaster
        self.config = config
        self.forecast_result: ForecastResult | None = None
        self.last_forecast_time: datetime | None = None

    def refresh_forecast(self, historical_df: pd.DataFrame) -> None:
        """Re-run Prophet forecast (daily)"""
        self.forecast_result = self.forecaster.fit_predict(historical_df, periods=365)
        self.last_forecast_time = datetime.now()

    def get_signal(self, current_price: float, current_date: date) -> StrategySignal:
        """
        Generate signal centered on forecast:
        1. Get predicted price N days ahead from forecast
        2. Calculate predicted change %
        3. Check cycle phase (from forecast)
        4. Add tactical confirmation (momentum, RSI)
        5. Return weighted signal
        """

    def _forecast_score(self, current_price: float, current_date: date) -> float:
        """Score based on forecast direction and magnitude"""
        predicted = self._get_predicted_price(current_date + timedelta(days=self.config.lookforward_days))
        change_pct = (predicted - current_price) / current_price * 100

        # Strong signals: 10%+ predicted move
        # Moderate: 5-10%
        # Weak: <5%
        return np.clip(change_pct / 15, -1, 1)

    def _tactical_score(self, df: pd.DataFrame) -> float:
        """Score based on entry timing signals"""
        # Price vs forecast deviation
        # Short-term momentum
        # RSI confirmation
```

### 6. Main Trading Loop (`live_trader.py`)

```python
async def main():
    # 1. Load config
    config = TradingConfig.from_env()

    # 2. Initialize components
    exchange = PaperExchange(config) if config.paper_trading else BybitClient(config)
    position_tracker = PositionTracker(config)
    risk_manager = RiskManager(config, position_tracker)
    order_manager = OrderManager(exchange)
    alerter = TelegramAlerter(config)

    # 3. Load historical data and create forecaster
    historical_df = fetch_btc_data("2015-01-01", datetime.now().strftime("%Y-%m-%d"))
    cycle_metrics = compute_cycle_metrics(historical_df)
    halving_averages = compute_halving_averages(cycle_metrics)
    forecaster = ProphetCycleForecaster(halving_averages, cycle_metrics)

    # 4. Initialize strategy
    strategy = LiveStrategy(forecaster, config, halving_averages)
    strategy.refresh_forecast(historical_df)

    # 5. Subscribe to price updates
    data_manager = DataManager(exchange, historical_df)

    # 6. Main loop
    while True:
        try:
            # Refresh forecast if stale
            if strategy.needs_refresh():
                strategy.refresh_forecast(data_manager.get_historical())

            # Get current state
            current_price = await exchange.get_price("BTCUSDT")
            current_date = datetime.now().date()

            # Generate signal
            signal = strategy.get_signal(current_price, current_date)

            # Check risk limits
            allowed, reason = risk_manager.check_pre_trade(signal, ...)
            if not allowed:
                logger.warning(f"Trade blocked: {reason}")
                continue

            # Execute if signal strong enough
            if abs(signal.score) >= config.min_signal_score:
                if signal.signal in [Signal.BUY, Signal.STRONG_BUY]:
                    order = await order_manager.buy(signal, config)
                elif signal.signal in [Signal.SELL, Signal.STRONG_SELL]:
                    order = await order_manager.sell(signal, config)

                await alerter.send_trade_alert(order, signal)

            # Update PnL
            risk_manager.update_pnl(position_tracker.get_equity(current_price))

            # Check stop losses
            for position in position_tracker.open_positions:
                if risk_manager.check_stop_loss(position, current_price):
                    await order_manager.close_position(position)
                    await alerter.send_stop_loss_alert(position)

            # Sleep until next check (hourly)
            await asyncio.sleep(3600)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await alerter.send_error_alert(e)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Files: 8)
- [ ] `src/trading/__init__.py`
- [ ] `src/trading/config.py` - TradingConfig dataclass
- [ ] `src/trading/exchange/__init__.py`
- [ ] `src/trading/exchange/base.py` - BaseExchange interface
- [ ] `src/trading/exchange/paper_exchange.py` - Simulated exchange
- [ ] `src/trading/orders/__init__.py`
- [ ] `src/trading/orders/order.py` - Order, Fill dataclasses
- [ ] `src/trading/orders/order_manager.py` - Order lifecycle

### Phase 2: Position & Risk (Files: 5)
- [ ] `src/trading/position/__init__.py`
- [ ] `src/trading/position/position_tracker.py` - Position state
- [ ] `src/trading/position/pnl.py` - PnL calculation
- [ ] `src/trading/risk/__init__.py`
- [ ] `src/trading/risk/risk_manager.py` - Risk limits, checks, kill switch

### Phase 3: Live Strategy (Files: 4)
- [ ] `src/trading/strategy/__init__.py`
- [ ] `src/trading/strategy/live_strategy.py` - Forecast-centric wrapper
- [ ] `src/trading/data/__init__.py`
- [ ] `src/trading/data/data_manager.py` - Real-time + historical merge

### Phase 4: Main Loop & Alerts (Files: 3)
- [ ] `src/trading/alerts/__init__.py`
- [ ] `src/trading/alerts/telegram.py` - Telegram notifications
- [ ] `live_trader.py` - Main entry point

### Phase 5: Bybit Integration (Files: 2)
- [ ] `src/trading/exchange/bybit_client.py` - REST API (V5)
- [ ] `src/trading/exchange/bybit_ws.py` - WebSocket streams

---

## Files to Modify

| File | Change |
|------|--------|
| `requirements.txt` | Add: `pybit`, `python-telegram-bot`, `aiohttp` |
| `.env.example` | Add: Bybit API keys, Telegram token, trading config |

---

## New Dependencies

```
pybit>=5.6.0                # Bybit V5 API SDK
python-telegram-bot>=20.0   # Telegram alerts
aiohttp>=3.9.0              # Async HTTP
```

---

## Environment Variables (New)

```bash
# Bybit API
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=true              # Use testnet for paper trading

# Trading Config
TRADING_PAPER_MODE=true
TRADING_INITIAL_CAPITAL=25000
TRADING_MAX_POSITION_PCT=0.10
TRADING_MAX_DRAWDOWN_PCT=0.15

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Risk Management Summary

| Limit | Value | Action |
|-------|-------|--------|
| Max position size | 10% of capital | Block trade |
| Max total exposure | 70% of capital | Block new buys |
| Per-trade stop loss | 8% | Close position |
| Daily loss limit | 2% | Pause trading 24h |
| Weekly loss limit | 5% | Pause trading 7d |
| Max drawdown | 15% | Kill switch, close all |

---

## Bybit-Specific Notes

- **API Version**: V5 Unified Account API
- **Symbol**: BTCUSDT (spot) or BTCUSDT perpetual
- **Fees**: 0.055% taker, 0.02% maker (with VIP tiers lower)
- **Testnet**: Available at api-testnet.bybit.com for paper trading
- **Rate Limits**: 120 requests/second for order endpoints
- **WebSocket**: Real-time orderbook, trades, account updates

---

## Verification Plan

1. **Unit tests**: Each component in isolation
2. **Bybit testnet**: Paper trade with testnet API for 2-4 weeks
3. **Metrics to track**:
   - Signal accuracy vs actual price moves
   - Execution latency
   - Fill rates and slippage
   - Risk limit triggers
4. **Go-live criteria**:
   - Testnet trading profitable or break-even
   - No unexpected risk limit triggers
   - All alerts working
   - Manual intervention tested

---

## File Count Summary

- **New files**: 22
- **Modified files**: 2
- **Total**: 24 files

This design keeps your forecast as the core decision-maker while adding the infrastructure needed for live trading on Bybit with proper risk controls.
