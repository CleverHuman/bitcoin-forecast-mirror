#!/usr/bin/env python3
"""
Live Trading System for Bitcoin using ProphetCycleForecaster.

This is the main entry point for the hybrid trading system that combines:
- ProphetCycleForecaster predictions (60% weight) for directional bias
- Tactical indicators (40% weight) for entry timing

Usage:
    python live_trader.py [--paper] [--capital 50000]

Paper mode (default) uses live BTC prices from Binance (no API key) so you can
evaluate the strategy against real market moves. Each iteration logs equity
and a buy-and-hold comparison; on shutdown you get a final strategy vs B&H summary.
Set PAPER_USE_LIVE_PRICES=false to use a static price (last historical close) instead.
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.db.btc_data import fetch_btc_data
from src.forecasting.prophet_cycle import ProphetCycleForecaster
from src.metrics.halving import compute_cycle_metrics, compute_halving_averages
from src.trading.config import TradingConfig
from src.trading.exchange.paper_exchange import PaperExchange
from src.trading.orders.order_manager import OrderManager
from src.trading.orders.order import OrderState, OrderSide
from src.trading.position.position_tracker import PositionTracker
from src.trading.position.pnl import PnLTracker
from src.trading.risk.risk_manager import RiskManager
from src.trading.strategy.live_strategy import LiveStrategy
from src.trading.data.data_manager import DataManager
from src.trading.alerts.telegram import TelegramAlerter
from src.trading.exchange.binance_public import fetch_spot_price as fetch_binance_spot_price
from src.trading.exchange.binance_ws import BinanceWebSocket
from src.trading.exchange.bybit_public import fetch_spot_price as fetch_bybit_spot_price
from src.trading.exchange.bybit_ws import BybitWebSocket
from src.backtesting.strategies.base import Signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live_trader.log"),
    ],
)
logger = logging.getLogger(__name__)


class LiveTrader:
    """Main live trading orchestrator.

    Coordinates all components and runs the main trading loop.
    """

    def __init__(self, config: TradingConfig):
        """Initialize live trader.

        Args:
            config: Trading configuration
        """
        self.config = config
        self.running = False
        self._shutdown_event = asyncio.Event()

        # Components (initialized in setup)
        self.exchange = None
        self.order_manager = None
        self.position_tracker = None
        self.pnl_tracker = None
        self.risk_manager = None
        self.strategy = None
        self.data_manager = None
        self.alerter = None
        self.forecaster = None
        self._paper_start_price: Decimal | None = None  # For B&H benchmark in paper mode
        self._ws_price: BinanceWebSocket | BybitWebSocket | None = None  # WebSocket for real-time price (optional)
        self._live_price_task: asyncio.Task | None = None  # Backup: logs live price if no WS
        self._last_ws_price_log_time: float = 0.0  # Throttle: log at most once per second on WS tick

    async def setup(self) -> None:
        """Initialize all trading components."""
        logger.info("Setting up trading system...")

        # Validate config
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")

        # Initialize exchange
        if self.config.paper_trading:
            logger.info("Using PAPER trading mode")
            self.exchange = PaperExchange(self.config)
        else:
            logger.warning("LIVE trading mode - real money at risk!")
            if self.config.exchange == "bybit":
                from src.trading.exchange.bybit_client import BybitClient
                self.exchange = BybitClient(self.config)
                logger.info(
                    "Using Bybit (testnet/demo)" if self.config.bybit_testnet else "Using Bybit"
                )
            else:
                from src.trading.exchange.binance_client import BinanceClient
                self.exchange = BinanceClient(self.config)
                logger.info("Using Binance")

        # Initialize trackers
        self.position_tracker = PositionTracker(self.config)
        self.pnl_tracker = PnLTracker(self.config.initial_capital)
        self.risk_manager = RiskManager(
            self.config,
            self.position_tracker,
            self.pnl_tracker,
        )
        self.order_manager = OrderManager(self.exchange, self.config)
        self.alerter = TelegramAlerter(self.config)

        # Load historical data
        logger.info("Loading historical data...")
        end_date = datetime.now().strftime("%Y-%m-%d")
        historical_df = fetch_btc_data("2015-01-01", end_date)
        historical_df = historical_df.sort_values("ds")
        logger.info(f"Loaded {len(historical_df)} days of historical data")

        # Initialize data manager
        self.data_manager = DataManager(self.exchange, self.config, historical_df)

        # Compute cycle metrics
        logger.info("Computing cycle metrics...")
        cycle_metrics = compute_cycle_metrics(historical_df)
        halving_averages = compute_halving_averages(cycle_metrics=cycle_metrics)

        # Initialize forecaster
        self.forecaster = ProphetCycleForecaster(
            halving_averages=halving_averages,
            cycle_metrics=cycle_metrics,
        )

        # Initialize strategy
        self.strategy = LiveStrategy(
            forecaster=self.forecaster,
            config=self.config,
            halving_averages=halving_averages,
        )

        # Generate initial forecast
        logger.info("Generating initial forecast...")
        self.strategy.refresh_forecast(historical_df)

        # Set initial price in paper exchange (and B&H benchmark)
        if self.config.paper_trading:
            latest_price = Decimal(str(historical_df["y"].iloc[-1]))
            self.exchange.set_price(self.config.symbol, latest_price)
            self._paper_start_price = latest_price
            live_note = " (live prices each iteration)" if self.config.paper_use_live_prices else " (static price until live fetch enabled)"
            logger.info(f"Initial price set to ${latest_price:,.2f}{live_note}")

        # WebSocket for real-time price (paper or live; no API key needed for public stream)
        use_live_price = self.config.paper_use_live_prices or not self.config.paper_trading
        if use_live_price and self.config.use_ws_price:
            self.data_manager.subscribe_price_updates(lambda _s, _p: None)  # Register so exchange updates flow to data_manager
            if self.config.exchange == "bybit":
                self._ws_price = BybitWebSocket(self.config)
            else:
                self._ws_price = BinanceWebSocket(self.config)

            def _on_ws_price(symbol: str, price: Decimal) -> None:
                self.exchange.set_price(symbol, price)
                # Log price on every WS update, throttled to once per second so you see it live
                now = time.time()
                if now - self._last_ws_price_log_time >= 1.0:
                    logger.info(f"Live price: ${price:,.2f}")
                    self._last_ws_price_log_time = now

            self._ws_price.on_price(_on_ws_price)
            await self._ws_price.start()
            logger.info("WebSocket price stream started (real-time); price logged on every tick (throttled 1s)")

        # When live prices but no WebSocket, fallback: log price every 30s
        if use_live_price and not self.config.use_ws_price:
            self._live_price_task = asyncio.create_task(self._live_price_logger_loop())
            logger.info("Live price logging started (every 30s)")

        logger.info("Trading system setup complete")

    async def _live_price_logger_loop(self) -> None:
        """Log current price every 30s so you can see live pricing in the console."""
        while self.running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)
                if not self.running or self._shutdown_event.is_set():
                    break
                price = await self.exchange.get_price(self.config.symbol)
                logger.info(f"Live price: ${price:,.2f}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Live price log skipped: %s", e)

    async def run(self) -> None:
        """Run the main trading loop."""
        self.running = True

        # Send startup alert
        await self.alerter.send_startup_message()

        logger.info("Starting main trading loop...")
        check_interval_seconds = 3600  # 1 hour between checks

        try:
            while self.running and not self._shutdown_event.is_set():
                try:
                    await self._trading_iteration()
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}", exc_info=True)
                    await self.alerter.send_error_alert(e, "trading_iteration")

                # Wait for next iteration
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=check_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass  # Normal - continue loop

        finally:
            await self.shutdown()

    async def _trading_iteration(self) -> None:
        """Execute one iteration of the trading loop."""
        logger.info("Starting trading iteration...")

        # In paper mode without WS, fetch live price each iteration; with WS price is already real-time
        if self.config.paper_trading and self.config.paper_use_live_prices and not (self._ws_price and self._ws_price.is_running):
            if self.config.exchange == "bybit":
                live_price = await fetch_bybit_spot_price(
                    self.config.symbol, testnet=self.config.bybit_testnet
                )
            else:
                live_price = await fetch_binance_spot_price(self.config.symbol)
            if live_price is not None:
                self.exchange.set_price(self.config.symbol, live_price)

        # Refresh forecast if needed (run in thread so main loop stays responsive)
        if self.strategy.needs_refresh():
            logger.info("Refreshing forecast...")
            historical_df = self.data_manager.get_historical()
            await asyncio.to_thread(
                self.strategy.refresh_forecast,
                historical_df,
            )
            next_refresh = self.strategy.hours_until_refresh()
            if next_refresh is not None:
                logger.info("Forecast refreshed; next refresh in %.1f hours", next_refresh)

        # Get current price (from exchange; in paper mode may have been set from live feed above)
        current_price = await self.data_manager.fetch_current_price()
        logger.info(f"Current price: ${current_price:,.2f}")

        # Update P&L tracking
        cash_balance = await self.exchange.get_balance(self.config.quote_asset)
        position_value = Decimal("0")
        for position in self.position_tracker.open_positions.values():
            position.update_price(current_price)
            position_value += position.current_value or Decimal("0")

        equity = float(cash_balance + position_value)
        self.risk_manager.update_pnl(
            equity,
            float(cash_balance),
            float(position_value),
        )

        # Check for stop losses
        await self._check_stop_losses(current_price)

        # Check if kill switch is active
        if self.risk_manager.kill_switch_active:
            logger.warning("Kill switch active - closing all positions")
            await self._close_all_positions("kill_switch")
            return

        # Generate signal
        signal = self.strategy.get_signal(float(current_price))
        logger.info(f"Signal: {signal.signal.value} (score: {signal.score:.2f})")

        # Check risk limits
        position_size = self.order_manager.calculate_position_size(current_price)
        order_size_usd = float(position_size * current_price)

        risk_check = self.risk_manager.check_pre_trade(
            side="BUY" if signal.score > 0 else "SELL",
            order_size_usd=order_size_usd,
            signal_score=signal.score,
        )

        if not risk_check.allowed:
            logger.info(f"Trade blocked: {risk_check.reason}")
            return

        # Execute trade if signal is strong enough
        if signal.signal in [Signal.STRONG_BUY, Signal.BUY]:
            await self._execute_buy(signal, position_size, current_price)
        elif signal.signal in [Signal.STRONG_SELL, Signal.SELL]:
            await self._execute_sell(signal, current_price)

        # Log current state and, in paper mode, compare to buy-and-hold
        log_msg = (
            f"Equity: ${equity:,.2f} | "
            f"Positions: {len(self.position_tracker.open_positions)} | "
            f"Daily P&L: {self.pnl_tracker.daily_pnl_pct:.2f}%"
        )
        if self.config.paper_trading and self._paper_start_price and self._paper_start_price > 0:
            bh_value = float(self.config.initial_capital) / float(self._paper_start_price) * float(current_price)
            bh_return_pct = (bh_value - self.config.initial_capital) / self.config.initial_capital * 100
            log_msg += f" | B&H same period: ${bh_value:,.2f} ({bh_return_pct:+.2f}%)"
        logger.info(log_msg)

    async def _execute_buy(
        self,
        signal,
        position_size: Decimal,
        current_price: Decimal,
    ) -> None:
        """Execute a buy order.

        Args:
            signal: Strategy signal
            position_size: Quantity to buy
            current_price: Current price
        """
        logger.info(f"Executing BUY: {position_size} @ ${current_price:,.2f}")

        order = await self.order_manager.buy(
            qty=position_size,
            metadata={
                "signal": signal.signal.value,
                "score": signal.score,
                "reason": signal.reason,
            },
        )

        if order.state == OrderState.FILLED:
            # Open position
            position = self.position_tracker.open_position(order)
            self.risk_manager.record_trade()

            await self.alerter.send_trade_alert(
                order,
                signal_score=signal.score,
                reason=signal.reason,
            )

            logger.info(
                f"Position opened: {position.position_id} - "
                f"{position.qty} @ ${position.entry_price:,.2f}"
            )
        else:
            logger.warning(f"Buy order not filled: {order.state.value}")

    async def _execute_sell(
        self,
        signal,
        current_price: Decimal,
    ) -> None:
        """Execute sell orders to close positions.

        Args:
            signal: Strategy signal
            current_price: Current price
        """
        if not self.position_tracker.open_positions:
            logger.info("No positions to sell")
            return

        for position_id, position in list(self.position_tracker.open_positions.items()):
            logger.info(
                f"Executing SELL for position {position_id}: "
                f"{position.qty} @ ${current_price:,.2f}"
            )

            order = await self.order_manager.sell(
                qty=position.qty,
                metadata={
                    "signal": signal.signal.value,
                    "score": signal.score,
                    "reason": signal.reason,
                    "position_id": position_id,
                },
            )

            if order.state == OrderState.FILLED:
                closed = self.position_tracker.close_position(
                    position_id,
                    order,
                    reason="signal",
                )
                self.risk_manager.record_trade()
                self.pnl_tracker.record_realized_pnl(closed["pnl"])

                await self.alerter.send_trade_alert(
                    order,
                    signal_score=signal.score,
                    reason=signal.reason,
                )

                logger.info(
                    f"Position closed: {position_id} - "
                    f"P&L: ${closed['pnl']:.2f} ({closed['pnl_pct']:.2f}%)"
                )
            else:
                logger.warning(f"Sell order not filled: {order.state.value}")

    async def _check_stop_losses(self, current_price: Decimal) -> None:
        """Check and execute stop losses.

        Args:
            current_price: Current market price
        """
        triggered_positions = self.risk_manager.get_positions_at_stop_loss(current_price)

        for position in triggered_positions:
            logger.warning(f"Stop loss triggered for {position.position_id}")

            order = await self.order_manager.sell(
                qty=position.qty,
                metadata={
                    "reason": "stop_loss",
                    "position_id": position.position_id,
                },
            )

            if order.state == OrderState.FILLED:
                closed = self.position_tracker.close_position(
                    position.position_id,
                    order,
                    reason="stop_loss",
                )
                self.pnl_tracker.record_realized_pnl(closed["pnl"])

                await self.alerter.send_stop_loss_alert(
                    position,
                    float(order.avg_fill_price),
                    closed["pnl"],
                )

    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions.

        Args:
            reason: Reason for closing
        """
        for position_id, position in list(self.position_tracker.open_positions.items()):
            logger.info(f"Closing position {position_id} due to {reason}")

            order = await self.order_manager.sell(
                qty=position.qty,
                metadata={"reason": reason, "position_id": position_id},
            )

            if order.state == OrderState.FILLED:
                closed = self.position_tracker.close_position(
                    position_id,
                    order,
                    reason=reason,
                )
                self.pnl_tracker.record_realized_pnl(closed["pnl"])

    async def shutdown(self) -> None:
        """Gracefully shutdown the trading system."""
        logger.info("Shutting down trading system...")
        self.running = False

        # Send shutdown alert
        await self.alerter.send_shutdown_message("graceful")

        # Log final state
        metrics = self.pnl_tracker.get_metrics()
        logger.info(f"Final equity: ${metrics['current_equity']:,.2f}")
        logger.info(f"Total return: ${metrics['total_return']:,.2f} ({metrics['total_return_pct']:.2f}%)")
        logger.info(f"Max drawdown: {metrics['max_drawdown_pct']:.2f}%")

        position_stats = self.position_tracker.get_summary()
        logger.info(f"Total trades: {position_stats['closed_positions']}")
        logger.info(f"Win rate: {position_stats['win_rate']:.1f}%")

        # Paper mode: compare to buy-and-hold (use last known price)
        if self.config.paper_trading and self._paper_start_price and self._paper_start_price > 0:
            try:
                current_price = await self.exchange.get_price(self.config.symbol)
                bh_value = float(self.config.initial_capital) / float(self._paper_start_price) * float(current_price)
                bh_return_pct = (bh_value - self.config.initial_capital) / self.config.initial_capital * 100
                logger.info(
                    f"Paper vs B&H: Strategy {metrics['total_return_pct']:.2f}% | "
                    f"Buy-and-hold same period: {bh_return_pct:.2f}%"
                )
            except Exception as e:
                logger.debug("Could not compute B&H at shutdown: %s", e)

        # Stop live price logger and WebSocket
        if self._live_price_task and not self._live_price_task.done():
            self._live_price_task.cancel()
            try:
                await self._live_price_task
            except asyncio.CancelledError:
                pass
        if self._ws_price and self._ws_price.is_running:
            await self._ws_price.stop()
            logger.info("WebSocket price stream stopped")
        if self.data_manager:
            await self.data_manager.close()
        if self.exchange:
            await self.exchange.close()

        logger.info("Trading system shutdown complete")

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        logger.info("Shutdown requested")
        self._shutdown_event.set()


def setup_signal_handlers(trader: LiveTrader) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        trader: LiveTrader instance
    """
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        trader.request_shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Bitcoin Live Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Run in paper trading mode (default)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (requires API keys)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (optional)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        help="Override initial capital",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = TradingConfig.from_env()

    # Override from arguments
    if args.live:
        config.paper_trading = False
    if args.capital:
        config.initial_capital = args.capital

    # Safety check for live trading
    if not config.paper_trading:
        print("\n" + "=" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("Real money will be at risk!")
        print("=" * 60)
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Aborted.")
            return

    # Create and run trader
    trader = LiveTrader(config)
    setup_signal_handlers(trader)

    try:
        await trader.setup()
        await trader.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
