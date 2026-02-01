"""Backtest runner for executing strategies.

Runs a strategy through historical data, tracking positions, trades, and equity.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .strategies.base import BaseStrategy, Signal
from .metrics import BacktestMetrics, Trade, compute_metrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 10000
    position_size: float = 0.25  # Fraction of capital per trade
    max_position: float = 1.0   # Max fraction of capital in BTC
    fee_pct: float = 0.1        # Trading fee (0.1% = 10 bps)
    slippage_pct: float = 0.05  # Slippage estimate
    stop_loss_pct: float | None = None     # e.g., 0.10 = 10% stop loss
    take_profit_pct: float | None = None   # e.g., 0.50 = 50% take profit
    trailing_stop_pct: float | None = None # e.g., 0.15 = 15% trailing stop
    min_trade_usd: float = 100  # Minimum trade size


class BacktestRunner:
    """Runs strategies through historical data.

    Usage:
        runner = BacktestRunner(df)
        result = runner.run(CycleSignalStrategy())
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: BacktestConfig | None = None,
    ):
        """Initialize the runner.

        Args:
            df: Historical data with 'ds' and 'y' columns.
            config: Backtest configuration. Uses defaults if None.
        """
        self.df = df.copy().sort_values("ds").reset_index(drop=True)
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: BaseStrategy,
        forecast: pd.DataFrame | None = None,
    ) -> BacktestMetrics:
        """Run a strategy through the historical data.

        Args:
            strategy: The trading strategy to test.
            forecast: Optional forecast for forecast-based strategies.

        Returns:
            BacktestMetrics with performance results.
        """
        # Generate signals
        df = strategy.generate_signals(self.df.copy(), forecast=forecast)

        # Execute trades
        trades, equity_curve = self._execute_trades(df)

        # Compute metrics
        return compute_metrics(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.config.initial_capital,
            df=df,
        )

    def _execute_trades(
        self, df: pd.DataFrame
    ) -> tuple[list[Trade], pd.DataFrame]:
        """Execute trades based on signals."""
        cfg = self.config

        capital = cfg.initial_capital
        btc_held = 0.0
        entry_price = 0.0
        highest_since_entry = 0.0
        trades: list[Trade] = []
        equity_history = []

        prev_signal = Signal.HOLD.value

        for _, row in df.iterrows():
            date = row["ds"]
            price = row["y"]
            signal = row.get("signal", Signal.HOLD.value)

            if pd.isna(price) or price <= 0:
                continue

            # Track equity
            portfolio_value = capital + (btc_held * price)
            equity_history.append({
                "ds": date,
                "equity": portfolio_value,
                "capital": capital,
                "btc_value": btc_held * price,
                "btc_held": btc_held,
                "price": price,
            })

            # Update high watermark for trailing stop
            if btc_held > 0:
                highest_since_entry = max(highest_since_entry, price)

            # Check stop loss / take profit / trailing stop
            if btc_held > 0 and entry_price > 0:
                sold = self._check_risk_exit(
                    date, price, signal, btc_held, entry_price,
                    highest_since_entry, trades
                )
                if sold:
                    sell_proceeds = sold["proceeds"]
                    capital += sell_proceeds
                    btc_held = 0.0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    continue

            # Signal-based trading
            if signal in [Signal.STRONG_BUY.value, Signal.BUY.value]:
                if prev_signal not in [Signal.STRONG_BUY.value, Signal.BUY.value]:
                    bought = self._try_buy(
                        date, price, signal, capital, btc_held,
                        portfolio_value, entry_price, trades
                    )
                    if bought:
                        capital -= bought["cost"]
                        btc_held += bought["btc"]
                        if entry_price == 0:
                            entry_price = price
                            highest_since_entry = price

            elif signal in [Signal.STRONG_SELL.value, Signal.SELL.value]:
                if prev_signal not in [Signal.STRONG_SELL.value, Signal.SELL.value]:
                    sold = self._try_sell(date, price, signal, btc_held, trades)
                    if sold:
                        capital += sold["proceeds"]
                        btc_held -= sold["btc"]
                        if btc_held < 0.0001:
                            btc_held = 0.0
                            entry_price = 0.0
                            highest_since_entry = 0.0

            prev_signal = signal

        equity_df = pd.DataFrame(equity_history)
        if equity_df.empty:
            equity_df = pd.DataFrame({"ds": [], "equity": [], "btc_held": []})

        return trades, equity_df

    def _check_risk_exit(
        self,
        date: pd.Timestamp,
        price: float,
        signal: str,
        btc_held: float,
        entry_price: float,
        highest_since_entry: float,
        trades: list[Trade],
    ) -> dict | None:
        """Check stop loss, take profit, and trailing stop."""
        cfg = self.config
        pnl_pct = (price - entry_price) / entry_price

        # Stop loss
        if cfg.stop_loss_pct and pnl_pct <= -cfg.stop_loss_pct:
            return self._execute_sell(
                date, price, signal, btc_held, trades, "stop_loss"
            )

        # Take profit
        if cfg.take_profit_pct and pnl_pct >= cfg.take_profit_pct:
            return self._execute_sell(
                date, price, signal, btc_held, trades, "take_profit"
            )

        # Trailing stop
        if cfg.trailing_stop_pct and highest_since_entry > 0:
            trailing_stop_price = highest_since_entry * (1 - cfg.trailing_stop_pct)
            if price <= trailing_stop_price:
                return self._execute_sell(
                    date, price, signal, btc_held, trades, "trailing_stop"
                )

        return None

    def _execute_sell(
        self,
        date: pd.Timestamp,
        price: float,
        signal: str,
        btc: float,
        trades: list[Trade],
        reason: str,
    ) -> dict:
        """Execute a sell and record it."""
        cfg = self.config
        gross = btc * price
        fee = gross * (cfg.fee_pct + cfg.slippage_pct) / 100
        net = gross - fee

        trades.append(Trade(
            date=date,
            action="SELL",
            price=price,
            amount_usd=gross,
            btc=btc,
            fee=fee,
            signal=signal,
            reason=reason,
        ))

        return {"proceeds": net, "btc": btc}

    def _try_buy(
        self,
        date: pd.Timestamp,
        price: float,
        signal: str,
        capital: float,
        btc_held: float,
        portfolio_value: float,
        entry_price: float,
        trades: list[Trade],
    ) -> dict | None:
        """Try to execute a buy."""
        cfg = self.config

        # Check position limit
        current_btc_value = btc_held * price
        current_position_pct = current_btc_value / portfolio_value if portfolio_value > 0 else 0

        if current_position_pct >= cfg.max_position:
            return None

        buy_amount = min(
            capital * cfg.position_size,
            portfolio_value * cfg.max_position - current_btc_value
        )

        if buy_amount < cfg.min_trade_usd:
            return None

        fee = buy_amount * (cfg.fee_pct + cfg.slippage_pct) / 100
        net_amount = buy_amount - fee
        btc_bought = net_amount / price

        trades.append(Trade(
            date=date,
            action="BUY",
            price=price,
            amount_usd=buy_amount,
            btc=btc_bought,
            fee=fee,
            signal=signal,
            reason="signal",
        ))

        return {"cost": buy_amount, "btc": btc_bought}

    def _try_sell(
        self,
        date: pd.Timestamp,
        price: float,
        signal: str,
        btc_held: float,
        trades: list[Trade],
    ) -> dict | None:
        """Try to execute a sell."""
        if btc_held <= 0:
            return None

        cfg = self.config
        sell_btc = btc_held * cfg.position_size
        gross = sell_btc * price
        fee = gross * (cfg.fee_pct + cfg.slippage_pct) / 100
        net = gross - fee

        trades.append(Trade(
            date=date,
            action="SELL",
            price=price,
            amount_usd=gross,
            btc=sell_btc,
            fee=fee,
            signal=signal,
            reason="signal",
        ))

        return {"proceeds": net, "btc": sell_btc}
