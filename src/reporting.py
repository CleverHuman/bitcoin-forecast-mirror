"""Text report formatting for signals and backtest results."""


def print_signal_summary(current: dict, backtest: dict) -> None:
    """Print summary of current signal and backtest results."""
    print("\n" + "=" * 60)
    print("CURRENT SIGNAL")
    print("=" * 60)
    print(f"  Date: {current.get('date')}")
    print(f"  Price: ${current.get('price'):,.0f}" if current.get('price') else "  Price: N/A")
    print(f"  Signal: {current.get('signal', 'N/A').upper()}")
    print(f"  Score: {current.get('signal_score', 0):.2f}")
    print(f"  Cycle Phase: {current.get('cycle_phase', 'N/A')}")

    days_until = current.get("days_until_halving")
    days_since = current.get("days_since_halving")
    if days_until:
        print(f"  Days Until Halving: {int(days_until)}")
    if days_since:
        print(f"  Days Since Halving: {int(days_since)}")

    if current.get("rsi"):
        print(f"  RSI: {current['rsi']:.1f}")

    # Buy timing guidance
    buy_timing = current.get("buy_timing")
    if buy_timing:
        print("\n" + "-" * 60)
        print("BUY TIMING GUIDANCE")
        print("-" * 60)
        print(f"  >>> {buy_timing['action']} <<<")
        print(f"  {buy_timing['reason']}")
        print()
        print(f"  Last cycle top: ${buy_timing['last_top_price']:,.0f} on {buy_timing['last_top_date'].strftime('%Y-%m-%d')}")
        print()
        print("  Predicted bottom (decay-adjusted):")
        low, high = buy_timing['decay_bottom_range']
        print(f"    Price range:  ${low:,.0f} - ${high:,.0f}")
        print(f"    Best estimate: ${buy_timing['decay_bottom_price']:,.0f}")
        print(f"    Expected drawdown: {buy_timing['decay_drawdown_pct']:.1f}%")
        print()
        print(f"  Current price vs predicted bottom: {buy_timing['price_vs_decay_bottom_pct']:+.1f}%")
        if buy_timing['days_to_predicted_bottom'] > 0:
            print(f"  Days to predicted bottom: ~{buy_timing['days_to_predicted_bottom']}")

    print(f"\n  Recommendation: {current.get('recommendation', 'N/A')}")

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Signal Strategy vs Buy & Hold)")
    print("=" * 60)
    print(f"  Initial Capital: ${backtest['initial_capital']:,.0f}")
    print(f"  Signal Strategy Final: ${backtest['final_value']:,.0f} ({backtest['total_return_pct']:+.1f}%)")
    print(f"  Buy & Hold Final: ${backtest['buy_hold_value']:,.0f} ({backtest['buy_hold_return_pct']:+.1f}%)")
    print(f"  Outperformance: {backtest['outperformance_pct']:+.1f}%")
    print(f"  Number of Trades: {backtest['num_trades']}")
    print("=" * 60 + "\n")
