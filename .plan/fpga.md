# FPGA Offload Plan for Trading Bot

Ideas to improve the bot and what to offload to an FPGA for low-latency, parallel computation.

---

## 1. Order book depth (best FPGA fit)

- **Metrics:** Bid/ask depth (top 5–20 levels), order book imbalance (bid vs ask volume), weighted mid, spread.
- **Use:** Short-term direction and execution quality (where to place limits, when to cross).
- **FPGA:** Subscribe to L2 (or L3) book stream; on every update compute depth, imbalance, and weighted mid in parallel. Send only small summary (e.g. imbalance, spread, depth score) to the host so the bot can use it without handling raw book on CPU.

---

## 2. Multiple averages & timeframes

- **Metrics:** Many EMAs/SMAs (e.g. 9, 21, 50, 100, 200) and optionally RSI, ATR on 1m/5m/1h.
- **Use:** Multi-timeframe trend alignment and momentum (e.g. “all timeframes above 50 EMA”).
- **FPGA:** Compute 20–50 EMAs (and a few other indicators) in parallel on every tick or candle; stream only the indicator values (or crosses) to the host. Reduces CPU load and keeps everything time-aligned.

---

## 3. Tick-level flow & volatility

- **Metrics:** Buy vs sell volume (delta), cumulative delta, simple trade flow; realized volatility (e.g. Garman-Klass or rolling std).
- **Use:** Confirming entries (e.g. “signal + positive delta”) and dynamic position sizing or stops from vol.
- **FPGA:** Aggregate ticks into volume delta and rolling vol in real time; send compact time series (e.g. 1s or 1m buckets) to the host.

---

## 4. Execution quality (FPGA-friendly)

- **Ideas:** Book-aware limit placement (join bid/lift ask), TWAP/VWAP slice sizing, minimal crossing.
- **FPGA:** Continuous order book snapshot + simple execution rules (e.g. “target participation rate”) so the FPGA can suggest or adjust limit price/size every tick; host sends final orders.

---

## 5. What to run where

| On FPGA (latency / throughput) | On CPU (complexity / flexibility) |
|-------------------------------|-----------------------------------|
| Order book depth & imbalance | Prophet / cycle model (already there) |
| Many EMAs + a few indicators (RSI, ATR) | Signal combination & risk (position size, SL) |
| Tick aggregation (delta, vol) | Strategy logic (“if signal + depth + vol…”) |
| Book-aware execution math | Order routing, API calls, logging |

---

## 6. Concrete FPGA “calc” block to design

A single block that:

1. **Inputs:** L2 book updates (and optionally raw trades).
2. **Outputs (every tick or every 100 ms):**
   - Order book: top-N depth, imbalance, spread, weighted mid.
   - Averages: e.g. 10–20 EMAs on mid/trade price.
   - Flow: cumulative delta (or buy/sell volume).
   - Vol: short-window realized vol (e.g. 1 min).

Then the host bot only:

- Subscribes to this one stream (or a small set of streams).
- Uses these numbers in existing logic (e.g. “strong buy only if forecast + tactical + book imbalance > threshold” and “size by vol”).

---

## Next steps

- Prioritise either **execution** (order book + execution logic) or **signals** (averages + flow + vol) first.
- Define minimal FPGA spec: exact inputs/outputs and update rate.
