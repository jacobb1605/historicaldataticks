# Strategy rules (ICT structure + FVG + Fib)

This project implements a pragmatic, rule-based version of common ICT-style concepts on **1-minute OHLCV** bars (or tick data resampled to M1).

## Market structure (bias)

Implemented in `xau_backtester/indicators/market_structure.py`.

- **Swing highs/lows** are confirmed using a window of `left + right + 1` bars.\n
  - A swing is only confirmed after `right` bars have elapsed (prevents lookahead/repainting).
- **Bias** is derived from breaks of the **last confirmed** swing levels:\n
  - Bias becomes **bullish** when `close > last_swing_high`.\n
  - Bias becomes **bearish** when `close < last_swing_low`.

This is a simplified BOS/MSS proxy suitable for systematic backtesting.

## FVG (Fair Value Gap)

Implemented in `xau_backtester/indicators/fvg.py`.

Using a 3-bar pattern (bars `i-2, i-1, i`) and confirming only when bar `i` is closed:

- **Bullish FVG** if `low[i] > high[i-2]`.\n
  - Zone is `[high[i-2], low[i]]`.
- **Bearish FVG** if `high[i] < low[i-2]`.\n
  - Zone is `[high[i], low[i-2]]`.

Gaps below `min_gap` are ignored.

## IFVG (Inverse FVG)

In this engine, a detected FVG is tagged as **IFVG** when it forms **against** the current market structure bias.\n
This gives you a clean separation between:

- **FVG**: aligned-with-bias gaps (default entry candidates)
- **IFVG**: counter-bias gaps (tracked for research/targets; not used for entry by default)

Tagging happens at detection time in `FVGDetector.update(...)`.

## Fibonacci retracement confluence

Implemented in `xau_backtester/indicators/fibonacci.py`.

- For **bullish bias** (impulse low→high), retracement is measured **from the swing high back down**.\n
- For **bearish bias** (impulse high→low), retracement is measured **from the swing low back up**.\n

The strategy requires the planned entry price to lie within the configured Fib band (e.g. 61.8%–79%).

## Entry, SL, TP

Implemented in `xau_backtester/strategies/ict_fvg_fibo.py`.

- **Session filter** (UTC): only place orders between `session_start_hour_utc` and `session_end_hour_utc`.\n
  (State is still updated outside the session so the model remains consistent.)
- **Entry trigger**:\n
  - Bias must be bullish for longs / bearish for shorts.\n
  - Price must *touch* an active, bias-aligned FVG zone.\n
  - Entry price must also satisfy Fib confluence.\n
  - Place a **LIMIT** order at the zone mid (or edge if configured).
- **Stop-loss**:\n
  - Long: `last_swing_low - stop_buffer`.\n
  - Short: `last_swing_high + stop_buffer`.
- **Take-profit**:\n
  - Fixed **R-multiple**, `rr_target` (e.g. 2.5R).

## No lookahead bias

The no-lookahead property comes from three design choices:

- Swings are confirmed only after `right` bars.\n
- FVGs are confirmed only after the third bar closes.\n
- The engine calls `Strategy.on_bar(...)` sequentially with only the current bar and the strategy’s own incremental state.

