from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResampleConfig:
    rule: str = "1min"
    label: str = "left"
    closed: str = "left"


def ticks_to_ohlcv(
    ticks: pd.DataFrame,
    *,
    cfg: ResampleConfig = ResampleConfig(),
    price_col: str = "mid",
) -> pd.DataFrame:
    """
    Convert tick data (bid/ask) into OHLCV bars.

    No-lookahead note:
    - This is a forward-only aggregation. If you later backtest on the resulting bars,
      your engine must still process bars sequentially.
    """
    if "time" not in ticks.columns:
        raise ValueError("ticks_to_ohlcv requires a 'time' column.")
    if "bid" not in ticks.columns or "ask" not in ticks.columns:
        raise ValueError("ticks_to_ohlcv requires 'bid' and 'ask' columns.")

    df = ticks.copy()
    df = df.sort_values("time", kind="mergesort")
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        raise ValueError("'time' must be datetime64 (timezone-aware preferred).")

    df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
    if price_col not in df.columns:
        raise ValueError(f"Unknown price_col={price_col!r}. Available: {list(df.columns)}")

    df = df.set_index("time", drop=True)

    ohlc = df[price_col].resample(cfg.rule, label=cfg.label, closed=cfg.closed).ohlc()
    out = ohlc.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})

    if "volume" in df.columns:
        vol = df["volume"].resample(cfg.rule, label=cfg.label, closed=cfg.closed).sum()
        out["volume"] = vol
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index().rename(columns={"time": "time"})
    return out

