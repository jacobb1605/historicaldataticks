from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from xau_backtester.engine.models import Bar


@dataclass
class BarDataFeed:
    bars: pd.DataFrame

    def __post_init__(self) -> None:
        if self.bars is None or self.bars.empty:
            raise ValueError("bars is empty")

        required = ["time", "open", "high", "low", "close"]
        missing = [c for c in required if c not in self.bars.columns]
        if missing:
            raise ValueError(f"bars missing required columns: {missing}")

        df = self.bars.sort_values("time", kind="mergesort").reset_index(drop=True).copy()

        numeric_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)

        if "volume" not in df.columns:
            df["volume"] = 0.0
        else:
            df["volume"] = df["volume"].fillna(0.0)

        self._rows = list(
            zip(
                df["time"].tolist(),
                df["open"].astype(float).tolist(),
                df["high"].astype(float).tolist(),
                df["low"].astype(float).tolist(),
                df["close"].astype(float).tolist(),
                df["volume"].astype(float).tolist(),
            )
        )
        self._idx = 0

    def __iter__(self) -> Iterator[Bar]:
        self._idx = 0
        return self

    def __next__(self) -> Bar:
        if self._idx >= len(self._rows):
            raise StopIteration

        time, open_, high, low, close, volume = self._rows[self._idx]
        self._idx += 1

        return Bar(
            time=time,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )