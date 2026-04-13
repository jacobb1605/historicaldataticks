from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from collections import deque

from xau_backtester.engine.models import Bar
from xau_backtester.indicators.market_structure import TrendBias


class GapDirection(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class FVGZone:
    created_time: object  # pd.Timestamp
    direction: GapDirection
    lower: float
    upper: float
    filled: bool = False
    tag: str = ""  # "fvg" or "ifvg"

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def touched_by_bar(self, bar: Bar) -> bool:
        return (bar.low <= self.upper) and (bar.high >= self.lower)

    def fully_filled_by_bar(self, bar: Bar) -> bool:
        return bar.low <= self.lower and bar.high >= self.upper


class FVGDetector:
    """
    Detects 3-bar Fair Value Gaps on OHLC data with no lookahead.

    Using bars (i-2, i-1, i):
    - Bullish FVG if low[i] > high[i-2]. Zone = [high[i-2], low[i]]
    - Bearish FVG if high[i] < low[i-2]. Zone = [high[i], low[i-2]]

    Detection is only confirmed once bar i is closed (i.e., when update() receives it).
    """

    def __init__(self, *, min_gap: float = 0.05) -> None:
        self.min_gap = float(min_gap)
        self._buf: Deque[Bar] = deque(maxlen=3)

    def update(self, bar: Bar, *, bias: TrendBias = TrendBias.NEUTRAL) -> Optional[FVGZone]:
        self._buf.append(bar)
        if len(self._buf) < 3:
            return None

        b0, _, b2 = self._buf[0], self._buf[1], self._buf[2]

        # Bullish FVG
        if b2.low > b0.high:
            lower = float(b0.high)
            upper = float(b2.low)
            if (upper - lower) >= self.min_gap:
                tag = "fvg" if bias in (TrendBias.NEUTRAL, TrendBias.BULL) else "ifvg"
                return FVGZone(
                    created_time=b2.time,
                    direction=GapDirection.BULLISH,
                    lower=lower,
                    upper=upper,
                    tag=tag,
                )

        # Bearish FVG
        if b2.high < b0.low:
            lower = float(b2.high)
            upper = float(b0.low)
            if (upper - lower) >= self.min_gap:
                tag = "fvg" if bias in (TrendBias.NEUTRAL, TrendBias.BEAR) else "ifvg"
                return FVGZone(
                    created_time=b2.time,
                    direction=GapDirection.BEARISH,
                    lower=lower,
                    upper=upper,
                    tag=tag,
                )

        return None

