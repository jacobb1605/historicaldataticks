from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional

from collections import deque

from xau_backtester.engine.models import Bar


class TrendBias(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class SwingPoint:
    time: object  # pd.Timestamp but keep light dependency here
    price: float
    kind: str  # "high" or "low"


class SwingDetector:
    """
    Confirms swing highs/lows without lookahead by delaying confirmation by `right` bars.

    A swing high at index k is confirmed when:
    - it is the max of the window [k-left, ..., k+right]
    - and we have observed up to k+right (i.e., right bars after it)
    """

    def __init__(self, *, left: int = 2, right: int = 2) -> None:
        if left < 1 or right < 1:
            raise ValueError("left/right must be >= 1 for non-trivial swing detection.")
        self.left = int(left)
        self.right = int(right)
        self._buf: Deque[Bar] = deque(maxlen=self.left + self.right + 1)

    def update(self, bar: Bar) -> tuple[Optional[SwingPoint], Optional[SwingPoint]]:
        self._buf.append(bar)
        if len(self._buf) < self._buf.maxlen:
            return None, None

        mid_idx = self.left
        mid_bar = self._buf[mid_idx]
        highs = [b.high for b in self._buf]
        lows = [b.low for b in self._buf]

        swing_high = None
        swing_low = None
        if mid_bar.high == max(highs):
            swing_high = SwingPoint(time=mid_bar.time, price=float(mid_bar.high), kind="high")
        if mid_bar.low == min(lows):
            swing_low = SwingPoint(time=mid_bar.time, price=float(mid_bar.low), kind="low")

        return swing_high, swing_low


@dataclass
class MarketStructureState:
    last_swing_high: Optional[SwingPoint] = None
    last_swing_low: Optional[SwingPoint] = None
    bias: TrendBias = TrendBias.NEUTRAL
    last_bos_time: Optional[object] = None


class MarketStructure:
    """
    A pragmatic ICT-style structure proxy:
    - Confirm swing highs/lows (delayed, no repaint).
    - Bias flips to BULL when close breaks above last confirmed swing high.
    - Bias flips to BEAR when close breaks below last confirmed swing low.
    """

    def __init__(self, *, swing_left: int = 2, swing_right: int = 2) -> None:
        self.swing = SwingDetector(left=swing_left, right=swing_right)
        self.state = MarketStructureState()

    def update(self, bar: Bar) -> MarketStructureState:
        sh, sl = self.swing.update(bar)
        if sh is not None:
            self.state.last_swing_high = sh
        if sl is not None:
            self.state.last_swing_low = sl

        # BOS/MSS proxy uses last confirmed levels only (no lookahead).
        if self.state.last_swing_high is not None and bar.close > self.state.last_swing_high.price:
            self.state.bias = TrendBias.BULL
            self.state.last_bos_time = bar.time
        if self.state.last_swing_low is not None and bar.close < self.state.last_swing_low.price:
            self.state.bias = TrendBias.BEAR
            self.state.last_bos_time = bar.time

        return self.state

