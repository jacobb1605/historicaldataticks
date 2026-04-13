from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class FibLevels:
    swing_low: float
    swing_high: float
    levels: dict[float, float]


def compute_retracement_from_high(
    *,
    swing_low: float,
    swing_high: float,
    ratios: Iterable[float],
) -> Optional[FibLevels]:
    """
    Bullish impulse: swing_low -> swing_high.

    Retracement levels are measured from the swing high back down.
    Negative ratios naturally become extension targets above the swing high.
    """
    lo = float(swing_low)
    hi = float(swing_high)

    if hi <= lo:
        return None

    span = hi - lo
    levels: dict[float, float] = {}

    for r in (float(x) for x in ratios):
        levels[r] = hi - span * r

    return FibLevels(swing_low=lo, swing_high=hi, levels=levels)


def compute_retracement_from_low(
    *,
    swing_low: float,
    swing_high: float,
    ratios: Iterable[float],
) -> Optional[FibLevels]:
    """
    Bearish impulse: swing_high -> swing_low.

    Retracement levels are measured from the swing low back up.
    Negative ratios naturally become extension targets below the swing low.
    """
    lo = float(swing_low)
    hi = float(swing_high)

    if hi <= lo:
        return None

    span = hi - lo
    levels: dict[float, float] = {}

    for r in (float(x) for x in ratios):
        levels[r] = lo + span * r

    return FibLevels(swing_low=lo, swing_high=hi, levels=levels)


def in_range(price: float, *, lower: float, upper: float) -> bool:
    p = float(price)
    lo = float(lower)
    hi = float(upper)
    return min(lo, hi) <= p <= max(lo, hi)