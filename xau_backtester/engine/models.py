from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(frozen=True)
class Bar:
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class Tick:
    time: pd.Timestamp
    bid: float
    ask: float
    volume: float = 0.0


@dataclass
class Order:
    id: str
    time: pd.Timestamp
    side: Side
    order_type: OrderType
    qty: float
    limit_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    tag: str = ""


@dataclass
class Fill:
    order_id: str
    time: pd.Timestamp
    side: Side
    qty: float
    price: float
    fee: float = 0.0
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None


@dataclass
class Position:
    side: Side
    qty: float
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    tag: str = ""

    def is_long(self) -> bool:
        return self.side == Side.BUY

    def is_short(self) -> bool:
        return self.side == Side.SELL

