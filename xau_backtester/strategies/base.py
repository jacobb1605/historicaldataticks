from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from xau_backtester.engine.models import Bar, Order
from xau_backtester.engine.portfolio import Portfolio


@dataclass(frozen=True)
class Strategy(ABC):
    name: str

    @abstractmethod
    def on_bar(self, *, bar: Bar, portfolio: Portfolio) -> List[Order]:
        """
        Called sequentially for each bar. Must not access future data.
        Return a list of orders to place (may be empty).
        """
        raise NotImplementedError

