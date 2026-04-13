from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from xau_backtester.engine.data_feed import BarDataFeed
from xau_backtester.engine.execution import ExecutionModel
from xau_backtester.engine.models import Bar, Fill, Order, Side
from xau_backtester.engine.portfolio import Portfolio
from xau_backtester.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame


class Backtester:
    def __init__(
        self,
        *,
        feed: BarDataFeed,
        strategy: Strategy,
        execution: ExecutionModel,
        portfolio: Portfolio,
        break_even_enabled: bool = True,
        break_even_trigger_tp_frac: float = 0.7125,
        break_even_offset: float = 0.05,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.execution = execution
        self.portfolio = portfolio

        self.break_even_enabled = bool(break_even_enabled)
        self.break_even_trigger_tp_frac = float(break_even_trigger_tp_frac)
        self.break_even_offset = float(break_even_offset)

        self._pending_orders: list[Order] = []

    def _mid_from_bar(self, bar: Bar) -> float:
        return float(bar.close)

    def _process_pending_orders(self, bar: Bar) -> list[Fill]:
        fills: list[Fill] = []
        if not self._pending_orders:
            return fills

        still_pending: list[Order] = []
        for o in self._pending_orders:
            fill = self.execution.fill_order_on_bar(o, bar)
            if fill is None:
                still_pending.append(o)
                continue
            fills.append(fill)

        self._pending_orders = still_pending
        return fills

    def _maybe_move_stop_to_break_even(self, bar: Bar) -> None:
        if not self.break_even_enabled:
            return

        pos = self.portfolio.position
        if pos is None:
            return

        entry = float(pos.entry_price)
        tp = float(pos.tp_price)
        sl = float(pos.sl_price)
        close = float(bar.close)

        if pos.side == Side.BUY:
            denom = tp - entry
            if denom <= 0:
                return

            progress = (close - entry) / denom
            if progress >= self.break_even_trigger_tp_frac:
                new_sl = entry + self.break_even_offset
                if new_sl > sl:
                    pos.sl_price = new_sl

        elif pos.side == Side.SELL:
            denom = entry - tp
            if denom <= 0:
                return

            progress = (entry - close) / denom
            if progress >= self.break_even_trigger_tp_frac:
                new_sl = entry - self.break_even_offset
                if new_sl < sl:
                    pos.sl_price = new_sl

    def run(self) -> BacktestResult:
        for bar in self.feed:
            mid = self._mid_from_bar(bar)

            # mark-to-market
            self.portfolio.mark_to_market(bar.time, mid_price=mid)

            # manage open position
            if self.portfolio.position is not None:
                self._maybe_move_stop_to_break_even(bar)

                pos = self.portfolio.position
                exit_hit = self.execution.check_exit_on_bar(
                    pos_side=pos.side,
                    sl_price=pos.sl_price,
                    tp_price=pos.tp_price,
                    bar=bar,
                )
                if exit_hit is not None:
                    reason, exit_price = exit_hit
                    self.portfolio.close_position(
                        time=bar.time,
                        exit_price=exit_price,
                        reason=reason,
                    )

            # fill pending orders
            fills = self._process_pending_orders(bar)
            for fill in fills:
                if self.portfolio.position is None:
                    self.portfolio.open_from_fill(fill, tag=self.strategy.name)

            # generate new orders
            orders = self.strategy.on_bar(bar=bar, portfolio=self.portfolio)
            for o in orders:
                self._pending_orders.append(o)

        equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=["time", "equity"])
        trades_df = pd.DataFrame([t.__dict__ for t in self.portfolio.trades])
        return BacktestResult(equity_curve=equity_df, trades=trades_df)