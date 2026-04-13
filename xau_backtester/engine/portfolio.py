from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from xau_backtester.engine.models import Fill, Position, Side


@dataclass
class ClosedTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
    qty: float
    entry_price: float
    exit_price: float
    sl_price: Optional[float]
    tp_price: Optional[float]
    pnl: float
    pnl_pct: float
    r_multiple: Optional[float]
    tag: str = ""
    exit_reason: str = ""
    initial_sl_price: Optional[float] = None
    initial_risk_per_unit: Optional[float] = None
    initial_risk_amount: Optional[float] = None


@dataclass
class Portfolio:
    initial_equity: float
    contract_multiplier: float = 1.0
    cash: float = field(init=False)
    equity: float = field(init=False)
    position: Optional[Position] = field(default=None, init=False)
    trades: list[ClosedTrade] = field(default_factory=list, init=False)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_equity)
        self.equity = float(self.initial_equity)

    def mark_to_market(self, time: pd.Timestamp, mid_price: float) -> None:
        if self.position is None:
            self.equity = float(self.cash)
        else:
            pos = self.position
            if pos.side == Side.BUY:
                unreal = (mid_price - pos.entry_price) * pos.qty * float(self.contract_multiplier)
            else:
                unreal = (pos.entry_price - mid_price) * pos.qty * float(self.contract_multiplier)
            self.equity = float(self.cash + unreal)
        self.equity_curve.append((time, float(self.equity)))

    def can_open(self) -> bool:
        return self.position is None

    def open_from_fill(self, fill: Fill, *, tag: str = "") -> None:
        if self.position is not None:
            raise RuntimeError("Position already open.")

        initial_sl_price = float(fill.sl_price) if fill.sl_price is not None else None
        initial_risk_per_unit: Optional[float] = None
        initial_risk_amount: Optional[float] = None

        if initial_sl_price is not None:
            initial_risk_per_unit = abs(float(fill.price) - initial_sl_price)
            initial_risk_amount = initial_risk_per_unit * float(fill.qty) * float(self.contract_multiplier)

        pos = Position(
            side=fill.side,
            qty=float(fill.qty),
            entry_time=fill.time,
            entry_price=float(fill.price),
            sl_price=fill.sl_price,
            tp_price=fill.tp_price,
            tag=tag,
        )

        # Attach original risk info so break-even updates do not corrupt R-multiple.
        pos.initial_sl_price = initial_sl_price
        pos.initial_risk_per_unit = initial_risk_per_unit
        pos.initial_risk_amount = initial_risk_amount

        self.position = pos

    def close_position(
        self,
        *,
        time: pd.Timestamp,
        exit_price: float,
        reason: str,
    ) -> ClosedTrade:
        if self.position is None:
            raise RuntimeError("No open position.")
        pos = self.position

        if pos.side == Side.BUY:
            pnl = (float(exit_price) - float(pos.entry_price)) * float(pos.qty) * float(self.contract_multiplier)
        else:
            pnl = (float(pos.entry_price) - float(exit_price)) * float(pos.qty) * float(self.contract_multiplier)

        self.cash = float(self.cash + pnl)
        self.equity = float(self.cash)

        # Always compute R from ORIGINAL risk captured at entry, not the current/moved SL.
        initial_sl_price = getattr(pos, "initial_sl_price", None)
        initial_risk_per_unit = getattr(pos, "initial_risk_per_unit", None)
        initial_risk_amount = getattr(pos, "initial_risk_amount", None)

        r_multiple: Optional[float] = None
        if initial_risk_amount is not None and float(initial_risk_amount) > 0:
            r_multiple = float(pnl / float(initial_risk_amount))

        trade = ClosedTrade(
            entry_time=pos.entry_time,
            exit_time=time,
            side=pos.side,
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=float(exit_price),
            sl_price=pos.sl_price,
            tp_price=pos.tp_price,
            pnl=float(pnl),
            pnl_pct=float(pnl / max(1e-9, self.initial_equity)),
            r_multiple=r_multiple,
            tag=pos.tag,
            exit_reason=reason,
            initial_sl_price=initial_sl_price,
            initial_risk_per_unit=initial_risk_per_unit,
            initial_risk_amount=initial_risk_amount,
        )
        self.trades.append(trade)
        self.position = None
        return trade

    def risk_position_size(
        self,
        *,
        risk_pct: float,
        entry_price: float,
        sl_price: float,
        contract_multiplier: float = 1.0,
        min_qty: float = 0.01,
        qty_step: float = 0.01,
    ) -> float:
        """
        Fixed-fractional sizing:
        qty = (equity * risk_pct) / (|entry - sl| * contract_multiplier)
        """
        risk_pct = float(risk_pct)
        if not (0.0 < risk_pct <= 0.05):
            raise ValueError("risk_pct should be a small fraction, e.g. 0.005 to 0.01.")

        stop_dist = abs(float(entry_price) - float(sl_price))
        if stop_dist <= 0:
            return 0.0

        risk_amount = float(self.equity * risk_pct)
        qty = risk_amount / (stop_dist * float(contract_multiplier))

        qty = max(float(min_qty), float(qty))
        qty = (qty // float(qty_step)) * float(qty_step)
        return float(qty)