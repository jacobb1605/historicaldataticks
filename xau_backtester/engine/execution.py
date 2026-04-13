from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from xau_backtester.engine.models import Bar, Fill, Order, OrderType, Side


@dataclass(frozen=True)
class ExecutionConfig:
    # Interpreted in price units (e.g., $0.10). If you use pips/points, convert beforehand.
    fixed_spread: float = 0.20
    # Slippage applied in price units to entry/exit market fills.
    slippage_mode: str = "fixed"  # "fixed" or "uniform"
    slippage: float = 0.05
    slippage_uniform_max: float = 0.10
    seed: int = 7


class ExecutionModel:
    """
    Execution model for bar-based backtests (OHLC).

    Key properties:
    - Applies spread and slippage to fills.
    - Resolves SL/TP hits with conservative intra-bar assumptions.
    """

    def __init__(self, cfg: ExecutionConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def _sample_slippage(self) -> float:
        if self.cfg.slippage_mode == "fixed":
            return float(self.cfg.slippage)
        if self.cfg.slippage_mode == "uniform":
            return float(self._rng.uniform(0.0, self.cfg.slippage_uniform_max))
        raise ValueError(f"Unknown slippage_mode={self.cfg.slippage_mode!r}")

    def _apply_spread_and_slippage(self, side: Side, mid_price: float) -> float:
        half_spread = 0.5 * float(self.cfg.fixed_spread)
        slip = self._sample_slippage()
        if side == Side.BUY:
            return mid_price + half_spread + slip
        return mid_price - half_spread - slip

    def fill_order_on_bar(self, order: Order, bar: Bar) -> Optional[Fill]:
        """
        Fill logic:
        - MARKET: filled at bar.open mid adjusted for spread/slippage.
        - LIMIT: filled if bar trades through the limit (conservative: require touch).
        """
        if order.order_type == OrderType.MARKET:
            fill_price = self._apply_spread_and_slippage(order.side, bar.open)
            return Fill(
                order_id=order.id,
                time=bar.time,
                side=order.side,
                qty=order.qty,
                price=fill_price,
                sl_price=order.sl_price,
                tp_price=order.tp_price,
            )

        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise ValueError("LIMIT order requires limit_price")
            touched = (bar.low <= order.limit_price <= bar.high)
            if not touched:
                return None
            # For a limit, assume you get filled at limit plus adverse micro-slippage.
            mid = float(order.limit_price)
            fill_price = self._apply_spread_and_slippage(order.side, mid)
            return Fill(
                order_id=order.id,
                time=bar.time,
                side=order.side,
                qty=order.qty,
                price=fill_price,
                sl_price=order.sl_price,
                tp_price=order.tp_price,
            )

        raise ValueError(f"Unknown order_type={order.order_type!r}")

    def check_exit_on_bar(
        self,
        *,
        pos_side: Side,
        sl_price: Optional[float],
        tp_price: Optional[float],
        bar: Bar,
    ) -> Optional[tuple[str, float]]:
        """
        Determine if SL/TP is hit inside this bar.

        Conservative resolution when both SL and TP are inside range:
        - For long: assume SL first if bar opens below TP? We don't know path, so take the worse outcome.
        - For short: similarly, assume the worse outcome.
        """
        if sl_price is None and tp_price is None:
            return None

        lo, hi = float(bar.low), float(bar.high)

        sl_hit = sl_price is not None and (lo <= float(sl_price) <= hi)
        tp_hit = tp_price is not None and (lo <= float(tp_price) <= hi)
        if not sl_hit and not tp_hit:
            return None

        if sl_hit and tp_hit:
            # Worst-case fill for the position.
            if pos_side == Side.BUY:
                exit_mid = float(sl_price)
                exit_side = Side.SELL
                fill = self._apply_spread_and_slippage(exit_side, exit_mid)
                return ("sl_and_tp_hit_sl_first", fill)
            else:
                exit_mid = float(sl_price)
                exit_side = Side.BUY
                fill = self._apply_spread_and_slippage(exit_side, exit_mid)
                return ("sl_and_tp_hit_sl_first", fill)

        if sl_hit:
            exit_mid = float(sl_price)
            exit_side = Side.SELL if pos_side == Side.BUY else Side.BUY
            fill = self._apply_spread_and_slippage(exit_side, exit_mid)
            return ("stop_loss", fill)

        exit_mid = float(tp_price)  # tp_hit must be True
        exit_side = Side.SELL if pos_side == Side.BUY else Side.BUY
        fill = self._apply_spread_and_slippage(exit_side, exit_mid)
        return ("take_profit", fill)

