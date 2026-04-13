from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional
from collections import deque

from xau_backtester.engine.models import Bar, Order, OrderType, Side
from xau_backtester.engine.portfolio import Portfolio
from xau_backtester.filters.sessions import SessionFilter
from xau_backtester.indicators.fibonacci import (
    compute_retracement_from_high,
    compute_retracement_from_low,
    in_range,
)
from xau_backtester.indicators.fvg import FVGDetector, FVGZone, GapDirection
from xau_backtester.indicators.market_structure import MarketStructure, TrendBias
from xau_backtester.strategies.base import Strategy


@dataclass(frozen=True)
class ICTFVGConfig:
    swing_left: int = 2
    swing_right: int = 2

    min_gap: float = 0.25
    max_active_zones: int = 20

    fib_lower: float = 0.50
    fib_upper: float = 0.65
    tp_extension: float = -1.0

    entry_at: str = "mid"  # "mid" or "edge"

    risk_pct: float = 0.005
    stop_buffer: float = 0.20

    allowed_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    session_start_hour_utc: int = 5
    session_end_hour_utc: int = 9

    contract_multiplier: float = 1.0
    min_qty: float = 0.01
    qty_step: float = 0.01

    max_zone_age_bars: int = 10
    min_impulse_size: float = 2.5
    one_trade_per_zone: bool = True

    strong_impulse_size: float = 4.0
    strong_only: bool = False
    require_rejection_for_non_strong: bool = True
    require_close_back_outside_zone: bool = True
    require_directional_close: bool = True
    require_rejection_beyond_zone_mid: bool = True


class ICTFVGStrategy(Strategy):
    """
    London-focused ICT/FVG/Fib strategy with hybrid entries.

    Strong setup:
      enter on first touch with a limit order.

    Weaker setup:
      require rejection confirmation before allowing the limit order.
    """

    def __init__(self, *, cfg: ICTFVGConfig) -> None:
        super().__init__(name="ICT_FVG_FIBO")
        self.cfg = cfg
        self.ms = MarketStructure(swing_left=cfg.swing_left, swing_right=cfg.swing_right)
        self.fvg = FVGDetector(min_gap=cfg.min_gap)

        self.active_zones: Deque[FVGZone] = deque(maxlen=cfg.max_active_zones)
        self.last_bar: Optional[Bar] = None
        self.bar_index: int = 0

        self.session = SessionFilter(
            allowed_weekdays=cfg.allowed_weekdays,
            start_hour_utc=cfg.session_start_hour_utc,
            end_hour_utc=cfg.session_end_hour_utc,
        )

        self.zone_birth_bar: dict[str, int] = {}
        self.traded_zone_keys: set[str] = set()

        # Daily trade control
        self.current_day = None
        self.trades_today = 0
        self.losses_today = 0
        self.last_counted_trade_key = None

    def _infer_impulse_swings(self) -> Optional[tuple[float, float]]:
        st = self.ms.state
        if st.last_swing_low is None or st.last_swing_high is None:
            return None
        return float(st.last_swing_low.price), float(st.last_swing_high.price)

    def _zone_entry_price(self, z: FVGZone) -> float:
        if self.cfg.entry_at == "edge":
            return float(z.upper) if z.direction == GapDirection.BULLISH else float(z.lower)
        return float((z.lower + z.upper) / 2.0)

    def _zone_mid(self, z: FVGZone) -> float:
        return float((z.lower + z.upper) / 2.0)

    def _zone_width(self, z: FVGZone) -> float:
        return abs(float(z.upper) - float(z.lower))

    def _zone_key(self, z: FVGZone) -> str:
        return f"{z.direction}|{round(float(z.lower), 6)}|{round(float(z.upper), 6)}|{z.tag}"

    def _register_zone(self, z: FVGZone) -> None:
        key = self._zone_key(z)
        if key not in self.zone_birth_bar:
            self.zone_birth_bar[key] = self.bar_index

    def _zone_age_bars(self, z: FVGZone) -> int:
        key = self._zone_key(z)
        born = self.zone_birth_bar.get(key, self.bar_index)
        return self.bar_index - born

    def _zone_is_fresh(self, z: FVGZone) -> bool:
        return self._zone_age_bars(z) <= self.cfg.max_zone_age_bars

    def _zone_already_traded(self, z: FVGZone) -> bool:
        return self._zone_key(z) in self.traded_zone_keys

    def _mark_zone_traded(self, z: FVGZone) -> None:
        self.traded_zone_keys.add(self._zone_key(z))

    def _prune_zone_state(self) -> None:
        live_keys = {self._zone_key(z) for z in self.active_zones}

        stale_birth_keys = [k for k in self.zone_birth_bar if k not in live_keys]
        for k in stale_birth_keys:
            del self.zone_birth_bar[k]

        self.traded_zone_keys.intersection_update(live_keys)

    def _bar_is_bullish(self, bar: Bar) -> bool:
        return float(bar.close) > float(bar.open)

    def _bar_is_bearish(self, bar: Bar) -> bool:
        return float(bar.close) < float(bar.open)

    def _passes_directional_close(self, bar: Bar, *, bias: TrendBias) -> bool:
        if not self.cfg.require_directional_close:
            return True
        if bias == TrendBias.BULL:
            return self._bar_is_bullish(bar)
        if bias == TrendBias.BEAR:
            return self._bar_is_bearish(bar)
        return False

    def _is_strong_setup(self, impulse_size: float, *, z: FVGZone) -> bool:
        if impulse_size < float(self.cfg.strong_impulse_size):
            return False
        if not self._zone_is_fresh(z):
            return False
        return True

    def _bullish_rejection_confirmed(self, *, bar: Bar, z: FVGZone) -> bool:
        if not z.touched_by_bar(bar):
            return False
        if not self._bar_is_bullish(bar):
            return False

        zone_mid = self._zone_mid(z)

        if self.cfg.require_rejection_beyond_zone_mid and float(bar.close) < zone_mid:
            return False

        if self.cfg.require_close_back_outside_zone and float(bar.close) < float(z.upper):
            return False

        return True

    def _bearish_rejection_confirmed(self, *, bar: Bar, z: FVGZone) -> bool:
        if not z.touched_by_bar(bar):
            return False
        if not self._bar_is_bearish(bar):
            return False

        zone_mid = self._zone_mid(z)

        if self.cfg.require_rejection_beyond_zone_mid and float(bar.close) > zone_mid:
            return False

        if self.cfg.require_close_back_outside_zone and float(bar.close) > float(z.lower):
            return False

        return True

    def _bullish_entry_allowed(self, *, bar: Bar, z: FVGZone, impulse_size: float) -> bool:
        if not z.touched_by_bar(bar):
            return False

        if self._is_strong_setup(impulse_size, z=z):
            return True

        if self.cfg.strong_only:
            return False

        if not self.cfg.require_rejection_for_non_strong:
            return True

        return self._bullish_rejection_confirmed(bar=bar, z=z)

    def _bearish_entry_allowed(self, *, bar: Bar, z: FVGZone, impulse_size: float) -> bool:
        if not z.touched_by_bar(bar):
            return False

        if self._is_strong_setup(impulse_size, z=z):
            return True

        if self.cfg.strong_only:
            return False

        if not self.cfg.require_rejection_for_non_strong:
            return True

        return self._bearish_rejection_confirmed(bar=bar, z=z)

    def _setup_type(self, *, impulse_size: float, z: FVGZone) -> str:
        return "strong_first_touch" if self._is_strong_setup(impulse_size, z=z) else "weak_rejection"

    def _build_order_tag(
        self,
        *,
        side_label: str,
        z: FVGZone,
        impulse_size: float,
        entry: float,
        sl: float,
        tp: float,
    ) -> str:
        setup = self._setup_type(impulse_size=impulse_size, z=z)
        zone_age = self._zone_age_bars(z)
        zone_width = self._zone_width(z)
        risk_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)

        return (
            f"side={side_label}|"
            f"setup={setup}|"
            f"entry_at={self.cfg.entry_at}|"
            f"impulse={impulse_size:.2f}|"
            f"zone_age={zone_age}|"
            f"zone_width={zone_width:.2f}|"
            f"risk={risk_distance:.2f}|"
            f"reward={reward_distance:.2f}|"
            f"tp_ext={self.cfg.tp_extension}"
        )

    def on_bar(self, *, bar: Bar, portfolio: Portfolio) -> List[Order]:
        self.last_bar = bar
        self.bar_index += 1

        # Reset daily counters on new day
        bar_day = bar.time.date()
        if self.current_day != bar_day:
            self.current_day = bar_day
            self.trades_today = 0
            self.losses_today = 0
            self.last_counted_trade_key = None

        # Count losses from closed trades once per trade
        if portfolio.trades:
            last_trade = portfolio.trades[-1]
            trade_key = (last_trade.entry_time, last_trade.exit_time, last_trade.pnl)
            if last_trade.exit_time.date() == self.current_day and last_trade.pnl < 0:
                if self.last_counted_trade_key != trade_key:
                    self.losses_today += 1
                    self.last_counted_trade_key = trade_key

        st = self.ms.update(bar)
        new_zone = self.fvg.update(bar, bias=st.bias)
        if new_zone is not None:
            self.active_zones.appendleft(new_zone)
            self._register_zone(new_zone)

        self._age_zones(bar)
        self._prune_zone_state()

        if not self.session.allows(bar.time):
            return []

        # Max trades per day = 3
        if self.trades_today >= 3:
            return []

        # Stop after 2 losses in a day
        if self.losses_today >= 3:
            return []

        if portfolio.position is not None:
            return []
        if not portfolio.can_open():
            return []
        if st.bias == TrendBias.NEUTRAL:
            return []
        if not self._passes_directional_close(bar, bias=st.bias):
            return []

        impulse = self._infer_impulse_swings()
        if impulse is None:
            return []

        swing_low, swing_high = impulse
        if float(swing_high) <= float(swing_low):
            return []

        impulse_size = float(swing_high) - float(swing_low)
        if impulse_size < float(self.cfg.min_impulse_size):
            return []

        orders: List[Order] = []

        if st.bias == TrendBias.BULL:
            fib = compute_retracement_from_high(
                swing_low=swing_low,
                swing_high=swing_high,
                ratios=[self.cfg.fib_lower, self.cfg.fib_upper, self.cfg.tp_extension],
            )
            if fib is None:
                return []

            fib_zone_low = min(fib.levels[self.cfg.fib_lower], fib.levels[self.cfg.fib_upper])
            fib_zone_high = max(fib.levels[self.cfg.fib_lower], fib.levels[self.cfg.fib_upper])
            tp = float(fib.levels[self.cfg.tp_extension])

            for z in list(self.active_zones):
                if z.filled:
                    continue
                if z.tag != "fvg":
                    continue
                if z.direction != GapDirection.BULLISH:
                    continue
                if not self._zone_is_fresh(z):
                    continue
                if self.cfg.one_trade_per_zone and self._zone_already_traded(z):
                    continue
                if not self._bullish_entry_allowed(bar=bar, z=z, impulse_size=impulse_size):
                    continue

                entry = self._zone_entry_price(z)

                if not in_range(entry, lower=fib_zone_low, upper=fib_zone_high):
                    continue
                if st.last_swing_low is None:
                    continue

                sl = float(st.last_swing_low.price) - float(self.cfg.stop_buffer)

                if sl >= entry:
                    continue
                if tp <= entry:
                    continue

                qty = portfolio.risk_position_size(
                    risk_pct=self.cfg.risk_pct,
                    entry_price=entry,
                    sl_price=sl,
                    contract_multiplier=self.cfg.contract_multiplier,
                    min_qty=self.cfg.min_qty,
                    qty_step=self.cfg.qty_step,
                )
                if qty <= 0:
                    continue

                orders.append(
                    Order(
                        id=f"entry_{bar.time.value}",
                        time=bar.time,
                        side=Side.BUY,
                        order_type=OrderType.LIMIT,
                        qty=qty,
                        limit_price=entry,
                        sl_price=sl,
                        tp_price=tp,
                        tag=self._build_order_tag(
                            side_label="bull",
                            z=z,
                            impulse_size=impulse_size,
                            entry=entry,
                            sl=sl,
                            tp=tp,
                        ),
                    )
                )

                self.trades_today += 1
                self._mark_zone_traded(z)
                z.filled = True
                break

        elif st.bias == TrendBias.BEAR:
            fib = compute_retracement_from_low(
                swing_low=swing_low,
                swing_high=swing_high,
                ratios=[self.cfg.fib_lower, self.cfg.fib_upper, self.cfg.tp_extension],
            )
            if fib is None:
                return []

            fib_zone_low = min(fib.levels[self.cfg.fib_lower], fib.levels[self.cfg.fib_upper])
            fib_zone_high = max(fib.levels[self.cfg.fib_lower], fib.levels[self.cfg.fib_upper])
            tp = float(fib.levels[self.cfg.tp_extension])

            for z in list(self.active_zones):
                if z.filled:
                    continue
                if z.tag != "fvg":
                    continue
                if z.direction != GapDirection.BEARISH:
                    continue
                if not self._zone_is_fresh(z):
                    continue
                if self.cfg.one_trade_per_zone and self._zone_already_traded(z):
                    continue
                if not self._bearish_entry_allowed(bar=bar, z=z, impulse_size=impulse_size):
                    continue

                entry = self._zone_entry_price(z)

                if not in_range(entry, lower=fib_zone_low, upper=fib_zone_high):
                    continue
                if st.last_swing_high is None:
                    continue

                sl = float(st.last_swing_high.price) + float(self.cfg.stop_buffer)

                if sl <= entry:
                    continue
                if tp >= entry:
                    continue

                qty = portfolio.risk_position_size(
                    risk_pct=self.cfg.risk_pct,
                    entry_price=entry,
                    sl_price=sl,
                    contract_multiplier=self.cfg.contract_multiplier,
                    min_qty=self.cfg.min_qty,
                    qty_step=self.cfg.qty_step,
                )
                if qty <= 0:
                    continue

                orders.append(
                    Order(
                        id=f"entry_{bar.time.value}",
                        time=bar.time,
                        side=Side.SELL,
                        order_type=OrderType.LIMIT,
                        qty=qty,
                        limit_price=entry,
                        sl_price=sl,
                        tp_price=tp,
                        tag=self._build_order_tag(
                            side_label="bear",
                            z=z,
                            impulse_size=impulse_size,
                            entry=entry,
                            sl=sl,
                            tp=tp,
                        ),
                    )
                )

                self.trades_today += 1
                self._mark_zone_traded(z)
                z.filled = True
                break

        return orders

    def _age_zones(self, bar: Bar) -> None:
        for z in self.active_zones:
            if z.filled:
                continue

            if z.fully_filled_by_bar(bar):
                z.filled = True
                continue

            if not self._zone_is_fresh(z):
                z.filled = True