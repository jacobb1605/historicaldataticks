from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    total_return: float
    cagr: Optional[float]
    max_drawdown: float
    sharpe_like: Optional[float]
    trades: int
    win_rate: Optional[float]
    avg_r: Optional[float]
    profit_factor: Optional[float]


def _max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0

    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if eq.empty:
        return 0.0

    peak = eq.cummax()
    valid = peak > 0
    if not valid.any():
        return 0.0

    dd = pd.Series(np.nan, index=eq.index, dtype=float)
    dd.loc[valid] = (eq.loc[valid] / peak.loc[valid]) - 1.0

    min_dd = dd.min(skipna=True)
    if pd.isna(min_dd):
        return 0.0
    return float(min_dd)


def _cagr(start_equity: float, end_equity: float, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    try:
        start_equity = float(start_equity)
        end_equity = float(end_equity)
    except Exception:
        return 0.0

    if not math.isfinite(start_equity) or not math.isfinite(end_equity):
        return 0.0

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    seconds = (end - start).total_seconds()
    if seconds <= 0:
        return 0.0

    years = seconds / (365.25 * 24 * 3600)
    if years <= 0:
        return 0.0

    if start_equity <= 0 or end_equity <= 0:
        return 0.0

    ratio = end_equity / start_equity
    if ratio <= 0 or not math.isfinite(ratio):
        return 0.0

    try:
        value = ratio ** (1.0 / years) - 1.0
    except Exception:
        return 0.0

    if not math.isfinite(value):
        return 0.0
    return float(value)


def _sharpe_like(equity: pd.Series) -> Optional[float]:
    if equity is None or len(equity) < 3:
        return None

    eq = pd.to_numeric(equity, errors="coerce").dropna()
    if len(eq) < 3:
        return None

    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(rets) < 2:
        return None

    std = rets.std(ddof=1)
    if pd.isna(std) or std == 0 or not np.isfinite(std):
        return None

    mean = rets.mean()
    if pd.isna(mean) or not np.isfinite(mean):
        return None

    value = mean / std * np.sqrt(len(rets))
    if not np.isfinite(value):
        return None
    return float(value)


def _numeric_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    return s


def _compute_avg_r(trades: pd.DataFrame) -> Optional[float]:
    if trades is None or trades.empty:
        return None

    pnl = _numeric_series(trades, "pnl")

    # Best case: compute R directly from pnl / initial_risk.
    for risk_col in ("initial_risk", "risk_amount", "risk_amt", "risk_dollars", "risk"):
        risk = _numeric_series(trades, risk_col)
        if pnl is not None and risk is not None:
            valid = risk > 0
            if valid.any():
                r = (pnl[valid] / risk[valid]).replace([np.inf, -np.inf], np.nan).dropna()
                if len(r) > 0:
                    avg = r.mean()
                    return float(avg) if np.isfinite(avg) else None

    # Fallback: use provided r_multiple, but sanity-check sign against pnl.
    r = _numeric_series(trades, "r_multiple")
    if r is not None and len(r.dropna()) > 0:
        r = r.replace([np.inf, -np.inf], np.nan).dropna()

        if pnl is not None:
            pnl_aligned = pnl.loc[r.index].replace([np.inf, -np.inf], np.nan).dropna()
            common_idx = r.index.intersection(pnl_aligned.index)

            if len(common_idx) > 0:
                r_cmp = r.loc[common_idx]
                pnl_cmp = pnl_aligned.loc[common_idx]

                # If most signs disagree, stored r_multiple is probably inverted.
                sign_match_ratio = float((np.sign(r_cmp) == np.sign(pnl_cmp)).mean())
                if sign_match_ratio < 0.5:
                    r = -r

        avg = r.mean()
        return float(avg) if np.isfinite(avg) else None

    return None


def compute_metrics(*, equity_curve: pd.DataFrame, trades: pd.DataFrame) -> PerformanceMetrics:
    if equity_curve is None or equity_curve.empty:
        raise ValueError("equity_curve is empty")

    if "time" not in equity_curve.columns or "equity" not in equity_curve.columns:
        raise ValueError("equity_curve must contain 'time' and 'equity' columns")

    equity_curve = equity_curve.sort_values("time").copy()

    eq = pd.to_numeric(equity_curve["equity"], errors="coerce").dropna().reset_index(drop=True)
    if eq.empty:
        raise ValueError("equity_curve has no valid numeric equity values")

    start_t = pd.Timestamp(equity_curve["time"].iloc[0])
    end_t = pd.Timestamp(equity_curve["time"].iloc[-1])

    start_eq = float(eq.iloc[0])
    end_eq = float(eq.iloc[-1])

    if start_eq > 0 and math.isfinite(start_eq) and math.isfinite(end_eq):
        total_return = float(end_eq / start_eq - 1.0)
        if not math.isfinite(total_return):
            total_return = 0.0
    else:
        total_return = 0.0

    cagr = _cagr(start_eq, end_eq, start_t, end_t)
    mdd = _max_drawdown(eq)
    sharpe = _sharpe_like(eq)

    n_trades = int(len(trades)) if trades is not None else 0
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    avg_r: Optional[float] = None

    if trades is not None and not trades.empty:
        pnl = _numeric_series(trades, "pnl")

        if pnl is not None:
            pnl = pnl.dropna()

            if len(pnl) > 0:
                win_rate = float((pnl > 0).mean())

                gross_win = float(pnl[pnl > 0].sum())
                gross_loss = float(-pnl[pnl < 0].sum())

                if gross_loss > 0:
                    pf = gross_win / gross_loss
                    profit_factor = float(pf) if math.isfinite(pf) else None
                elif gross_win > 0:
                    profit_factor = float("inf")

        avg_r = _compute_avg_r(trades)

    return PerformanceMetrics(
        start=start_t,
        end=end_t,
        bars=int(len(eq)),
        total_return=total_return,
        cagr=cagr,
        max_drawdown=mdd,
        sharpe_like=sharpe,
        trades=n_trades,
        win_rate=win_rate,
        avg_r=avg_r,
        profit_factor=profit_factor,
    )