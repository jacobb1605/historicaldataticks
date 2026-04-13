from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from xau_backtester.analytics.metrics import compute_metrics
from xau_backtester.engine.backtester import Backtester
from xau_backtester.engine.data_feed import BarDataFeed
from xau_backtester.engine.execution import ExecutionConfig, ExecutionModel
from xau_backtester.engine.portfolio import Portfolio
from xau_backtester.strategies.ict_fvg_fibo import ICTFVGConfig, ICTFVGStrategy


@dataclass(frozen=True)
class OptimizationResult:
    results: pd.DataFrame
    best_params: Dict[str, Any]


def _score(metrics) -> float:
    # Simple objective: reward return, penalize drawdown. Tune as needed.
    return float(metrics.total_return - 0.5 * abs(metrics.max_drawdown))


def optimize_grid(
    *,
    bars: pd.DataFrame,
    base_cfg: ICTFVGConfig,
    grid: Dict[str, Iterable[Any]],
    execution_cfg: ExecutionConfig,
    initial_equity: float = 100_000.0,
) -> OptimizationResult:
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]

    rows: List[Dict[str, Any]] = []
    best_s = float("-inf")
    best_params: Dict[str, Any] = {}

    for combo in product(*values):
        overrides = dict(zip(keys, combo))
        cfg = ICTFVGConfig(**{**base_cfg.__dict__, **overrides})

        feed = BarDataFeed(bars=bars)
        strat = ICTFVGStrategy(cfg=cfg)
        exec_model = ExecutionModel(execution_cfg)
        port = Portfolio(initial_equity=initial_equity)
        bt = Backtester(feed=feed, strategy=strat, execution=exec_model, portfolio=port)
        res = bt.run()
        m = compute_metrics(equity_curve=res.equity_curve, trades=res.trades)
        s = _score(m)

        row = {**overrides}
        row.update(
            {
                "score": s,
                "total_return": m.total_return,
                "max_drawdown": m.max_drawdown,
                "trades": m.trades,
                "win_rate": m.win_rate,
                "profit_factor": m.profit_factor,
            }
        )
        rows.append(row)

        if s > best_s:
            best_s = s
            best_params = overrides

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return OptimizationResult(results=df, best_params=best_params)

