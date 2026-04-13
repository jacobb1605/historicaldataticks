from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from xau_backtester.analytics.metrics import PerformanceMetrics, compute_metrics
from xau_backtester.engine.backtester import Backtester, BacktestResult
from xau_backtester.engine.data_feed import BarDataFeed
from xau_backtester.engine.execution import ExecutionConfig, ExecutionModel
from xau_backtester.engine.portfolio import Portfolio
from xau_backtester.strategies.ict_fvg_fibo import ICTFVGConfig, ICTFVGStrategy


@dataclass(frozen=True)
class ValidationRun:
    params: Dict[str, Any]
    result: BacktestResult
    metrics: PerformanceMetrics


def run_with_params(
    *,
    bars: pd.DataFrame,
    base_cfg: ICTFVGConfig,
    params: Dict[str, Any],
    execution_cfg: ExecutionConfig,
    initial_equity: float = 100_000.0,
) -> ValidationRun:
    cfg = ICTFVGConfig(**{**base_cfg.__dict__, **params})
    feed = BarDataFeed(bars=bars)
    strat = ICTFVGStrategy(cfg=cfg)
    exec_model = ExecutionModel(execution_cfg)
    port = Portfolio(initial_equity=initial_equity)
    bt = Backtester(feed=feed, strategy=strat, execution=exec_model, portfolio=port)
    res = bt.run()
    m = compute_metrics(equity_curve=res.equity_curve, trades=res.trades)
    return ValidationRun(params=params, result=res, metrics=m)

