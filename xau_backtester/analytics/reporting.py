from __future__ import annotations

import pandas as pd
from tabulate import tabulate

from xau_backtester.analytics.metrics import PerformanceMetrics


def summary_table(metrics: PerformanceMetrics, *, label: str) -> pd.DataFrame:
    rows = [
        ("label", label),
        ("start", str(metrics.start)),
        ("end", str(metrics.end)),
        ("bars", metrics.bars),
        ("trades", metrics.trades),
        ("total_return", metrics.total_return),
        ("cagr", metrics.cagr),
        ("max_drawdown", metrics.max_drawdown),
        ("sharpe_like", metrics.sharpe_like),
        ("win_rate", metrics.win_rate),
        ("avg_r", metrics.avg_r),
        ("profit_factor", metrics.profit_factor),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def print_summary(df: pd.DataFrame) -> None:
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

