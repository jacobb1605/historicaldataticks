from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    in_sample_ratio: float = 0.7


def chronological_split(
    df: pd.DataFrame,
    *,
    cfg: SplitConfig = SplitConfig(),
    time_col: str = "time",
    split_time: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split empty DataFrame.")
    d = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    if split_time is None:
        n = len(d)
        k = int(n * float(cfg.in_sample_ratio))
        k = max(1, min(n - 1, k))
        ins = d.iloc[:k].copy()
        oos = d.iloc[k:].copy()
        return ins, oos

    st = pd.Timestamp(split_time)
    ins = d[d[time_col] < st].copy()
    oos = d[d[time_col] >= st].copy()
    if ins.empty or oos.empty:
        raise ValueError("Split produced an empty segment. Adjust split_time or data range.")
    return ins, oos

