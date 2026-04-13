from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from xau_backtester.analytics.metrics import compute_metrics
from xau_backtester.analytics.reporting import print_summary, summary_table
from xau_backtester.data.loader import load_csv_folder
from xau_backtester.data.resample import ticks_to_ohlcv
from xau_backtester.engine.execution import ExecutionConfig
from xau_backtester.experiments.split import SplitConfig, chronological_split
from xau_backtester.experiments.validate import run_with_params
from xau_backtester.strategies.ict_fvg_fibo import ICTFVGConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XAUUSD locked-baseline split robustness test.")
    p.add_argument("--tick_dir", type=str, default="", help="Folder containing tick bid/ask CSVs.")
    p.add_argument("--start", type=str, default="", help="Optional start timestamp (inclusive).")
    p.add_argument("--end", type=str, default="", help="Optional end timestamp (inclusive).")
    p.add_argument("--split_ratio", type=float, default=0.7, help="In-sample ratio if split_time not given.")
    p.add_argument("--split_time", type=str, default="", help="Explicit split timestamp, e.g. 2025-08-01T00:00:00Z")
    p.add_argument("--initial_equity", type=float, default=100_000.0)
    p.add_argument("--spread", type=float, default=0.20, help="Fixed spread in price units.")
    p.add_argument("--slippage", type=float, default=0.05, help="Fixed slippage in price units.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output folder for CSV results.")
    return p.parse_args()


def _load_bars(args: argparse.Namespace) -> pd.DataFrame:
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None

    if args.tick_dir:
        ticks = load_csv_folder(args.tick_dir, kind="tick", tz="UTC", start=start, end=end).df
        return ticks_to_ohlcv(ticks)

    raise ValueError("Provide --tick_dir.")


def main() -> None:
    args = _parse_args()
    bars = _load_bars(args)
    bars = bars.sort_values("time", kind="mergesort").reset_index(drop=True)

    split_time = pd.Timestamp(args.split_time) if args.split_time else None
    ins, oos = chronological_split(
        bars,
        cfg=SplitConfig(in_sample_ratio=args.split_ratio),
        split_time=split_time,
    )

    # LOCKED BASELINE
    base_cfg = ICTFVGConfig(
        risk_pct=0.005,
        min_gap=0.25,
        fib_lower=0.55,
        fib_upper=0.70,
        tp_extension=-1.0,
        stop_buffer=0.20,
        entry_at="mid",
        allowed_weekdays=(2, 4),  # Wednesday, Friday
        session_start_hour_utc=5,
        session_end_hour_utc=9,
        max_zone_age_bars=8,
        min_impulse_size=3.0,
        strong_impulse_size=4.5,
        one_trade_per_zone=True,
        strong_only=True,
        require_rejection_for_non_strong=True,
        require_close_back_outside_zone=True,
        require_directional_close=True,
        require_rejection_beyond_zone_mid=True,
    )

    exec_cfg = ExecutionConfig(
        fixed_spread=args.spread,
        slippage=args.slippage,
        slippage_mode="fixed",
    )

    ins_run = run_with_params(
        bars=ins,
        base_cfg=base_cfg,
        params={},
        execution_cfg=exec_cfg,
        initial_equity=args.initial_equity,
    )
    oos_run = run_with_params(
        bars=oos,
        base_cfg=base_cfg,
        params={},
        execution_cfg=exec_cfg,
        initial_equity=args.initial_equity,
    )

    split_label = args.split_time if args.split_time else f"ratio_{args.split_ratio}"

    ins_tbl = summary_table(ins_run.metrics, label=f"in_sample_{split_label}")
    oos_tbl = summary_table(oos_run.metrics, label=f"out_of_sample_{split_label}")
    print_summary(pd.concat([ins_tbl, oos_tbl], ignore_index=True))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ins_run.result.trades.to_csv(out_dir / "in_sample_trades.csv", index=False)
    oos_run.result.trades.to_csv(out_dir / "out_of_sample_trades.csv", index=False)
    ins_run.result.equity_curve.to_csv(out_dir / "in_sample_equity.csv", index=False)
    oos_run.result.equity_curve.to_csv(out_dir / "out_of_sample_equity.csv", index=False)

    _ = compute_metrics(
        equity_curve=oos_run.result.equity_curve,
        trades=oos_run.result.trades,
    )

    print("\nLocked baseline params:")
    print("  min_gap = 0.25")
    print("  fib_lower = 0.55")
    print("  fib_upper = 0.70")
    print("  tp_extension = -1.0")
    print("  stop_buffer = 0.20")
    print("  entry_at = mid")
    print("  max_zone_age_bars = 8")
    print("  min_impulse_size = 3.0")
    print("  strong_impulse_size = 4.5")
    print("  strong_only = True")
    print("  one_trade_per_zone = True")
    print("  allowed_weekdays = (2, 4)")
    print("  session_start_hour_utc = 5")
    print("  session_end_hour_utc = 9")


if __name__ == "__main__":
    main()