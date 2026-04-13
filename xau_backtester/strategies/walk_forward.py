from __future__ import annotations

from pathlib import Path

import pandas as pd

from xau_backtester.analytics.metrics import compute_metrics
from xau_backtester.analytics.reporting import print_summary, summary_table
from xau_backtester.data.loader import load_csv_folder
from xau_backtester.data.resample import ticks_to_ohlcv
from xau_backtester.engine.execution import ExecutionConfig
from xau_backtester.experiments.validate import run_with_params
from xau_backtester.strategies.ict_fvg_fibo import ICTFVGConfig


TICK_DIR = r"C:\Users\saeed\OneDrive - University of La Verne\Desktop\xauusd_tick"
TZ = "UTC"
START = "2025-08-01"
END = "2026-04-01"

INITIAL_EQUITY = 100_000.0
SPREAD = 0.20
SLIPPAGE = 0.05

TRAIN_DAYS = 42   # 6 weeks
TEST_DAYS = 14    # 2 weeks
STEP_DAYS = 14    # roll forward every 2 weeks

OUT_DIR = Path("outputs_walk_forward")


def load_bars() -> pd.DataFrame:
    start = pd.Timestamp(START)
    end = pd.Timestamp(END)

    ticks = load_csv_folder(
        TICK_DIR,
        kind="tick",
        tz=TZ,
        start=start,
        end=end,
    ).df

    bars = ticks_to_ohlcv(ticks)
    bars = bars.sort_values("time", kind="mergesort").reset_index(drop=True)
    return bars


def main() -> None:
    bars = load_bars()

    base_cfg = ICTFVGConfig(
        risk_pct=0.005,
        min_gap=0.25,
        fib_lower=0.55,
        fib_upper=0.70,
        tp_extension=-1.0,
        stop_buffer=0.20,
        allowed_weekdays=(2, 4),  # Wed, Thu, Fri
        session_start_hour_utc=7,
        session_end_hour_utc=11,
        max_zone_age_bars=8,
        min_impulse_size=3.0,
        strong_impulse_size=4.5,
        one_trade_per_zone=True,
        strong_only=True,
        require_rejection_for_non_strong=True,
        require_close_back_outside_zone=True,
        require_directional_close=True,
        require_rejection_beyond_zone_mid=True,
        entry_at="mid",
    )

    exec_cfg = ExecutionConfig(
        fixed_spread=SPREAD,
        slippage=SLIPPAGE,
        slippage_mode="fixed",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_start = pd.Timestamp(START, tz="UTC")
    overall_end = pd.Timestamp(END, tz="UTC")

    rows: list[dict] = []

    window_start = overall_start

    while True:
        train_start = window_start
        train_end = train_start + pd.Timedelta(days=TRAIN_DAYS)
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=TEST_DAYS)

        if test_end > overall_end:
            break

        train_bars = bars[(bars["time"] >= train_start) & (bars["time"] < train_end)].copy()
        test_bars = bars[(bars["time"] >= test_start) & (bars["time"] < test_end)].copy()

        if train_bars.empty or test_bars.empty:
            window_start += pd.Timedelta(days=STEP_DAYS)
            continue

        train_run = run_with_params(
            bars=train_bars,
            base_cfg=base_cfg,
            params={},
            execution_cfg=exec_cfg,
            initial_equity=INITIAL_EQUITY,
        )
        test_run = run_with_params(
            bars=test_bars,
            base_cfg=base_cfg,
            params={},
            execution_cfg=exec_cfg,
            initial_equity=INITIAL_EQUITY,
        )

        train_m = train_run.metrics
        test_m = test_run.metrics

        rows.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "train_trades": train_m.trades,
                "train_return": train_m.total_return,
                "train_pf": train_m.profit_factor,
                "train_avg_r": train_m.avg_r,
                "train_dd": train_m.max_drawdown,
                "test_trades": test_m.trades,
                "test_return": test_m.total_return,
                "test_pf": test_m.profit_factor,
                "test_avg_r": test_m.avg_r,
                "test_dd": test_m.max_drawdown,
            }
        )

        window_start += pd.Timedelta(days=STEP_DAYS)

    wf = pd.DataFrame(rows)
    wf.to_csv(OUT_DIR / "walk_forward_summary.csv", index=False)

    print("\n=== WALK-FORWARD SUMMARY ===")
    print(wf)

    if not wf.empty:
        print("\n=== AGGREGATE TEST SUMMARY ===")
        print(f"Windows: {len(wf)}")
        print(f"Avg test PF: {wf['test_pf'].dropna().mean():.4f}")
        print(f"Median test PF: {wf['test_pf'].dropna().median():.4f}")
        print(f"Avg test return: {wf['test_return'].dropna().mean():.4f}")
        print(f"Avg test avg_r: {wf['test_avg_r'].dropna().mean():.4f}")
        print(f"Worst test DD: {wf['test_dd'].dropna().min():.4f}")
        print(f"Positive test windows: {(wf['test_return'] > 0).sum()} / {len(wf)}")


if __name__ == "__main__":
    main()