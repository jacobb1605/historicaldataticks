from __future__ import annotations

import pandas as pd


TRADES_PATH = r"outputs_london_gold_tuned_wed_thu_fri\out_of_sample_trades_london.csv"


def safe_pf(gross_win: float, gross_loss: float):
    if gross_loss == 0:
        return None
    return gross_win / gross_loss


def main() -> None:
    df = pd.read_csv(TRADES_PATH)

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    df["date"] = df["entry_time"].dt.date
    df["weekday"] = df["entry_time"].dt.day_name()
    df["entry_hour_utc"] = df["entry_time"].dt.hour

    daily = df.groupby(["date", "weekday"]).agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum()),
        gross_win=("pnl", lambda x: x[x > 0].sum()),
        gross_loss=("pnl", lambda x: -x[x < 0].sum()),
        best_trade=("pnl", "max"),
        worst_trade=("pnl", "min"),
    ).reset_index()

    daily["win_rate"] = daily["wins"] / daily["trades"]
    daily["profit_factor"] = daily.apply(
        lambda row: safe_pf(row["gross_win"], row["gross_loss"]),
        axis=1,
    )

    weekday_summary = df.groupby("weekday").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum()),
        gross_win=("pnl", lambda x: x[x > 0].sum()),
        gross_loss=("pnl", lambda x: -x[x < 0].sum()),
    ).reset_index()

    weekday_summary["win_rate"] = weekday_summary["wins"] / weekday_summary["trades"]
    weekday_summary["profit_factor"] = weekday_summary.apply(
        lambda row: safe_pf(row["gross_win"], row["gross_loss"]),
        axis=1,
    )

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday_summary["weekday"] = pd.Categorical(
        weekday_summary["weekday"],
        categories=weekday_order,
        ordered=True,
    )
    weekday_summary = weekday_summary.sort_values("weekday").reset_index(drop=True)

    hour_summary = df.groupby("entry_hour_utc").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
        wins=("pnl", lambda x: (x > 0).sum()),
        losses=("pnl", lambda x: (x < 0).sum()),
        gross_win=("pnl", lambda x: x[x > 0].sum()),
        gross_loss=("pnl", lambda x: -x[x < 0].sum()),
    ).reset_index()

    hour_summary["win_rate"] = hour_summary["wins"] / hour_summary["trades"]
    hour_summary["profit_factor"] = hour_summary.apply(
        lambda row: safe_pf(row["gross_win"], row["gross_loss"]),
        axis=1,
    )

    print("\n=== DAILY PERFORMANCE ===")
    print(daily.to_string(index=False))

    print("\n=== WEEKDAY SUMMARY ===")
    print(weekday_summary.to_string(index=False))

    print("\n=== ENTRY HOUR SUMMARY (UTC) ===")
    print(hour_summary.to_string(index=False))

    print("\n=== SUMMARY ===")
    print(f"Total days: {len(daily)}")
    print(f"Profitable days: {(daily['pnl'] > 0).sum()}")
    print(f"Losing days: {(daily['pnl'] < 0).sum()}")
    print(f"Avg daily PnL: {daily['pnl'].mean():.2f}")
    print(f"Median daily PnL: {daily['pnl'].median():.2f}")
    print(f"Best day: {daily['pnl'].max():.2f}")
    print(f"Worst day: {daily['pnl'].min():.2f}")


if __name__ == "__main__":
    main()