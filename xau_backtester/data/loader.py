from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import pandas as pd


DataKind = Literal["tick", "m1"]


@dataclass(frozen=True)
class LoadedData:
    kind: DataKind
    df: pd.DataFrame
    source_files: list[Path]


def _read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _parse_time_to_tz(series: pd.Series, *, tz: str = "UTC") -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce", utc=True)
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps.")
    return ts.dt.tz_convert(tz)


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}. Got: {list(df.columns)}")


def _coerce_numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_bound(ts: pd.Timestamp, *, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def load_csv_folder(
    folder: str | Path,
    *,
    kind: DataKind,
    file_glob: str = "*.csv",
    tz: str = "UTC",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> LoadedData:
    """
    Load multiple CSV files from a folder into a single, strictly time-ordered DataFrame.

    Notes:
    - Column names are normalized to lowercase.
    - Timestamps are parsed as UTC, then converted to `tz`.
    - Duplicate timestamps are dropped with keep='last'.
    """

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(str(folder))

    files = sorted(folder.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched {file_glob!r} under {str(folder)!r}")

    frames: list[pd.DataFrame] = []

    for f in files:
        df = _normalize_columns(_read_csv_safely(f))

        if "time" not in df.columns:
            for alt in ("timestamp", "datetime", "date"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "time"})
                    break

        df = df.rename(
            columns={
                "bidprice": "bid",
                "askprice": "ask",
                "bid_price": "bid",
                "ask_price": "ask",
                "openprice": "open",
                "highprice": "high",
                "lowprice": "low",
                "closeprice": "close",
                "open_price": "open",
                "high_price": "high",
                "low_price": "low",
                "close_price": "close",
            }
        )

        _require_columns(df, ["time"], context=f"file={f.name}")
        df["time"] = _parse_time_to_tz(df["time"], tz=tz)

        if kind == "tick":
            _require_columns(df, ["bid", "ask"], context=f"file={f.name} kind=tick")
            keep_cols = [c for c in ["time", "bid", "ask", "volume"] if c in df.columns]
            df = df[keep_cols]
            df = _coerce_numeric_columns(df, ["bid", "ask", "volume"])
            df = df.dropna(subset=["bid", "ask"])

        elif kind == "m1":
            _require_columns(df, ["open", "high", "low", "close"], context=f"file={f.name} kind=m1")
            keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[keep_cols]
            df = _coerce_numeric_columns(df, ["open", "high", "low", "close", "volume"])
            df = df.dropna(subset=["open", "high", "low", "close"])

        else:
            raise ValueError(f"Unknown kind: {kind!r}")

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["time"])
    out = out.sort_values("time", kind="mergesort").reset_index(drop=True)

    if start is not None:
        start_ts = _normalize_bound(pd.Timestamp(start), tz=tz)
        out = out[out["time"] >= start_ts]

    if end is not None:
        end_ts = _normalize_bound(pd.Timestamp(end), tz=tz)
        out = out[out["time"] <= end_ts]

    out = out.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    return LoadedData(kind=kind, df=out, source_files=files)