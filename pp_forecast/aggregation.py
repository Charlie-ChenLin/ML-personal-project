from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class MonthlyAggResult:
    month: pd.PeriodIndex
    data: pd.DataFrame


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _to_month_index(dts: pd.Series) -> pd.PeriodIndex:
    dts = _to_datetime(dts)
    return dts.dt.to_period("M")


def monthly_aggregate_min_max_avg(
    df: pd.DataFrame,
    *,
    date_col: str,
    avg_col: str = "平均",
    min_col: str = "最低",
    max_col: str = "最高",
) -> MonthlyAggResult:
    keep = [date_col, avg_col, min_col, max_col]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    work = df[keep].copy()
    work[date_col] = _to_datetime(work[date_col])
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    work["month"] = work[date_col].dt.to_period("M")

    g = work.groupby("month", sort=True)
    out = pd.DataFrame(
        {
            "mean": g[avg_col].mean(),
            "min": g[min_col].min(),
            "max": g[max_col].max(),
            "last": g[avg_col].last(),
        }
    )
    out["range"] = out["max"] - out["min"]

    out.index = out.index.astype("period[M]")
    return MonthlyAggResult(month=out.index, data=out)


def monthly_aggregate_single_value(
    df: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
) -> MonthlyAggResult:
    keep = [date_col, value_col]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    work = df[keep].copy()
    work[date_col] = _to_datetime(work[date_col])
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    work["month"] = work[date_col].dt.to_period("M")

    g = work.groupby("month", sort=True)
    out = pd.DataFrame(
        {
            "mean": g[value_col].mean(),
            "min": g[value_col].min(),
            "max": g[value_col].max(),
            "last": g[value_col].last(),
        }
    )
    out["range"] = out["max"] - out["min"]

    out.index = out.index.astype("period[M]")
    return MonthlyAggResult(month=out.index, data=out)


def detect_date_col(df: pd.DataFrame) -> str:
    for cand in ["日期", "date", "Date"]:
        if cand in df.columns:
            return cand
    raise KeyError("date column not found (expected one of: 日期/date/Date)")


def detect_value_mode(df: pd.DataFrame) -> Literal["mma", "single"]:
    if all(c in df.columns for c in ["最低", "最高", "平均"]):
        return "mma"
    return "single"


def detect_single_value_col(df: pd.DataFrame) -> str:
    for cand in ["inventory", "值"]:
        if cand in df.columns:
            return cand

    numeric = df.select_dtypes(include="number").columns.tolist()
    candidates = [c for c in numeric if c not in {"DIID", "年", "月"}]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        "single value column not found; expected one of inventory/值 or a unique numeric column"
    )

