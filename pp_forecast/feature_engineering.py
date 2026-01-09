from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    momentum_lags: tuple[int, ...] = (1, 3, 12)
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    base_stat: str = "mean"
    add_intra_month_shape: bool = True
    add_calendar_cyclical: bool = True
    add_time_index: bool = True
    add_spreads: bool = True
    add_ratios_and_indices: bool = True


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = a.astype(float)
    b = b.astype(float)
    out = a / b.replace({0.0: np.nan})
    return out


def _month_period(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.astype(str), freq="M")


def add_engineered_features(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    config: FeatureEngineeringConfig | None = None,
) -> pd.DataFrame:
    cfg = config or FeatureEngineeringConfig()
    out = df.copy()

    if month_col not in out.columns:
        raise KeyError(f"missing month column: {month_col}")

    month = _month_period(out[month_col])
    out["_month"] = month
    out = out.sort_values("_month").reset_index(drop=True)

    new_cols: dict[str, pd.Series | np.ndarray] = {}

    if cfg.add_time_index:
        new_cols["cal_time_index"] = np.arange(len(out), dtype=float)

    if cfg.add_calendar_cyclical and "cal_month" in out.columns:
        m = out["cal_month"].astype(float)
        new_cols["cal_month_sin"] = np.sin(2.0 * np.pi * m / 12.0)
        new_cols["cal_month_cos"] = np.cos(2.0 * np.pi * m / 12.0)

    feature_cols = [c for c in out.columns if "__" in c]
    groups = sorted({c.split("__", 1)[0] for c in feature_cols})

    # Per-factor time-series features on the base_stat column (already lagged in our dataset).
    for group in groups:
        base_col = f"{group}__{cfg.base_stat}"
        if base_col not in out.columns:
            continue

        s = out[base_col].astype(float)
        for lag in cfg.momentum_lags:
            new_cols[f"{group}__{cfg.base_stat}_mom_{lag}"] = s.pct_change(lag, fill_method=None)

        for w in cfg.rolling_windows:
            min_p = min(3, w)
            new_cols[f"{group}__{cfg.base_stat}_roll_mean_{w}"] = s.rolling(w, min_periods=min_p).mean()
            new_cols[f"{group}__{cfg.base_stat}_roll_std_{w}"] = s.rolling(w, min_periods=min_p).std(
                ddof=0
            )

        if cfg.add_intra_month_shape:
            mean_col = f"{group}__mean"
            min_col = f"{group}__min"
            max_col = f"{group}__max"
            last_col = f"{group}__last"
            range_col = f"{group}__range"

            if all(c in out.columns for c in [mean_col, range_col]):
                new_cols[f"{group}__range_over_mean"] = _safe_div(out[range_col], out[mean_col])
            if all(c in out.columns for c in [last_col, mean_col]):
                new_cols[f"{group}__last_minus_mean"] = (
                    out[last_col].astype(float) - out[mean_col].astype(float)
                )
            if all(c in out.columns for c in [max_col, min_col]):
                new_cols[f"{group}__max_minus_min"] = (
                    out[max_col].astype(float) - out[min_col].astype(float)
                )

    # Cross-factor spreads and ratios (domain-inspired).
    if cfg.add_spreads and "PP价格__mean" in out.columns:
        pp = out["PP价格__mean"].astype(float)
        cost_groups = ["PDH成本", "乙烯成本", "MTO成本", "CTO成本", "丙烯成本"]
        for g in cost_groups:
            c = f"{g}__mean"
            if c in out.columns:
                new_cols[f"价差__PP价格_minus_{g}__mean"] = pp - out[c].astype(float)
                new_cols[f"比值__PP价格_div_{g}__mean"] = _safe_div(pp, out[c])

    if cfg.add_ratios_and_indices:
        # Supply/demand style ratios
        if all(c in out.columns for c in ["PP进口量__mean", "PP产量__mean"]):
            new_cols["供需__进口_div_产量__mean"] = _safe_div(out["PP进口量__mean"], out["PP产量__mean"])
        if all(c in out.columns for c in ["PP石化库存__mean", "PP产量__mean"]):
            new_cols["供需__库存_div_产量__mean"] = _safe_div(out["PP石化库存__mean"], out["PP产量__mean"])
        if all(c in out.columns for c in ["检修损失__mean", "PP产量__mean"]):
            new_cols["供需__检修损失_div_产量__mean"] = _safe_div(out["检修损失__mean"], out["PP产量__mean"])

        # Downstream operating rate index
        downstream_cols = [c for c in ["塑编开工率__mean", "PP开工率__mean", "BOPP开工率__mean"] if c in out.columns]
        if downstream_cols:
            new_cols["指数__下游开工均值__mean"] = out[downstream_cols].astype(float).mean(axis=1)

        # Futures basis if futures features exist (same month lag already applied).
        if "期货价格__mean" in out.columns and "PP价格__mean" in out.columns:
            new_cols["价差__现货_minus_期货__mean"] = (
                out["PP价格__mean"].astype(float) - out["期货价格__mean"].astype(float)
            )

    if new_cols:
        new_df = pd.DataFrame(new_cols)
        new_df = new_df.dropna(axis=1, how="all")
        out = pd.concat([out, new_df], axis=1)

    out = out.drop(columns=["_month"])
    return out
