from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StageConfig:
    trend_window: int = 6
    vol_window: int = 6
    trend_threshold: float = 0.02
    vol_quantile: float = 0.6
    min_stage_len: int = 6


def to_month_period(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.astype(str), freq="M")


def compute_regimes(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    y_col: str = "y",
    y_return_col: str = "y_return",
    config: StageConfig | None = None,
) -> pd.DataFrame:
    cfg = config or StageConfig()

    work = df[[month_col, y_col, y_return_col]].copy()
    work[month_col] = to_month_period(work[month_col])
    work = work.sort_values(month_col).reset_index(drop=True)

    y = work[y_col].astype(float)
    r = work[y_return_col].astype(float)

    trend = y.pct_change(cfg.trend_window).fillna(0.0)
    vol = r.rolling(cfg.vol_window, min_periods=2).std(ddof=0).fillna(0.0)

    vol_ref = vol[vol > 0]
    vol_th = float(vol_ref.quantile(cfg.vol_quantile)) if not vol_ref.empty else 0.0

    trend_state = np.where(
        trend >= cfg.trend_threshold,
        "up",
        np.where(trend <= -cfg.trend_threshold, "down", "flat"),
    )
    vol_state = np.where(vol >= vol_th, "high_vol", "low_vol")
    regime = pd.Series(trend_state, index=work.index) + "_" + pd.Series(vol_state, index=work.index)

    out = work[[month_col]].copy()
    out["trend"] = trend.to_numpy()
    out["vol"] = vol.to_numpy()
    out["trend_state"] = trend_state
    out["vol_state"] = vol_state
    out["regime"] = regime.to_numpy()
    return out


def _rle(labels: Sequence[str]) -> list[tuple[int, int, str]]:
    if not labels:
        return []
    segs: list[tuple[int, int, str]] = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            segs.append((start, i - 1, cur))
            start = i
            cur = labels[i]
    segs.append((start, len(labels) - 1, cur))
    return segs


def merge_short_regimes(
    regimes: pd.Series,
    *,
    min_len: int,
    returns: pd.Series | None = None,
) -> pd.Series:
    if min_len <= 1:
        return regimes

    labels = regimes.astype(str).tolist()
    r = returns.astype(float).tolist() if returns is not None else None

    max_passes = len(labels) + 5
    for _ in range(max_passes):
        segs = _rle(labels)
        if len(segs) <= 1:
            break

        short_idxs = [i for i, (s, e, _) in enumerate(segs) if (e - s + 1) < min_len]
        if not short_idxs:
            break

        i = short_idxs[0]
        s, e, _ = segs[i]

        if i == 0:
            new_label = segs[i + 1][2]
        elif i == len(segs) - 1:
            new_label = segs[i - 1][2]
        else:
            prev_label = segs[i - 1][2]
            next_label = segs[i + 1][2]
            if r is None:
                new_label = prev_label
            else:
                prev_s, prev_e, _ = segs[i - 1]
                next_s, next_e, _ = segs[i + 1]
                mean_cur = float(np.mean(r[s : e + 1]))
                mean_prev = float(np.mean(r[prev_s : prev_e + 1]))
                mean_next = float(np.mean(r[next_s : next_e + 1]))
                new_label = prev_label if abs(mean_cur - mean_prev) <= abs(mean_cur - mean_next) else next_label

        for j in range(s, e + 1):
            labels[j] = new_label

    return pd.Series(labels, index=regimes.index, name=regimes.name)


def assign_stage_id(regimes: pd.Series, *, min_stage_len: int, returns: pd.Series | None = None) -> pd.Series:
    _, stage_id = finalize_regimes_and_stage_id(regimes, min_stage_len=min_stage_len, returns=returns)
    return stage_id


def finalize_regimes_and_stage_id(
    regimes: pd.Series,
    *,
    min_stage_len: int,
    returns: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    merged = merge_short_regimes(regimes, min_len=min_stage_len, returns=returns)

    stage = np.zeros(len(merged), dtype=int)
    cur = None
    sid = 0
    for i, lab in enumerate(merged.astype(str).tolist()):
        if lab != cur:
            sid += 1
            cur = lab
        stage[i] = sid

    merged = merged.rename("regime_merged")
    stage_id = pd.Series(stage, index=merged.index, name="stage_id")
    return merged, stage_id


def summarize_stages(
    df: pd.DataFrame,
    *,
    month_col: str = "month",
    y_col: str = "y",
    y_return_col: str = "y_return",
    stage_id: pd.Series,
    regime: pd.Series,
) -> pd.DataFrame:
    work = df[[month_col, y_col, y_return_col]].copy()
    work[month_col] = to_month_period(work[month_col])
    work = work.sort_values(month_col).reset_index(drop=True)

    sid = stage_id.to_numpy()
    reg = regime.to_numpy()
    work["stage_id"] = sid
    work["regime"] = reg

    rows = []
    for gid, g in work.groupby("stage_id", sort=True):
        g = g.sort_values(month_col)
        start = g[month_col].iloc[0]
        end = g[month_col].iloc[-1]
        y_start = float(g[y_col].iloc[0])
        y_end = float(g[y_col].iloc[-1])
        rows.append(
            {
                "stage_id": int(gid),
                "regime": str(g["regime"].iloc[0]),
                "start_month": str(start),
                "end_month": str(end),
                "n_months": int(len(g)),
                "y_start": y_start,
                "y_end": y_end,
                "cum_return": float(y_end / y_start - 1.0) if y_start != 0 else np.nan,
                "mean_return": float(g[y_return_col].mean()),
                "vol_return": float(g[y_return_col].std(ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("stage_id").reset_index(drop=True)


def _default_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {
        "month",
        "stage_id",
        "regime",
        "trend",
        "vol",
        "trend_state",
        "vol_state",
        "y",
        "y_prev",
        "y_direction",
        "y_return",
        "y_strength",
        "cfg_strong_threshold",
        "cfg_flat_threshold",
    }
    return [c for c in df.columns if c not in exclude]


def _factor_group(feature_name: str) -> str | None:
    if feature_name.startswith("missingindicator_"):
        return None
    if "__" in feature_name:
        return feature_name.split("__", 1)[0]
    if feature_name.startswith("cal_"):
        return "Calendar"
    return feature_name


def ridge_stage_factor_weights(
    df_stage: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    target_col: str = "y",
    alpha: float = 1.0,
    random_state: int = 42,
    exclude_groups: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = feature_cols or _default_feature_cols(df_stage)
    feature_cols = [c for c in feature_cols if df_stage[c].notna().any()]
    if not feature_cols:
        raise ValueError("no usable features (all NaN in this stage)")
    X = df_stage[feature_cols]
    y = df_stage[target_col].astype(float).to_numpy()

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Ridge(alpha=alpha, random_state=random_state)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X, y)

    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = np.asarray(pipe.named_steps["model"].coef_, dtype=float)

    fdf = pd.DataFrame({"feature": names, "coef": coef})
    fdf["abs_coef"] = fdf["coef"].abs()
    fdf["group"] = fdf["feature"].map(_factor_group)
    fdf = fdf.dropna(subset=["group"]).reset_index(drop=True)
    if exclude_groups:
        fdf = fdf[~fdf["group"].isin(set(exclude_groups))].reset_index(drop=True)

    gdf = (
        fdf.groupby("group", sort=False)
        .agg(group_abs_coef=("abs_coef", "sum"), group_coef=("coef", "sum"))
        .reset_index()
    )
    total = float(gdf["group_abs_coef"].sum())
    gdf["group_abs_weight"] = gdf["group_abs_coef"] / total if total > 0 else np.nan
    gdf = gdf.sort_values("group_abs_coef", ascending=False).reset_index(drop=True)
    return fdf, gdf


def logreg_stage_factor_weights(
    df_stage: pd.DataFrame,
    *,
    target_col: str = "y_strength",
    feature_cols: list[str] | None = None,
    random_state: int = 42,
    exclude_groups: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_col not in df_stage.columns:
        raise KeyError(f"missing target column: {target_col}")

    y_raw = df_stage[target_col].astype(str)
    if y_raw.nunique(dropna=True) < 2:
        raise ValueError("need >=2 classes to fit LogisticRegression")

    feature_cols = feature_cols or _default_feature_cols(df_stage)
    feature_cols = [c for c in feature_cols if df_stage[c].notna().any()]
    if not feature_cols:
        raise ValueError("no usable features (all NaN in this stage)")
    X = df_stage[feature_cols]
    y = y_raw.to_numpy()

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(max_iter=5000, random_state=random_state)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X, y)

    names = pipe.named_steps["pre"].get_feature_names_out()
    coef = np.asarray(pipe.named_steps["model"].coef_, dtype=float)  # (K, D) or (1, D)
    mag = np.mean(np.abs(coef), axis=0)

    fdf = pd.DataFrame({"feature": names, "coef_mag": mag})
    fdf["group"] = fdf["feature"].map(_factor_group)
    fdf = fdf.dropna(subset=["group"]).reset_index(drop=True)
    if exclude_groups:
        fdf = fdf[~fdf["group"].isin(set(exclude_groups))].reset_index(drop=True)

    gdf = (
        fdf.groupby("group", sort=False)
        .agg(group_coef_mag=("coef_mag", "sum"))
        .reset_index()
    )
    total = float(gdf["group_coef_mag"].sum())
    gdf["group_weight"] = gdf["group_coef_mag"] / total if total > 0 else np.nan
    gdf = gdf.sort_values("group_coef_mag", ascending=False).reset_index(drop=True)
    return fdf, gdf
