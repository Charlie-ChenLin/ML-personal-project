from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .aggregation import (
    detect_date_col,
    detect_single_value_col,
    detect_value_mode,
    monthly_aggregate_min_max_avg,
    monthly_aggregate_single_value,
)
from .feature_engineering import FeatureEngineeringConfig, add_engineered_features


TargetMetric = Literal["mean", "last"]


@dataclass(frozen=True)
class PPDataset:
    target_metric: TargetMetric
    include_futures: bool
    engineer_features: bool
    strong_threshold: float
    flat_threshold: float
    data: pd.DataFrame


FILE_SPECS: list[tuple[str, str]] = [
    ("PP价格", "1-华东市场PP粒市场价_法定工作日.xlsx"),
    ("PP产量", "2-中国PP月度产量.xlsx"),
    ("塑编开工率", "3-塑编行业周度开工率.xlsx"),
    ("排产比例", "4-PP拉丝级生产比例日度数据.xlsx"),
    ("PP进口量", "5-PP进口量月度数据_滞后一月更新.xlsx"),
    ("PP开工率", "6-PP注塑制品周度开工率.xlsx"),
    ("BOPP开工率", "7-BOPP月度开工率.xlsx"),
    ("PDH成本", "8-PDH生产路线日度含税成本.xlsx"),
    ("乙烯成本", "9-乙烯裂解生产路线日度含税成本.xlsx"),
    ("MTO成本", "10-MTO生产路线日度含税成本.xlsx"),
    ("CTO成本", "11-CTO生产路线日度含税成本.xlsx"),
    ("丙烯成本", "12-外采丙烯生产路线日度含税成本.xlsx"),
    ("期货价格", "13-大商所PP期货价格_短数据.xlsx"),
    ("检修损失", "14-PP月度检修实际损失量.xlsx"),
    ("PP石化库存", "15-pp石化库存_每3个工作日更新.xlsx"),
    ("GDP", "16-年度GDP.xlsx"),
]

StrengthLabel = Literal["big_up", "small_up", "flat", "small_down", "big_down"]


def strength_label(
    ret: float, *, strong_threshold: float, flat_threshold: float
) -> StrengthLabel:
    if strong_threshold <= 0:
        raise ValueError("strong_threshold must be > 0")
    if flat_threshold < 0:
        raise ValueError("flat_threshold must be >= 0")
    if flat_threshold >= strong_threshold:
        raise ValueError("flat_threshold must be < strong_threshold")

    if ret <= -strong_threshold:
        return "big_down"
    if ret < -flat_threshold:
        return "small_down"
    if abs(ret) <= flat_threshold:
        return "flat"
    if ret < strong_threshold:
        return "small_up"
    return "big_up"


def _agg_one_source(df: pd.DataFrame) -> pd.DataFrame:
    date_col = detect_date_col(df)
    mode = detect_value_mode(df)
    if mode == "mma":
        res = monthly_aggregate_min_max_avg(df, date_col=date_col)
        return res.data
    value_col = detect_single_value_col(df)
    res = monthly_aggregate_single_value(df, date_col=date_col, value_col=value_col)
    return res.data


def build_pp_dataset(
    *,
    data_dir: Path,
    target_metric: TargetMetric = "mean",
    include_futures: bool = False,
    engineer_features: bool = False,
    strong_threshold: float = 0.05,
    flat_threshold: float = 0.005,
    fe_config: FeatureEngineeringConfig | None = None,
) -> PPDataset:
    if target_metric not in ("mean", "last"):
        raise ValueError("target_metric must be one of: mean/last")

    data_dir = Path(data_dir)
    features: list[pd.DataFrame] = []

    for factor_name, filename in FILE_SPECS:
        if (not include_futures) and (factor_name == "期货价格"):
            continue

        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        raw = pd.read_excel(path)
        agg = _agg_one_source(raw)
        agg = agg.rename(columns={c: f"{factor_name}__{c}" for c in agg.columns})
        features.append(agg)

    feature_table = pd.concat(features, axis=1).sort_index()

    # GDP is annual; forward-fill its latest value to subsequent months so it can act as a
    # slow-moving macro feature in monthly models.
    gdp_cols = [c for c in feature_table.columns if c.startswith("GDP__")]
    if gdp_cols:
        feature_table[gdp_cols] = feature_table[gdp_cols].ffill()

    price_col = f"PP价格__{target_metric}"
    if price_col not in feature_table.columns:
        raise KeyError(f"target price column not found: {price_col}")

    target = feature_table[price_col].rename("y")
    y_prev = target.shift(1).rename("y_prev")
    y_direction = ((target - y_prev) > 0).astype("Int64").rename("y_direction")
    y_return = ((target - y_prev) / y_prev).rename("y_return")
    y_strength = (
        y_return.map(
            lambda r: strength_label(
                float(r),
                strong_threshold=strong_threshold,
                flat_threshold=flat_threshold,
            )
        )
        .astype("string")
        .rename("y_strength")
    )

    # X(t) uses info up to t-1 to predict y(t)
    X = feature_table.shift(1)

    out = pd.concat([X, target, y_prev, y_direction, y_return, y_strength], axis=1)
    out = out.dropna(subset=["y", "y_prev"])  # first month cannot be predicted

    # calendar features (for seasonality/trend)
    out["cal_year"] = out.index.year
    out["cal_month"] = out.index.month
    out["cfg_strong_threshold"] = strong_threshold
    out["cfg_flat_threshold"] = flat_threshold

    out = out.reset_index(names="month")
    out["month"] = out["month"].astype(str)  # e.g. 2021-07

    if engineer_features:
        out = add_engineered_features(out, config=fe_config)

    return PPDataset(
        target_metric=target_metric,
        include_futures=include_futures,
        engineer_features=engineer_features,
        strong_threshold=strong_threshold,
        flat_threshold=flat_threshold,
        data=out,
    )
