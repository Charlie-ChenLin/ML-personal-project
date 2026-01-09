from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pp_forecast.aggregation import (
    detect_date_col,
    detect_single_value_col,
    detect_value_mode,
    monthly_aggregate_min_max_avg,
    monthly_aggregate_single_value,
)
from pp_forecast.feature_engineering import FeatureEngineeringConfig, add_engineered_features
from pp_forecast.q1_dataset import FILE_SPECS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EDA for PP project: raw Excel + monthly feature table + modeling datasets."
    )
    p.add_argument("--data-dir", type=Path, default=Path("PP数据"))
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        type=Path,
        help="Path to a processed dataset CSV (repeatable). Default: outputs/datasets/q1_long.csv & q1_with_futures.csv if exist.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    p.add_argument("--test-start", default="2021-01")
    p.add_argument("--test-end", default="2021-07")
    p.add_argument(
        "--engineer-features",
        action="store_true",
        help="Apply feature engineering before dataset EDA (for comparison).",
    )
    p.add_argument("--no-raw-plots", action="store_true", help="Skip per-raw-file plots.")
    return p.parse_args()


def _ensure_mpl() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".mplconfig").resolve()))


def _configure_matplotlib_fonts() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Arial Unicode MS",  # macOS often has this
        "STHeiti",
        "Songti SC",
        "PingFang SC",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen is not None:
        plt.rcParams["font.family"] = ["sans-serif"]
        # Put the CJK-capable font first to avoid missing-glyph warnings.
        fallback = ["DejaVu Sans"]
        plt.rcParams["font.sans-serif"] = [chosen] + [f for f in fallback if f != chosen]
    plt.rcParams["axes.unicode_minus"] = False


def _slug(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _guess_freq(dts: pd.Series) -> tuple[str, float | None]:
    dt = pd.to_datetime(dts, errors="coerce").dropna().drop_duplicates().sort_values()
    if len(dt) < 3:
        return "unknown", None
    diffs = dt.diff().dt.total_seconds().dropna() / 86400.0
    if diffs.empty:
        return "unknown", None
    med = float(diffs.median())
    if med <= 2:
        return "daily_or_irregular", med
    if med <= 10:
        return "weekly", med
    if med <= 40:
        return "monthly", med
    if med <= 200:
        return "quarterly_or_sparse", med
    return "annual_or_sparse", med


def _series_stats(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce")
    out: dict[str, float] = {
        "count": float(s.notna().sum()),
        "missing_rate": float(s.isna().mean()),
    }
    if s.notna().sum() == 0:
        out.update({k: float("nan") for k in ["mean", "std", "min", "p25", "p50", "p75", "max"]})
        return out
    qs = s.quantile([0.25, 0.5, 0.75])
    out.update(
        {
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "p25": float(qs.loc[0.25]),
            "p50": float(qs.loc[0.5]),
            "p75": float(qs.loc[0.75]),
            "max": float(s.max()),
        }
    )
    return out


@dataclass(frozen=True)
class RawProfile:
    factor: str
    filename: str
    mode: str
    start_date: str
    end_date: str
    n_rows: int
    n_cols: int
    n_unique_dates: int
    n_unique_months: int
    freq_guess: str
    median_gap_days: float | None


def profile_raw_files(*, data_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    raw_tables: dict[str, pd.DataFrame] = {}
    for factor, filename in FILE_SPECS:
        path = data_dir / filename
        if not path.exists():
            continue
        df = pd.read_excel(path)
        raw_tables[factor] = df

        mode = detect_value_mode(df)
        date_col = detect_date_col(df)

        dt = pd.to_datetime(df[date_col], errors="coerce")
        freq_guess, median_gap = _guess_freq(dt)
        n_unique_dates = int(dt.dropna().nunique())
        n_unique_months = int(dt.dropna().dt.to_period("M").nunique())

        start = dt.dropna().min()
        end = dt.dropna().max()
        start_s = start.date().isoformat() if pd.notna(start) else ""
        end_s = end.date().isoformat() if pd.notna(end) else ""

        base = RawProfile(
            factor=factor,
            filename=filename,
            mode=mode,
            start_date=start_s,
            end_date=end_s,
            n_rows=int(df.shape[0]),
            n_cols=int(df.shape[1]),
            n_unique_dates=n_unique_dates,
            n_unique_months=n_unique_months,
            freq_guess=freq_guess,
            median_gap_days=median_gap,
        )
        base_row = base.__dict__.copy()

        if mode == "mma":
            for col in ["平均", "最低", "最高"]:
                if col in df.columns:
                    for k, v in _series_stats(df[col]).items():
                        base_row[f"{col}__{k}"] = v
        else:
            value_col = detect_single_value_col(df)
            base_row["value_col"] = value_col
            for k, v in _series_stats(df[value_col]).items():
                base_row[f"value__{k}"] = v

        rows.append(base_row)

    prof = pd.DataFrame(rows).sort_values(["factor"]).reset_index(drop=True)
    return prof, raw_tables


def _raw_main_series(df: pd.DataFrame) -> tuple[str, pd.Series, pd.Series | None, pd.Series | None]:
    mode = detect_value_mode(df)
    date_col = detect_date_col(df)
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if mode == "mma":
        y = pd.to_numeric(df["平均"], errors="coerce")
        y_min = pd.to_numeric(df["最低"], errors="coerce")
        y_max = pd.to_numeric(df["最高"], errors="coerce")
        return date_col, pd.Series(y.values, index=dt), pd.Series(y_min.values, index=dt), pd.Series(
            y_max.values, index=dt
        )
    value_col = detect_single_value_col(df)
    y = pd.to_numeric(df[value_col], errors="coerce")
    return date_col, pd.Series(y.values, index=dt), None, None


def plot_raw_files(raw_tables: dict[str, pd.DataFrame], *, out_dir: Path) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)

    for factor, df in raw_tables.items():
        _, y, y_min, y_max = _raw_main_series(df)
        s = y.dropna().sort_index()
        if s.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        ax.plot(s.index, s.values, linewidth=1.0, alpha=0.5, color="tab:blue", label="raw")

        # overlay monthly mean (smoothed)
        monthly = s.groupby(s.index.to_period("M")).mean()
        ax.plot(
            monthly.index.to_timestamp("M"),
            monthly.values,
            linewidth=2.0,
            color="tab:red",
            label="monthly mean",
        )
        if y_min is not None and y_max is not None:
            smin = y_min.dropna().sort_index()
            smax = y_max.dropna().sort_index()
            if not smin.empty and not smax.empty:
                mmin = smin.groupby(smin.index.to_period("M")).min()
                mmax = smax.groupby(smax.index.to_period("M")).max()
                ax.fill_between(
                    mmin.index.to_timestamp("M"),
                    mmin.values,
                    mmax.reindex(mmin.index).values,
                    color="tab:blue",
                    alpha=0.10,
                    label="monthly min/max band",
                )

        ax.set_title(f"{factor} (raw + monthly smoothed)")
        ax.set_xlabel("date")
        ax.set_ylabel("value")
        ax.legend(loc="best")

        ax2 = axes[1]
        sns.histplot(s.values, bins=40, kde=True, ax=ax2, color="tab:blue")
        ax2.set_title(f"{factor} distribution")
        ax2.set_xlabel("value")

        fig.tight_layout()
        fig.savefig(out_dir / f"raw_{_slug(factor)}.png", dpi=200)
        plt.close(fig)


def build_monthly_feature_table(*, data_dir: Path, include_futures: bool = True) -> pd.DataFrame:
    features = []
    for factor, filename in FILE_SPECS:
        if (not include_futures) and factor == "期货价格":
            continue
        path = data_dir / filename
        if not path.exists():
            continue
        raw = pd.read_excel(path)
        date_col = detect_date_col(raw)
        mode = detect_value_mode(raw)
        if mode == "mma":
            agg = monthly_aggregate_min_max_avg(raw, date_col=date_col).data
        else:
            value_col = detect_single_value_col(raw)
            agg = monthly_aggregate_single_value(raw, date_col=date_col, value_col=value_col).data
        agg = agg.rename(columns={c: f"{factor}__{c}" for c in agg.columns})
        features.append(agg)

    if not features:
        raise SystemExit("no raw files found to build monthly feature table")

    table = pd.concat(features, axis=1).sort_index()
    table.index = table.index.astype("period[M]")
    return table


def plot_monthly_coverage(monthly_table: pd.DataFrame, *, out_dir: Path) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()

    # Coverage by factor (based on the mean column).
    factors = sorted({c.split("__", 1)[0] for c in monthly_table.columns if c.endswith("__mean")})
    months = monthly_table.index
    coverage = pd.DataFrame(index=months.astype(str))
    for f in factors:
        col = f"{f}__mean"
        if col in monthly_table.columns:
            coverage[f] = monthly_table[col].notna().astype(int)

    if not coverage.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            coverage.T,
            cmap="Greys",
            cbar=False,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        ax.set_title("Monthly data availability (1=available, 0=missing)")
        ax.set_xlabel("month")
        ax.set_ylabel("factor")
        fig.tight_layout()
        fig.savefig(out_dir / "monthly_coverage_heatmap.png", dpi=200)
        plt.close(fig)

        counts = coverage.sum(axis=0).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        counts.plot(kind="bar", ax=ax, color="tab:blue")
        ax.set_title("Available months per factor (mean series)")
        ax.set_xlabel("factor")
        ax.set_ylabel("n_months")
        fig.tight_layout()
        fig.savefig(out_dir / "monthly_coverage_counts.png", dpi=200)
        plt.close(fig)


def lag_correlation_table(
    monthly_table: pd.DataFrame,
    *,
    target_col: str,
    max_lag: int = 12,
    method: str = "spearman",
) -> pd.DataFrame:
    if target_col not in monthly_table.columns:
        raise KeyError(f"target_col not found: {target_col}")

    factors = sorted({c.split("__", 1)[0] for c in monthly_table.columns if c.endswith("__mean")})
    y = pd.to_numeric(monthly_table[target_col], errors="coerce")

    rows = []
    for f in factors:
        col = f"{f}__mean"
        if col not in monthly_table.columns:
            continue
        if col == target_col:
            continue
        x = pd.to_numeric(monthly_table[col], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue

        row = {"factor": f}
        for lag in range(0, max_lag + 1):
            shifted = x.shift(lag)  # x(t-lag) aligned to y(t)
            valid = pd.concat([y, shifted], axis=1).dropna()
            if len(valid) < 12:
                row[f"lag_{lag}"] = np.nan
                continue
            row[f"lag_{lag}"] = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
    return out


def plot_lag_correlation_heatmap(lag_corr: pd.DataFrame, *, out_dir: Path) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()

    if lag_corr.empty:
        return

    mat = lag_corr.set_index("factor")
    # keep only numeric lag columns
    lag_cols = [c for c in mat.columns if c.startswith("lag_")]
    mat = mat[lag_cols]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(mat))))
    sns.heatmap(mat, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("Lag correlation heatmap (factor mean vs PP price mean)")
    ax.set_xlabel("lag (months): factor(t-lag) vs price(t)")
    ax.set_ylabel("factor")
    fig.tight_layout()
    fig.savefig(out_dir / "lag_corr_heatmap.png", dpi=200)
    plt.close(fig)


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.astype(str), freq="M")


def _adf_test(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 12:
        return {"adf_stat": np.nan, "pvalue": np.nan, "nobs": float(len(s))}
    stat, pvalue, _, _, _, _ = __import__("statsmodels.tsa.stattools", fromlist=["adfuller"]).adfuller(
        s.to_numpy(), autolag="AIC"
    )
    return {"adf_stat": float(stat), "pvalue": float(pvalue), "nobs": float(len(s))}


def dataset_eda(
    df: pd.DataFrame,
    *,
    name: str,
    out_dir: Path,
    test_start: str,
    test_end: str,
    engineer_features: bool,
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.seasonal import seasonal_decompose

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()

    work = df.copy()
    work["month"] = _to_period(work["month"])
    work = work.sort_values("month").reset_index(drop=True)
    if engineer_features:
        work = add_engineered_features(work, config=FeatureEngineeringConfig())

    # Basic summary
    summary = {
        "dataset": name,
        "n_rows": int(len(work)),
        "n_cols": int(work.shape[1]),
        "month_start": str(work["month"].min()),
        "month_end": str(work["month"].max()),
    }

    y = pd.to_numeric(work["y"], errors="coerce")
    y_ret = pd.to_numeric(work["y_return"], errors="coerce") if "y_return" in work.columns else None

    stats = pd.DataFrame(
        [{"var": "y", **_series_stats(y)}]
        + ([{"var": "y_return", **_series_stats(y_ret)}] if y_ret is not None else [{"var": "y_return"}])
    )
    stats.to_csv(out_dir / "summary_stats.csv", index=False)

    # Counts
    if "y_direction" in work.columns:
        work["y_direction"] = work["y_direction"].astype("Int64")
        direction_counts = work["y_direction"].value_counts(dropna=False).sort_index()
        direction_counts.rename_axis("y_direction").to_frame("count").to_csv(
            out_dir / "direction_counts.csv", index=False
        )
    if "y_strength" in work.columns:
        strength_counts = work["y_strength"].astype(str).value_counts().rename_axis("y_strength")
        strength_counts.to_frame("count").reset_index().to_csv(out_dir / "strength_counts.csv", index=False)

    # Target plots
    months_ts = work["month"].dt.to_timestamp("M")
    test_start_p = pd.Period(test_start, freq="M")
    test_end_p = pd.Period(test_end, freq="M")
    test_mask = (work["month"] >= test_start_p) & (work["month"] <= test_end_p)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(months_ts, y, color="black", linewidth=1.8)
    axes[0].set_title("Target price (y)")
    axes[0].axvspan(
        test_start_p.to_timestamp("M"),
        test_end_p.to_timestamp("M"),
        color="tab:orange",
        alpha=0.15,
        label="test window",
    )
    axes[0].legend(loc="best")
    if y_ret is not None:
        axes[1].plot(months_ts, y_ret, color="tab:blue", linewidth=1.4)
        axes[1].axhline(0.0, color="grey", linewidth=1.0)
        axes[1].set_title("Monthly return (y_return)")
    axes[1].set_xlabel("month")
    fig.tight_layout()
    fig.savefig(out_dir / "target_series.png", dpi=200)
    plt.close(fig)

    # Seasonality
    if "cal_month" in work.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=work["cal_month"], y=y, ax=axes[0], color="tab:blue")
        axes[0].set_title("y by calendar month")
        axes[0].set_xlabel("month_of_year")
        axes[0].set_ylabel("y")
        if y_ret is not None:
            sns.boxplot(x=work["cal_month"], y=y_ret, ax=axes[1], color="tab:orange")
            axes[1].axhline(0.0, color="grey", linewidth=1.0)
            axes[1].set_title("y_return by calendar month")
            axes[1].set_xlabel("month_of_year")
            axes[1].set_ylabel("y_return")
        fig.tight_layout()
        fig.savefig(out_dir / "seasonality_boxplots.png", dpi=200)
        plt.close(fig)

    # Missingness by factor group (only original factors by default)
    factor_cols = [c for c in work.columns if "__" in c]
    groups = sorted({c.split("__", 1)[0] for c in factor_cols})
    group_missing = pd.DataFrame(index=work["month"].astype(str))
    for g in groups:
        cols = [c for c in factor_cols if c.startswith(f"{g}__")]
        if cols:
            # Use numpy to avoid index alignment issues (work has RangeIndex; group_missing uses month strings).
            group_missing[g] = work[cols].isna().mean(axis=1).to_numpy()

    if not group_missing.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(group_missing.T, cmap="magma", vmin=0, vmax=1, ax=ax)
        ax.set_title("Missing rate by factor group (per month)")
        ax.set_xlabel("month")
        ax.set_ylabel("group")
        fig.tight_layout()
        fig.savefig(out_dir / "missingness_heatmap.png", dpi=200)
        plt.close(fig)

        group_missing.mean(axis=0).sort_values(ascending=False).to_frame("missing_rate").reset_index(
            names="group"
        ).to_csv(out_dir / "missingness_by_group.csv", index=False)

    # Correlation with y (features are already lagged in our dataset)
    numeric = work.select_dtypes(include="number").copy()
    if "y" in numeric.columns:
        numeric = numeric.drop(columns=["y"])
    corr_rows = []
    for c in numeric.columns:
        s = pd.to_numeric(numeric[c], errors="coerce")
        if s.notna().sum() < 10:
            continue
        if s.nunique(dropna=True) < 2:
            continue
        corr_rows.append(
            {
                "feature": c,
                "pearson_corr": float(pd.Series(s).corr(y, method="pearson")),
                "spearman_corr": float(pd.Series(s).corr(y, method="spearman")),
                "non_na": int(s.notna().sum()),
            }
        )
    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        corr_df["abs_spearman"] = corr_df["spearman_corr"].abs()
        corr_df = corr_df.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
        corr_df.to_csv(out_dir / "corr_with_y.csv", index=False)

        top = corr_df.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top["feature"], top["spearman_corr"], color="tab:green")
        ax.set_title("Top-20 Spearman correlations with y (lagged features)")
        ax.set_xlabel("spearman_corr")
        fig.tight_layout()
        fig.savefig(out_dir / "top_corr_with_y.png", dpi=200)
        plt.close(fig)

    # Autocorrelation (y and y_return)
    def _acf_vals(series: pd.Series, max_lag: int = 12) -> pd.DataFrame:
        s = pd.to_numeric(series, errors="coerce")
        out = []
        for lag in range(1, max_lag + 1):
            out.append({"lag": lag, "autocorr": float(s.autocorr(lag=lag))})
        return pd.DataFrame(out)

    acf_y = _acf_vals(y)
    acf_y.to_csv(out_dir / "acf_y.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(acf_y["lag"], acf_y["autocorr"], color="tab:blue")
    ax.axhline(0.0, color="grey", linewidth=1.0)
    ax.set_title("ACF of y")
    ax.set_xlabel("lag (months)")
    fig.tight_layout()
    fig.savefig(out_dir / "acf_y.png", dpi=200)
    plt.close(fig)

    if y_ret is not None:
        acf_r = _acf_vals(y_ret)
        acf_r.to_csv(out_dir / "acf_y_return.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(acf_r["lag"], acf_r["autocorr"], color="tab:orange")
        ax.axhline(0.0, color="grey", linewidth=1.0)
        ax.set_title("ACF of y_return")
        ax.set_xlabel("lag (months)")
        fig.tight_layout()
        fig.savefig(out_dir / "acf_y_return.png", dpi=200)
        plt.close(fig)

    # Stationarity tests
    adf = {"y": _adf_test(y)}
    if y_ret is not None:
        adf["y_return"] = _adf_test(y_ret)
    pd.DataFrame(
        [{"var": k, **v} for k, v in adf.items()]
    ).to_csv(out_dir / "adf_tests.csv", index=False)

    # Seasonal decomposition for y (if long enough)
    y_idx = pd.Series(y.to_numpy(), index=months_ts).dropna()
    if len(y_idx) >= 24:
        try:
            dec = seasonal_decompose(y_idx, model="additive", period=12, extrapolate_trend="freq")
        except Exception:
            dec = None
        if dec is not None:
            fig = dec.plot()
            fig.set_size_inches(12, 6)
            fig.tight_layout()
            fig.savefig(out_dir / "seasonal_decompose_y.png", dpi=200)
            plt.close(fig)

    with open(out_dir / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets default
    datasets = list(args.dataset)
    if not datasets:
        for cand in [Path("outputs/datasets/q1_long.csv"), Path("outputs/datasets/q1_with_futures.csv")]:
            if cand.exists():
                datasets.append(cand)

    # 1) Raw profiles + plots
    raw_out = args.output_dir / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)
    prof, raw_tables = profile_raw_files(data_dir=args.data_dir)
    prof.to_csv(raw_out / "raw_profile.csv", index=False)
    prof.to_markdown(raw_out / "raw_profile.md", index=False)
    if not args.no_raw_plots:
        plot_raw_files(raw_tables, out_dir=raw_out / "figures")

    # 2) Monthly feature table (unshifted)
    monthly_out = args.output_dir / "monthly"
    monthly_table = build_monthly_feature_table(data_dir=args.data_dir, include_futures=True)
    monthly_out.mkdir(parents=True, exist_ok=True)
    monthly_table.reset_index(names="month").to_csv(monthly_out / "monthly_feature_table.csv", index=False)
    plot_monthly_coverage(monthly_table, out_dir=monthly_out)
    try:
        lag_corr = lag_correlation_table(
            monthly_table,
            target_col="PP价格__mean",
            max_lag=12,
            method="spearman",
        )
    except Exception:
        lag_corr = pd.DataFrame()
    if not lag_corr.empty:
        lag_corr.to_csv(monthly_out / "lag_corr_spearman.csv", index=False)
        plot_lag_correlation_heatmap(lag_corr, out_dir=monthly_out)

    # 3) Processed dataset EDA
    for ds_path in datasets:
        if not ds_path.exists():
            print("[WARN] dataset not found:", ds_path)
            continue
        df = pd.read_csv(ds_path)
        tag = ds_path.stem + ("__engineered" if args.engineer_features else "")
        out_dir = args.output_dir / tag
        dataset_eda(
            df,
            name=ds_path.name,
            out_dir=out_dir,
            test_start=args.test_start,
            test_end=args.test_end,
            engineer_features=args.engineer_features,
        )

    print("[OK] wrote EDA outputs to:", args.output_dir)


if __name__ == "__main__":
    main()
