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
from pp_forecast.dataset import FILE_SPECS


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
        help="Path to a processed dataset CSV (repeatable). Default: outputs/datasets/pp_base.csv & pp_with_futures.csv if exist.",
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
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


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

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax = axes[0]
        ax.plot(s.index, s.values, linewidth=1.0, alpha=0.5, color="tab:blue", label="原始")

        # overlay monthly mean (smoothed)
        monthly = s.groupby(s.index.to_period("M")).mean()
        ax.plot(
            monthly.index.to_timestamp("M"),
            monthly.values,
            linewidth=2.0,
            color="tab:red",
            label="月度均值",
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
                    label="月度最小/最大范围",
                )

        ax.set_title(f"{factor}（原始序列 + 月度平滑）")
        ax.set_xlabel("日期")
        ax.set_ylabel("数值")
        ax.legend(loc="best")

        ax2 = axes[1]
        sns.histplot(s.values, bins=40, kde=True, ax=ax2, color="tab:blue")
        ax2.set_title(f"{factor}分布")
        ax2.set_xlabel("数值")
        ax2.set_ylabel("频数")

        fig.tight_layout()
        fig.savefig(
            out_dir / f"raw_{_slug(factor)}.pdf",
            bbox_inches="tight",
            pad_inches=0.06,
        )
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
    # GDP is annual; forward-fill to subsequent months so monthly EDA reflects the modeling feature.
    gdp_cols = [c for c in table.columns if c.startswith("GDP__")]
    if gdp_cols:
        table[gdp_cols] = table[gdp_cols].ffill()
    table.index = table.index.astype("period[M]")
    return table


def plot_monthly_coverage(monthly_table: pd.DataFrame, *, out_dir: Path) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import BoundaryNorm, ListedColormap

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()

    # For modeling we use x(t-1) -> y(t). Here we visualize "feature availability" for each
    # prediction month t, based on whether the previous month value exists.
    X = _monthly_factor_mean_table(monthly_table)
    if X.empty:
        return
    if "PP价格" not in X.columns:
        return

    y = pd.to_numeric(X["PP价格"], errors="coerce")
    y_avail = y.dropna()
    if y_avail.empty:
        return

    pred_start = y_avail.index.min() + 1  # first month that can be predicted
    pred_end = y_avail.index.max()
    pred_months = pd.period_range(pred_start, pred_end, freq="M")
    total_months = int(len(pred_months))

    # Binary availability matrix: rows=factors, cols=prediction months t.
    availability = pd.DataFrame(index=pred_months.astype(str))
    summary_rows = []
    for factor in X.columns:
        s = pd.to_numeric(X[factor], errors="coerce").shift(1).reindex(pred_months)
        ok = s.notna()
        availability[factor] = ok.astype(int)

        first = str(s.dropna().index.min()) if ok.any() else ""
        last = str(s.dropna().index.max()) if ok.any() else ""
        miss = int((~ok).sum())
        summary_rows.append(
            {
                "factor": factor,
                "first_usable": first,
                "last_usable": last,
                "available_months": int(ok.sum()),
                "missing_months": miss,
                "missing_rate": float(miss / total_months) if total_months else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "monthly_coverage_summary.csv", index=False)

    # Write a LaTeX table for the report.
    def _pct(x: float) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{100.0 * x:.1f}\\%"

    lines = []
    lines.append("% Auto-generated by scripts/run_pp_eda.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(
        f"\\caption{{建模窗口的月度特征可用性汇总（预测月 $t$ 使用 $t-1$ 月特征；共 {total_months} 个月）}}"
    )
    lines.append("\\label{tab:monthly-coverage-summary}")
    lines.append("\\small")
    lines.append("\\begin{TableFit}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("因子 & 首次可用月 & 最后可用月 & 可用(月) & 缺失(月) & 缺失率 \\\\")
    lines.append("\\midrule")
    for _, r in summary.iterrows():
        lines.append(
            f"{r['factor']} & {r['first_usable']} & {r['last_usable']} & "
            f"{int(r['available_months'])} & {int(r['missing_months'])} & {_pct(float(r['missing_rate']))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{TableFit}")
    lines.append("\\end{table}")
    (out_dir / "monthly_coverage_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Heatmap (make it explicit and readable).
    fig, ax = plt.subplots(figsize=(12.5, max(4.8, 0.33 * len(availability.columns))))
    cmap = ListedColormap(["#f7f7f7", "#2c7fb8"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], ncolors=cmap.N)
    xlabels = [str(m) if m.month == 1 else "" for m in pred_months]
    hm = sns.heatmap(
        availability.T,
        cmap=cmap,
        norm=norm,
        cbar=True,
        cbar_kws={"ticks": [0, 1], "shrink": 0.8},
        xticklabels=xlabels,
        yticklabels=True,
        linewidths=0.0,
        ax=ax,
    )
    cbar = hm.collections[0].colorbar
    cbar.set_ticklabels(["缺失", "可用"])
    cbar.set_label("特征可用性", rotation=270, labelpad=12)
    ax.set_title("月度特征覆盖（建模口径：X(t-1) → y(t)；按因子均值列判断）")
    ax.set_xlabel("预测月份 t（仅标注每年1月）")
    ax.set_ylabel("因子")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_coverage_heatmap.pdf")
    plt.close(fig)

    # Coverage counts (in the same modeling window).
    counts = availability.sum(axis=0).sort_values()
    fig_h = max(4.2, 0.28 * len(counts) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    counts.plot(kind="barh", ax=ax, color="tab:blue")
    ax.set_title(f"各因子在建模窗口的可用月份数（共 {total_months} 个月；特征使用 t-1 月）")
    ax.set_xlabel("可用月份数")
    ax.set_ylabel("因子")
    ax.set_xlim(0, total_months)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "monthly_coverage_counts.pdf")
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
    mat = mat.rename(columns={c: c.replace("lag_", "") for c in mat.columns})

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(mat))))
    sns.heatmap(mat, cmap="coolwarm", center=0.0, ax=ax)
    ax.set_title("滞后相关热力图（因子均值与 PP价格均值）")
    ax.set_xlabel("滞后（月）k：因子(t-k) 与 价格(t)")
    ax.set_ylabel("因子")
    fig.tight_layout()
    fig.savefig(out_dir / "lag_corr_heatmap.pdf")
    plt.close(fig)


def _ordered_factors_for_monthly(monthly_table: pd.DataFrame) -> list[str]:
    factors = sorted({c.split("__", 1)[0] for c in monthly_table.columns if c.endswith("__mean")})
    preferred = [
        "PP价格",
        "期货价格",
        # costs
        "丙烯成本",
        "PDH成本",
        "乙烯成本",
        "MTO成本",
        "CTO成本",
        # supply / structure
        "PP产量",
        "PP进口量",
        "检修损失",
        "排产比例",
        # inventory
        "PP石化库存",
        # downstream
        "塑编开工率",
        "PP开工率",
        "BOPP开工率",
        # macro
        "GDP",
    ]
    ordered = [f for f in preferred if f in factors]
    ordered += [f for f in factors if f not in set(ordered)]
    return ordered


def _monthly_factor_mean_table(monthly_table: pd.DataFrame) -> pd.DataFrame:
    factors = _ordered_factors_for_monthly(monthly_table)
    cols = {}
    for f in factors:
        c = f"{f}__mean"
        if c in monthly_table.columns:
            cols[f] = pd.to_numeric(monthly_table[c], errors="coerce")
    return pd.DataFrame(cols, index=monthly_table.index).sort_index()


def factor_correlation_outputs(monthly_table: pd.DataFrame, *, out_dir: Path) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()

    X = _monthly_factor_mean_table(monthly_table)
    if X.empty:
        return

    # Pairwise overlap counts (months where both series are available).
    mask = X.notna().astype(int)
    overlap = (mask.T @ mask).astype(int)

    corr = X.corr(method="spearman", min_periods=12)
    corr.to_csv(out_dir / "factor_corr_spearman.csv")
    overlap.to_csv(out_dir / "factor_corr_overlap_months.csv")

    # Top highly-correlated pairs (helps identify redundancy / collinearity).
    rows = []
    factors = list(corr.columns)
    for i, a in enumerate(factors):
        for j in range(i + 1, len(factors)):
            b = factors[j]
            c = float(corr.loc[a, b]) if pd.notna(corr.loc[a, b]) else np.nan
            n = int(overlap.loc[a, b]) if (a in overlap.index and b in overlap.columns) else 0
            if np.isnan(c) or n < 36:
                continue
            rows.append({"a": a, "b": b, "spearman": c, "overlap_months": n, "abs": abs(c)})
    pairs = pd.DataFrame(rows).sort_values("abs", ascending=False).drop(columns=["abs"])
    pairs.to_csv(out_dir / "factor_corr_pairs_top.csv", index=False)

    # Heatmap (show lower-triangle to reduce clutter).
    mat = corr.copy()
    tri_mask = np.triu(np.ones_like(mat, dtype=bool), k=1) | mat.isna().to_numpy()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        mat,
        mask=tri_mask,
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.85},
        ax=ax,
    )
    ax.set_title("因子月度均值的相关矩阵（Spearman）")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "factor_corr_heatmap_spearman.pdf")
    plt.close(fig)


def _rolling_spearman_corr(
    x: pd.Series,
    y: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    idx = x.index
    out = pd.Series(index=idx, dtype=float)
    n = len(idx)
    for i in range(n):
        start = max(0, i - window + 1)
        win = pd.concat([x.iloc[start : i + 1], y.iloc[start : i + 1]], axis=1).dropna()
        if len(win) < min_periods:
            out.iloc[i] = np.nan
            continue
        xr = win.iloc[:, 0].rank()
        yr = win.iloc[:, 1].rank()
        out.iloc[i] = float(xr.corr(yr))
    return out


def rolling_lag_correlation_outputs(
    monthly_table: pd.DataFrame,
    *,
    target_col: str,
    factors: list[str],
    out_dir: Path,
    window: int = 24,
    min_periods: int = 18,
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()

    if target_col not in monthly_table.columns:
        return

    y = pd.to_numeric(monthly_table[target_col], errors="coerce")
    if y.dropna().empty:
        return

    results = {}
    for f in factors:
        col = f"{f}__mean"
        if col not in monthly_table.columns:
            continue
        x = pd.to_numeric(monthly_table[col], errors="coerce").shift(1)
        results[f] = _rolling_spearman_corr(x, y, window=window, min_periods=min_periods)

    if not results:
        return

    df = pd.DataFrame(results)
    df.index = df.index.astype("period[M]")
    df.to_csv(out_dir / "rolling_lag1_spearman.csv", index=True)

    # Plot in small multiples (avoids cluttered multi-line plot).
    names = list(df.columns)
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3.0, 2.6 * nrows)), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    axes = axes.reshape(nrows, ncols)

    months_ts = df.index.to_timestamp("M")
    for k, name in enumerate(names):
        r = k // ncols
        c = k % ncols
        ax = axes[r, c]
        ax.plot(months_ts, df[name], color="tab:blue", linewidth=1.5)
        ax.axhline(0.0, color="grey", linewidth=1.0)
        ax.set_title(f"{name}：rolling Spearman corr(x(t-1), y(t))")
        ax.set_ylabel("相关系数")
        ax.grid(True, alpha=0.25)

    # Hide unused panels
    for k in range(n, nrows * ncols):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("月份")

    fig.suptitle(f"滚动滞后相关（窗口={window}月，min={min_periods}）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_lag1_spearman.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def factor_stationarity_outputs(monthly_table: pd.DataFrame, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X = _monthly_factor_mean_table(monthly_table)
    if X.empty:
        return

    rows = []
    for f in X.columns:
        s = X[f]
        level = _adf_test(s)
        diff = _adf_test(s.diff())
        rows.append(
            {
                "factor": f,
                "adf_p_level": level["pvalue"],
                "adf_p_diff": diff["pvalue"],
                "nobs_level": level["nobs"],
                "nobs_diff": diff["nobs"],
            }
        )

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "adf_factors.csv", index=False)

    def _fmt_p(x: float) -> str:
        if not np.isfinite(x):
            return "--"
        if x < 0.001:
            return "<0.001"
        return f"{x:.3f}"

    def _fmt_n(x: float) -> str:
        if not np.isfinite(x):
            return "--"
        return str(int(x))

    lines = []
    lines.append("% Auto-generated by scripts/run_pp_eda.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{月度因子平稳性检验（ADF；水平值 vs 一阶差分）}")
    lines.append("\\label{tab:adf-factors}")
    lines.append("\\small")
    lines.append("\\begin{TableFit}")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("因子 & ADF $p$（水平） & ADF $p$（差分） & 样本数（月） & 差分样本数（月）\\\\")
    lines.append("\\midrule")
    for _, r in res.iterrows():
        lines.append(
            f"{r['factor']} & {_fmt_p(float(r['adf_p_level']))} & {_fmt_p(float(r['adf_p_diff']))} & "
            f"{_fmt_n(float(r['nobs_level']))} & {_fmt_n(float(r['nobs_diff']))}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{TableFit}")
    lines.append("\\end{table}")
    (out_dir / "adf_factors.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        direction_counts.rename_axis("y_direction").to_frame("count").reset_index().to_csv(
            out_dir / "direction_counts.csv", index=False
        )

        # Plot class balance for direction (up/down).
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        labels = ["跌/不涨(0)", "涨(1)"]
        vals = [int(direction_counts.get(0, 0)), int(direction_counts.get(1, 0))]
        ax.bar(labels, vals, color=["tab:gray", "tab:green"])
        ax.set_title("方向标签分布（y_direction）")
        ax.set_ylabel("样本数（月）")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "direction_counts.pdf")
        plt.close(fig)

    if "y_strength" in work.columns:
        strength_counts = work["y_strength"].astype(str).value_counts().rename_axis("y_strength")
        strength_counts.to_frame("count").reset_index().to_csv(out_dir / "strength_counts.csv", index=False)

        # Plot class balance for 5-class strength label.
        fig, ax = plt.subplots(figsize=(8.5, 3.6))
        order = ["big_down", "small_down", "flat", "small_up", "big_up"]
        vals = [int(strength_counts.get(k, 0)) for k in order]
        ax.bar(order, vals, color="tab:blue")
        ax.set_title("强度标签分布（y_strength；五分类）")
        ax.set_ylabel("样本数（月）")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "strength_counts.pdf")
        plt.close(fig)

    # Target plots
    months_ts = work["month"].dt.to_timestamp("M")
    test_start_p = pd.Period(test_start, freq="M")
    test_end_p = pd.Period(test_end, freq="M")
    test_mask = (work["month"] >= test_start_p) & (work["month"] <= test_end_p)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(months_ts, y, color="black", linewidth=1.8)
    axes[0].set_title("目标价格（y）")
    axes[0].axvspan(
        test_start_p.to_timestamp("M"),
        test_end_p.to_timestamp("M"),
        color="tab:orange",
        alpha=0.15,
        label="测试窗口",
    )
    axes[0].legend(loc="best")
    if y_ret is not None:
        axes[1].plot(months_ts, y_ret, color="tab:blue", linewidth=1.4)
        axes[1].axhline(0.0, color="grey", linewidth=1.0)
        axes[1].set_title("月度收益率（y_return）")
    axes[1].set_xlabel("月份")
    fig.tight_layout()
    fig.savefig(out_dir / "target_series.pdf")
    plt.close(fig)

    # Distribution of returns (heavy tails / imbalance-friendly view).
    if y_ret is not None:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        sns.histplot(y_ret.dropna(), bins=28, kde=True, ax=ax, color="tab:blue")
        ax.axvline(0.0, color="grey", linewidth=1.0)
        ax.set_title("月度收益率分布（y_return）")
        ax.set_xlabel("收益率")
        ax.set_ylabel("频数")
        fig.tight_layout()
        fig.savefig(out_dir / "return_distribution.pdf")
        plt.close(fig)

    # Seasonality
    if "cal_month" in work.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=work["cal_month"], y=y, ax=axes[0], color="tab:blue")
        axes[0].set_title("y 按自然月分布")
        axes[0].set_xlabel("月份（1-12）")
        axes[0].set_ylabel("y")
        if y_ret is not None:
            sns.boxplot(x=work["cal_month"], y=y_ret, ax=axes[1], color="tab:orange")
            axes[1].axhline(0.0, color="grey", linewidth=1.0)
            axes[1].set_title("y_return 按自然月分布")
            axes[1].set_xlabel("月份（1-12）")
            axes[1].set_ylabel("y_return")
        fig.tight_layout()
        fig.savefig(out_dir / "seasonality_boxplots.pdf")
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
        ax.set_title("各因子组逐月缺失率")
        ax.set_xlabel("月份")
        ax.set_ylabel("因子组")
        fig.tight_layout()
        fig.savefig(out_dir / "missingness_heatmap.pdf")
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
        ax.set_title("与 y 的斯皮尔曼相关（前20，已滞后特征）")
        ax.set_xlabel("斯皮尔曼相关系数")
        fig.tight_layout()
        fig.savefig(out_dir / "top_corr_with_y.pdf")
        plt.close(fig)

    # Correlation with y_return (更贴近方向/强度任务)
    if y_ret is not None:
        numeric_ret = work.select_dtypes(include="number").copy()
        drop_cols = [c for c in ["y", "y_prev", "y_direction", "y_return"] if c in numeric_ret.columns]
        if drop_cols:
            numeric_ret = numeric_ret.drop(columns=drop_cols)

        corr_rows = []
        for c in numeric_ret.columns:
            s = pd.to_numeric(numeric_ret[c], errors="coerce")
            if s.notna().sum() < 10:
                continue
            if s.nunique(dropna=True) < 2:
                continue
            corr_rows.append(
                {
                    "feature": c,
                    "pearson_corr": float(pd.Series(s).corr(y_ret, method="pearson")),
                    "spearman_corr": float(pd.Series(s).corr(y_ret, method="spearman")),
                    "non_na": int(s.notna().sum()),
                }
            )

        corr_ret = pd.DataFrame(corr_rows)
        if not corr_ret.empty:
            corr_ret["abs_spearman"] = corr_ret["spearman_corr"].abs()
            corr_ret = corr_ret.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
            corr_ret.to_csv(out_dir / "corr_with_y_return.csv", index=False)

            top = corr_ret.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top["feature"], top["spearman_corr"], color="tab:purple")
            ax.set_title("与 y_return 的斯皮尔曼相关（前20，已滞后特征）")
            ax.set_xlabel("斯皮尔曼相关系数")
            fig.tight_layout()
            fig.savefig(out_dir / "top_corr_with_y_return.pdf")
            plt.close(fig)

    # Feature engineering composition (counts by family)
    def _fe_family(col: str) -> str:
        if col in {"cal_time_index", "cal_month_sin", "cal_month_cos"}:
            return "日历/趋势"
        if col.startswith("价差__"):
            return "价差/基差"
        if col.startswith("比值__"):
            return "比值"
        if col.startswith("供需__"):
            return "供需比值"
        if col.startswith("指数__"):
            return "指数"
        if "_mom_" in col:
            return "动量(pct_change)"
        if "_roll_mean_" in col:
            return "滚动均值"
        if "_roll_std_" in col:
            return "滚动波动"
        if any(k in col for k in ["range_over_mean", "last_minus_mean", "max_minus_min"]):
            return "月内形态"
        if "__" in col:
            # treat plain monthly stats as base features
            parts = col.split("__")
            if len(parts) == 2 and parts[1] in {"mean", "min", "max", "last", "range"}:
                return "原始月度口径"
            return "其他(含派生)"
        return "其他"

    fe_cols = [c for c in work.columns if ("__" in c) or c in {"cal_time_index", "cal_month_sin", "cal_month_cos"}]
    if fe_cols:
        fe_count = (
            pd.Series([_fe_family(c) for c in fe_cols])
            .value_counts()
            .rename_axis("family")
            .to_frame("n_features")
            .reset_index()
        )
        fe_count.to_csv(out_dir / "feature_engineering_counts.csv", index=False)

        # Plot when the dataset contains engineered-feature families (or when explicitly enabled).
        has_engineered = engineer_features or any(
            (
                c.startswith(("价差__", "比值__", "供需__", "指数__"))
                or ("_mom_" in c)
                or ("_roll_mean_" in c)
                or ("_roll_std_" in c)
                or any(k in c for k in ["range_over_mean", "last_minus_mean", "max_minus_min"])
                or c in {"cal_time_index", "cal_month_sin", "cal_month_cos"}
            )
            for c in fe_cols
        )
        if has_engineered:
            fig_h = max(3.6, 0.35 * len(fe_count) + 1.8)
            fig, ax = plt.subplots(figsize=(9.5, fig_h))
            order = fe_count.sort_values("n_features", ascending=True)
            ax.barh(order["family"], order["n_features"], color="#2ca25f")
            ax.set_title("派生特征工程：特征族数量统计")
            ax.set_xlabel("特征数")
            ax.grid(True, axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "feature_engineering_counts.pdf")
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
    ax.set_title("y 的自相关")
    ax.set_xlabel("滞后（月）")
    fig.tight_layout()
    fig.savefig(out_dir / "acf_y.pdf")
    plt.close(fig)

    if y_ret is not None:
        acf_r = _acf_vals(y_ret)
        acf_r.to_csv(out_dir / "acf_y_return.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(acf_r["lag"], acf_r["autocorr"], color="tab:orange")
        ax.axhline(0.0, color="grey", linewidth=1.0)
        ax.set_title("y_return 的自相关")
        ax.set_xlabel("滞后（月）")
        fig.tight_layout()
        fig.savefig(out_dir / "acf_y_return.pdf")
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
            for ax, label in zip(fig.axes, ["原序列", "趋势", "季节项", "残差"]):
                ax.set_ylabel(label)
            fig.suptitle("y 的季节分解", fontsize=12)
            if fig.axes:
                fig.axes[-1].set_xlabel("月份")
            fig.tight_layout()
            fig.savefig(out_dir / "seasonal_decompose_y.pdf")
            plt.close(fig)

    with open(out_dir / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Datasets default
    datasets = list(args.dataset)
    if not datasets:
        for cand in [
            Path("outputs/datasets/pp_base.csv"),
            Path("outputs/datasets/pp_with_futures.csv"),
        ]:
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

    # Extra monthly EDA (model-relevant): collinearity, rolling lag-corr, stationarity.
    factor_correlation_outputs(monthly_table, out_dir=monthly_out)
    rolling_lag_correlation_outputs(
        monthly_table,
        target_col="PP价格__mean",
        factors=["丙烯成本", "PP石化库存", "塑编开工率", "PP开工率", "PP进口量", "PP产量"],
        out_dir=monthly_out,
        window=24,
        min_periods=18,
    )
    factor_stationarity_outputs(monthly_table, out_dir=monthly_out)

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
