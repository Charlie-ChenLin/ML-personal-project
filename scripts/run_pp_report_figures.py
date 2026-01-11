from __future__ import annotations

import argparse
import os
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a small set of report figures from outputs/*.")
    p.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--q3-dir", type=Path, default=Path("outputs/q3"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/report_figures"))
    p.add_argument("--test-start", default="2021-01")
    p.add_argument("--test-end", default="2021-07")
    return p.parse_args()


def _ensure_mpl() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".mplconfig").resolve()))


def _configure_matplotlib_fonts() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Arial Unicode MS",
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
        plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.astype(str), freq="M")


def q1_best_models_overlay(
    *,
    metrics_dir: Path,
    out_dir: Path,
    test_start: str,
    test_end: str,
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)

    schemes = [
        ("pp_base", "不含期货"),
        ("pp_base_engineered", "不含期货+特征工程"),
        ("pp_with_futures", "含期货(restrict)"),
        ("pp_with_futures_engineered", "含期货+特征工程(restrict)"),
    ]

    merged: pd.DataFrame | None = None
    legends: list[str] = []
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for i, (stem, label) in enumerate(schemes):
        m_path = metrics_dir / stem / "pp_model_metrics.csv"
        p_path = metrics_dir / stem / "pp_test_predictions.csv"
        if not m_path.exists() or not p_path.exists():
            continue

        metrics = pd.read_csv(m_path).dropna(subset=["RMSE"]).copy()
        if metrics.empty:
            continue
        best = metrics.loc[metrics["RMSE"].idxmin()]
        best_model = str(best["model"])

        preds = pd.read_csv(p_path)
        preds = preds[preds["model"] == best_model].copy()
        if preds.empty:
            continue

        preds["month"] = _to_period(preds["month"])
        preds = preds.sort_values("month")
        preds = preds[(preds["month"] >= pd.Period(test_start, "M")) & (preds["month"] <= pd.Period(test_end, "M"))]

        keep = preds[["month", "y_true", "y_pred"]].rename(columns={"y_pred": f"y_pred__{stem}"})
        if merged is None:
            merged = keep
        else:
            merged = pd.merge(merged, keep[["month", f"y_pred__{stem}"]], on="month", how="outer")

        legends.append(f"{label}: {best_model}")

    if merged is None or merged.empty:
        return

    merged = merged.sort_values("month").reset_index(drop=True)
    x = merged["month"].dt.to_timestamp("M")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(x, merged["y_true"], color="black", linewidth=2.0, marker="o", label="真实值 y")

    for i, (stem, _) in enumerate(schemes):
        col = f"y_pred__{stem}"
        if col not in merged.columns:
            continue
        ax.plot(
            x,
            merged[col],
            linewidth=1.8,
            marker="o",
            color=colors[i % len(colors)],
            label=legends[i] if i < len(legends) else stem,
            alpha=0.95,
        )

    ax.set_title(f"问题1：测试窗口({test_start}~{test_end})最佳模型预测对比（每种方案取 RMSE 最低）")
    ax.set_xlabel("月份")
    ax.set_ylabel("PP 价格")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "q1_test_predictions_best_models.pdf")
    plt.close(fig)


def q2_strength_probabilities(
    *,
    metrics_dir: Path,
    out_dir: Path,
    dataset_stem: str = "pp_base_engineered",
    test_start: str,
    test_end: str,
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)

    m_path = metrics_dir / dataset_stem / "pp_strength_model_metrics.csv"
    p_path = metrics_dir / dataset_stem / "pp_strength_test_predictions.csv"
    if not m_path.exists() or not p_path.exists():
        return

    metrics = pd.read_csv(m_path).dropna(subset=["Strength_F1_macro"]).copy()
    if metrics.empty:
        return
    best = metrics.loc[metrics["Strength_F1_macro"].idxmax()]
    best_model = str(best["model"])

    preds = pd.read_csv(p_path)
    preds = preds[preds["model"] == best_model].copy()
    if preds.empty:
        return

    preds["month"] = _to_period(preds["month"])
    preds = preds.sort_values("month")
    preds = preds[(preds["month"] >= pd.Period(test_start, "M")) & (preds["month"] <= pd.Period(test_end, "M"))]

    proba_cols = [c for c in preds.columns if c.startswith("proba__")]
    if not proba_cols:
        return

    # Keep a stable order for stacked bars.
    cls_order = ["big_down", "small_down", "flat", "small_up", "big_up"]
    cols_ordered = [f"proba__{c}" for c in cls_order if f"proba__{c}" in proba_cols]
    if not cols_ordered:
        cols_ordered = sorted(proba_cols)

    x = np.arange(len(preds))
    bottom = np.zeros(len(preds), dtype=float)
    colors = ["#b2182b", "#ef8a62", "#f7f7f7", "#67a9cf", "#2166ac"]

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, c in enumerate(cols_ordered):
        vals = preds[c].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=c.replace("proba__", ""), color=colors[i % len(colors)])
        bottom = bottom + vals

    months = preds["month"].astype(str).to_list()
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("预测概率")
    ax.set_xlabel("月份")
    ax.set_title(f"问题2：强度五分类概率输出（{dataset_stem}；最佳模型={best_model}）")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    # Annotate true label for each month (small font to keep compact).
    if "strength_true" in preds.columns:
        for i, lab in enumerate(preds["strength_true"].astype(str).to_list()):
            ax.text(i, 1.02, lab, ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(out_dir / "q2_strength_probabilities_best_model.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def q3_group_weights_heatmap(*, q3_dir: Path, out_dir: Path, dataset_stem: str = "pp_base") -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)

    path = q3_dir / dataset_stem / "q3_ridge_group_weights.csv"
    if not path.exists():
        return

    w = pd.read_csv(path)
    if w.empty:
        return

    pivot = (
        w.pivot_table(index="group", columns="stage_id", values="group_abs_weight", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    # Order groups by overall importance.
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(4.0, 0.28 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    sns.heatmap(pivot, cmap="YlGnBu", annot=False, ax=ax)
    ax.set_title(f"问题3：阶段内因子组权重热力图（Ridge 绝对系数归一化；{dataset_stem}）")
    ax.set_xlabel("阶段 stage_id")
    ax.set_ylabel("因子组")
    fig.tight_layout()
    fig.savefig(out_dir / "q3_ridge_group_weights_heatmap.pdf")
    plt.close(fig)


def q3_strength_group_weights_heatmap(
    *, q3_dir: Path, out_dir: Path, dataset_stem: str = "pp_base"
) -> None:
    _ensure_mpl()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    _configure_matplotlib_fonts()
    out_dir.mkdir(parents=True, exist_ok=True)

    path = q3_dir / dataset_stem / "q3_strength_logreg_group_weights.csv"
    if not path.exists():
        return

    w = pd.read_csv(path)
    if w.empty:
        return

    pivot = (
        w.pivot_table(index="group", columns="stage_id", values="group_weight", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(4.0, 0.28 * len(pivot) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    sns.heatmap(pivot, cmap="PuBuGn", annot=False, ax=ax)
    ax.set_title(f"问题3（强度任务）：阶段内因子组权重热力图（LogReg 系数幅度归一化；{dataset_stem}）")
    ax.set_xlabel("阶段 stage_id")
    ax.set_ylabel("因子组")
    fig.tight_layout()
    fig.savefig(out_dir / "q3_strength_logreg_group_weights_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q1_best_models_overlay(
        metrics_dir=args.metrics_dir,
        out_dir=args.output_dir,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    q2_strength_probabilities(
        metrics_dir=args.metrics_dir,
        out_dir=args.output_dir,
        dataset_stem="pp_base_engineered",
        test_start=args.test_start,
        test_end=args.test_end,
    )
    q3_group_weights_heatmap(q3_dir=args.q3_dir, out_dir=args.output_dir, dataset_stem="pp_base")
    q3_strength_group_weights_heatmap(q3_dir=args.q3_dir, out_dir=args.output_dir, dataset_stem="pp_base")
    print("[OK] wrote report figures to:", args.output_dir)


if __name__ == "__main__":
    main()
