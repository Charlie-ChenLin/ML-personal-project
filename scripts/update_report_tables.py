from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update LaTeX report tables from outputs/* CSVs.")
    p.add_argument("--metrics-dir", type=Path, default=Path("outputs/metrics"))
    p.add_argument("--q3-dir", type=Path, default=Path("outputs/q3"))
    p.add_argument("--report-tables-dir", type=Path, default=Path("report/tables"))
    p.add_argument("--test-start", default="2021-01")
    p.add_argument("--test-end", default="2021-07")
    return p.parse_args()


def _tex_escape(s: str) -> str:
    # Minimal escaping for our tables.
    # Insert break opportunities after underscores so identifiers (dataset/model names)
    # can wrap inside p{...} columns, avoiding overlap in appendix longtables.
    return s.replace("_", "\\_\\allowbreak ")


def _fmt(x: float, *, digits: int) -> str:
    return f"{x:.{digits}f}"


def write_q1_table(*, metrics_dir: Path, out_path: Path, test_start: str, test_end: str) -> None:
    rows_spec = [
        ("pp_base", "不含期货"),
        ("pp_base_engineered", "不含期货+特征工程"),
        ("pp_with_futures", "含期货(restrict)"),
        ("pp_with_futures_engineered", "含期货+特征工程(restrict)"),
    ]

    rows = []
    for stem, scheme in rows_spec:
        path = metrics_dir / stem / "pp_model_metrics.csv"
        df = pd.read_csv(path)
        df = df.dropna(subset=["RMSE"]).copy()
        best = df.loc[df["RMSE"].idxmin()]
        rows.append(
            {
                "dataset": stem,
                "scheme": scheme,
                "model": str(best["model"]),
                "MAE": float(best["MAE"]),
                "RMSE": float(best["RMSE"]),
                "MAPE_pct": 100.0 * float(best["MAPE"]),
                "Acc": float(best["Accuracy"]),
                "Prec": float(best["Precision"]),
                "Recall": float(best["Recall"]),
            }
        )

    lines: list[str] = []
    lines.append(f"% Summary from outputs/metrics/*/pp_model_metrics.csv (test window: {test_start}~{test_end})")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题1：价格点预测与涨跌方向（最佳模型摘要，按 RMSE 选取）}")
    lines.append("\\label{tab:q1-summary}")
    lines.append("\\small")
    lines.append("\\begin{TableFit}")
    lines.append("\\begin{tabular}{llllrrrr}")
    lines.append("\\toprule")
    lines.append("数据集 & 方案 & 最佳模型 & MAE & RMSE & MAPE(\\%) & Acc & Precision/Recall \\\\")
    lines.append("\\midrule")

    for r in rows:
        lines.append(
            f"{_tex_escape(r['dataset'])} & {r['scheme']} & {_tex_escape(r['model'])} & "
            f"{_fmt(r['MAE'], digits=2)} & {_fmt(r['RMSE'], digits=2)} & {_fmt(r['MAPE_pct'], digits=2)} & "
            f"{_fmt(r['Acc'], digits=3)} & {_fmt(r['Prec'], digits=3)} / {_fmt(r['Recall'], digits=3)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{TableFit}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_q2_table(*, metrics_dir: Path, out_path: Path, test_start: str, test_end: str) -> None:
    rows_spec = [
        ("pp_base", "不含期货"),
        ("pp_base_engineered", "不含期货+特征工程"),
        ("pp_with_futures", "含期货(restrict)"),
        ("pp_with_futures_engineered", "含期货+特征工程(restrict)"),
    ]

    rows = []
    for stem, scheme in rows_spec:
        path = metrics_dir / stem / "pp_strength_model_metrics.csv"
        df = pd.read_csv(path)
        df = df.dropna(subset=["Strength_F1_macro"]).copy()
        best = df.loc[df["Strength_F1_macro"].idxmax()]
        rows.append(
            {
                "dataset": stem,
                "scheme": scheme,
                "model": str(best["model"]),
                "F1": float(best["Strength_F1_macro"]),
                "Acc": float(best["Strength_Accuracy"]),
                "Prec": float(best["Strength_Precision_macro"]),
                "Recall": float(best["Strength_Recall_macro"]),
            }
        )

    lines: list[str] = []
    lines.append(
        f"% Summary from outputs/metrics/*/pp_strength_model_metrics.csv (test window: {test_start}~{test_end})"
    )
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题2：涨跌强度（五分类）与概率输出（最佳模型摘要，按 macro-F1 选取）}")
    lines.append("\\label{tab:q2-summary}")
    lines.append("\\small")
    lines.append("\\begin{TableFit}")
    lines.append("\\begin{tabular}{llllrrrr}")
    lines.append("\\toprule")
    lines.append("数据集 & 方案 & 最佳模型 & F1\\_macro & Acc & Precision\\_macro & Recall\\_macro \\\\")
    lines.append("\\midrule")

    for r in rows:
        lines.append(
            f"{_tex_escape(r['dataset'])} & {r['scheme']} & {_tex_escape(r['model'])} & "
            f"{_fmt(r['F1'], digits=3)} & {_fmt(r['Acc'], digits=3)} & {_fmt(r['Prec'], digits=3)} & {_fmt(r['Recall'], digits=3)} \\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{TableFit}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_q3_stage_tables(*, q3_dir: Path, report_tables_dir: Path) -> None:
    stage = pd.read_csv(q3_dir / "pp_base" / "q3_stage_summary.csv")
    weights = pd.read_csv(q3_dir / "pp_base" / "q3_ridge_group_weights.csv")
    strength_weights_path = q3_dir / "pp_base" / "q3_strength_logreg_group_weights.csv"
    strength_weights = (
        pd.read_csv(strength_weights_path) if strength_weights_path.exists() else pd.DataFrame()
    )

    # Stage summary table
    lines: list[str] = []
    lines.append("% Stage summary from outputs/q3/pp_base/q3_stage_summary.csv")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{问题3：阶段划分结果摘要（以 pp\\_base 为例）}")
    lines.append("\\label{tab:q3-stage-summary}")
    lines.append("\\small")
    lines.append("\\begin{TableFit}")
    lines.append("\\begin{tabular}{rllrrr}")
    lines.append("\\toprule")
    lines.append("阶段 & 市场状态(regime) & 起止月份 & 月数 & 累计涨跌(\\%) & 波动(\\%/月) \\\\")
    lines.append("\\midrule")
    for _, r in stage.iterrows():
        lines.append(
            f"{int(r['stage_id'])} & {_tex_escape(str(r['regime']))} & "
            f"{r['start_month']} $\\sim$ {r['end_month']} & {int(r['n_months'])}  & "
            f"{_fmt(100.0 * float(r['cum_return']), digits=2)} & "
            f"{_fmt(100.0 * float(r['vol_return']), digits=2)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{TableFit}")
    lines.append("\\end{table}")
    (report_tables_dir / "q3_stage_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Top factor groups table (Top-5 by weight per stage)
    stage_regime = stage.set_index("stage_id")["regime"].to_dict()
    out_lines: list[str] = []
    out_lines.append("% Top factor groups by Ridge absolute coefficient weight (pp_base)")
    out_lines.append("\\begin{table}[H]")
    out_lines.append("\\centering")
    out_lines.append("\\caption{问题3：各阶段关键因子组（Top-5，Ridge 绝对系数归一化权重，pp\\_base）}")
    out_lines.append("\\label{tab:q3-top-groups}")
    out_lines.append("\\small")
    out_lines.append("\\begin{TableFit}")
    out_lines.append("\\begin{tabular}{rll}")
    out_lines.append("\\toprule")
    out_lines.append("阶段 & regime & Top-5 因子组（权重） \\\\")
    out_lines.append("\\midrule")

    for sid in sorted(weights["stage_id"].unique()):
        g = (
            weights[weights["stage_id"] == sid]
            .sort_values("group_abs_weight", ascending=False)
            .head(5)
        )
        regime = _tex_escape(str(stage_regime.get(int(sid), "")))
        top = "，".join([f"{r.group}({_fmt(float(r.group_abs_weight), digits=3)})" for r in g.itertuples(index=False)])
        out_lines.append(f"{int(sid)} & {regime} & {top} \\\\")

    out_lines.append("\\bottomrule")
    out_lines.append("\\end{tabular}")
    out_lines.append("\\end{TableFit}")
    out_lines.append("\\end{table}")
    (report_tables_dir / "q3_top_groups.tex").write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Strength-task Top factor groups table (Top-5 by normalized coefficient magnitude per stage)
    if not strength_weights.empty:
        out_lines = []
        out_lines.append("% Top factor groups by multinomial LogReg coefficient magnitude (pp_base; strength task)")
        out_lines.append("\\begin{table}[H]")
        out_lines.append("\\centering")
        out_lines.append("\\caption{问题3（强度任务）：各阶段关键因子组（Top-5，LogReg 系数幅度归一化权重，pp\\_base）}")
        out_lines.append("\\label{tab:q3-top-groups-strength}")
        out_lines.append("\\small")
        out_lines.append("\\begin{TableFit}")
        out_lines.append("\\begin{tabular}{rll}")
        out_lines.append("\\toprule")
        out_lines.append("阶段 & regime & Top-5 因子组（权重） \\\\")
        out_lines.append("\\midrule")

        for sid in sorted(strength_weights["stage_id"].unique()):
            g = (
                strength_weights[strength_weights["stage_id"] == sid]
                .sort_values("group_weight", ascending=False)
                .head(5)
            )
            regime = _tex_escape(str(stage_regime.get(int(sid), "")))
            top = "，".join(
                [f"{r.group}({_fmt(float(r.group_weight), digits=3)})" for r in g.itertuples(index=False)]
            )
            out_lines.append(f"{int(sid)} & {regime} & {top} \\\\")

        out_lines.append("\\bottomrule")
        out_lines.append("\\end{tabular}")
        out_lines.append("\\end{TableFit}")
        out_lines.append("\\end{table}")
        (report_tables_dir / "q3_top_groups_strength.tex").write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8"
        )


def write_appendix_full_model_tables(
    *,
    metrics_dir: Path,
    report_tables_dir: Path,
    test_start: str,
    test_end: str,
) -> None:
    """Write longtable-format full model comparison tables for Q1/Q2 into report/tables/."""

    stems = [
        "pp_base",
        "pp_base_engineered",
        "pp_with_futures",
        "pp_with_futures_engineered",
    ]

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    # Q1: regression metrics detail
    rows = []
    for stem in stems:
        path = metrics_dir / stem / "pp_model_metrics.csv"
        if not path.exists():
            continue
        df = _load(path)
        df = df.dropna(subset=["RMSE"]).copy()
        df["dataset"] = stem
        rows.append(df)
    if rows:
        full = pd.concat(rows, axis=0, ignore_index=True)
        full = full.sort_values(["dataset", "RMSE", "MAE"], ascending=[True, True, True])

        out_lines: list[str] = []
        out_lines.append(
            f"% Auto-generated by scripts/update_report_tables.py (test window: {test_start}~{test_end})"
        )
        out_lines.append("{\\small")
        out_lines.append("\\setlength{\\tabcolsep}{4pt}")
        out_lines.append("\\begin{LongTableFit}{@{}p{0.20\\textwidth}p{0.38\\textwidth}rrr@{}}")
        out_lines.append(
            f"\\caption{{问题1：全模型回归指标明细（测试窗 {test_start}$\\sim${test_end}；含集成与可选时序模型）}}"
            "\\label{tab:q1-full-reg}\\\\"
        )
        out_lines.append("\\toprule")
        out_lines.append("数据集 & 模型 & MAE & RMSE & MAPE(\\%) \\\\")
        out_lines.append("\\midrule")
        out_lines.append("\\endfirsthead")
        out_lines.append(
            f"\\caption[]{{问题1：全模型回归指标明细（续；测试窗 {test_start}$\\sim${test_end}）}}\\\\"
        )
        out_lines.append("\\toprule")
        out_lines.append("数据集 & 模型 & MAE & RMSE & MAPE(\\%) \\\\")
        out_lines.append("\\midrule")
        out_lines.append("\\endhead")
        out_lines.append("\\midrule")
        out_lines.append("\\multicolumn{5}{r}{续下页}\\\\")
        out_lines.append("\\endfoot")
        out_lines.append("\\bottomrule")
        out_lines.append("\\endlastfoot")

        for _, r in full.iterrows():
            out_lines.append(
                f"{_tex_escape(str(r['dataset']))} & {_tex_escape(str(r['model']))} & "
                f"{_fmt(float(r['MAE']), digits=2)} & {_fmt(float(r['RMSE']), digits=2)} & "
                f"{_fmt(100.0 * float(r['MAPE']), digits=2)} \\\\"
            )
        out_lines.append("\\end{LongTableFit}")
        out_lines.append("}")
        (report_tables_dir / "appendix_q1_full_regression.tex").write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8"
        )

        # Q1: direction metrics detail
        out_lines = []
        out_lines.append(
            f"% Auto-generated by scripts/update_report_tables.py (test window: {test_start}~{test_end})"
        )
        out_lines.append("{\\small")
        out_lines.append("\\setlength{\\tabcolsep}{4pt}")
        out_lines.append("\\begin{LongTableFit}{@{}p{0.20\\textwidth}p{0.38\\textwidth}rrr@{}}")
        out_lines.append(
            f"\\caption{{问题1：全模型方向指标明细（测试窗 {test_start}$\\sim${test_end}；由回归点预测派生方向）}}"
            "\\label{tab:q1-full-dir}\\\\"
        )
        out_lines.append("\\toprule")
        out_lines.append("数据集 & 模型 & Acc & Precision & Recall \\\\")
        out_lines.append("\\midrule")
        out_lines.append("\\endfirsthead")
        out_lines.append(
            f"\\caption[]{{问题1：全模型方向指标明细（续；测试窗 {test_start}$\\sim${test_end}）}}\\\\"
        )
        out_lines.append("\\toprule")
        out_lines.append("数据集 & 模型 & Acc & Precision & Recall \\\\")
        out_lines.append("\\midrule")
        out_lines.append("\\endhead")
        out_lines.append("\\midrule")
        out_lines.append("\\multicolumn{5}{r}{续下页}\\\\")
        out_lines.append("\\endfoot")
        out_lines.append("\\bottomrule")
        out_lines.append("\\endlastfoot")

        for _, r in full.iterrows():
            out_lines.append(
                f"{_tex_escape(str(r['dataset']))} & {_tex_escape(str(r['model']))} & "
                f"{_fmt(float(r['Accuracy']), digits=3)} & {_fmt(float(r['Precision']), digits=3)} & "
                f"{_fmt(float(r['Recall']), digits=3)} \\\\"
            )
        out_lines.append("\\end{LongTableFit}")
        out_lines.append("}")
        (report_tables_dir / "appendix_q1_full_direction.tex").write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8"
        )

    # Q2: strength classification metrics detail
    rows = []
    for stem in stems:
        path = metrics_dir / stem / "pp_strength_model_metrics.csv"
        if not path.exists():
            continue
        df = _load(path)
        df = df.dropna(subset=["Strength_F1_macro"]).copy()
        df["dataset"] = stem
        rows.append(df)
    if rows:
        full = pd.concat(rows, axis=0, ignore_index=True)
        full = full.sort_values(
            ["dataset", "Strength_F1_macro", "Strength_Accuracy"],
            ascending=[True, False, False],
        )

        out_lines = []
        out_lines.append(
            f"% Auto-generated by scripts/update_report_tables.py (test window: {test_start}~{test_end})"
        )
        out_lines.append("{\\small")
        out_lines.append("\\setlength{\\tabcolsep}{4pt}")
        out_lines.append("\\begin{LongTableFit}{@{}p{0.20\\textwidth}p{0.38\\textwidth}rrrr@{}}")
        out_lines.append(
            f"\\caption{{问题2：全模型强度五分类指标明细（测试窗 {test_start}$\\sim${test_end}；macro-F1 降序）}}"
            "\\label{tab:q2-full-strength}\\\\"
        )
        out_lines.append("\\toprule")
        out_lines.append("数据集 & 模型 & Acc & Prec$_m$ & Rec$_m$ & F1$_m$ \\\\")
        out_lines.append("\\midrule")
        out_lines.append("\\endfirsthead")
        out_lines.append(
            f"\\caption[]{{问题2：全模型强度五分类指标明细（续；测试窗 {test_start}$\\sim${test_end}）}}\\\\"
        )
        out_lines.append("\\toprule")
        out_lines.append("数据集 & 模型 & Acc & Prec$_m$ & Rec$_m$ & F1$_m$ \\\\")
        out_lines.append("\\midrule")
        out_lines.append("\\endhead")
        out_lines.append("\\midrule")
        out_lines.append("\\multicolumn{6}{r}{续下页}\\\\")
        out_lines.append("\\endfoot")
        out_lines.append("\\bottomrule")
        out_lines.append("\\endlastfoot")

        for _, r in full.iterrows():
            out_lines.append(
                f"{_tex_escape(str(r['dataset']))} & {_tex_escape(str(r['model']))} & "
                f"{_fmt(float(r['Strength_Accuracy']), digits=3)} & "
                f"{_fmt(float(r['Strength_Precision_macro']), digits=3)} & "
                f"{_fmt(float(r['Strength_Recall_macro']), digits=3)} & "
                f"{_fmt(float(r['Strength_F1_macro']), digits=3)} \\\\"
            )
        out_lines.append("\\end{LongTableFit}")
        out_lines.append("}")
        (report_tables_dir / "appendix_q2_full_strength.tex").write_text(
            "\n".join(out_lines) + "\n", encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    args.report_tables_dir.mkdir(parents=True, exist_ok=True)

    write_q1_table(
        metrics_dir=args.metrics_dir,
        out_path=args.report_tables_dir / "q1_results_summary.tex",
        test_start=args.test_start,
        test_end=args.test_end,
    )
    write_q2_table(
        metrics_dir=args.metrics_dir,
        out_path=args.report_tables_dir / "q2_results_summary.tex",
        test_start=args.test_start,
        test_end=args.test_end,
    )
    write_q3_stage_tables(q3_dir=args.q3_dir, report_tables_dir=args.report_tables_dir)
    write_appendix_full_model_tables(
        metrics_dir=args.metrics_dir,
        report_tables_dir=args.report_tables_dir,
        test_start=args.test_start,
        test_end=args.test_end,
    )
    print("[OK] updated report tables in:", args.report_tables_dir)


if __name__ == "__main__":
    main()
