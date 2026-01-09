from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pp_forecast.q3_stage import (
    StageConfig,
    compute_regimes,
    finalize_regimes_and_stage_id,
    logreg_stage_factor_weights,
    ridge_stage_factor_weights,
    summarize_stages,
    to_month_period,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q3: stage segmentation + automated factor selection.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/q3/<dataset_stem>/",
    )
    p.add_argument("--trend-window", type=int, default=6)
    p.add_argument("--vol-window", type=int, default=6)
    p.add_argument("--trend-threshold", type=float, default=0.02, help="Trend threshold (e.g. 0.02 = 2%).")
    p.add_argument("--vol-quantile", type=float, default=0.6, help="High-vol threshold quantile (0~1).")
    p.add_argument("--min-stage-len", type=int, default=6)

    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=6, help="Top-K factor groups per stage.")
    p.add_argument("--top-features-per-group", type=int, default=5)
    p.add_argument("--min-stage-samples", type=int, default=8, help="Skip stages shorter than this after filtering.")

    p.add_argument("--include-own-price", action="store_true", help="Include PP价格 group in selection.")
    p.add_argument("--include-calendar", action="store_true", help="Include Calendar features in selection.")
    p.add_argument(
        "--keep-cross-boundary",
        action="store_true",
        help="Keep the first month of each stage even though its features come from previous month (default: drop).",
    )
    p.add_argument("--also-strength", action="store_true", help="Also compute weights for y_strength via LogReg.")
    return p.parse_args()


def _try_plot_price_stages(df: pd.DataFrame, *, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if "month" not in df.columns or "y" not in df.columns or "stage_id" not in df.columns:
        return

    try:
        work = df.copy()
        work["month_ts"] = work["month"].dt.to_timestamp()
        work = work.sort_values("month_ts").reset_index(drop=True)

        x = work["month_ts"]
        y = work["y"].astype(float)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(x, y, color="black", linewidth=1.5)

        change_idx = work.index[work["stage_id"].ne(work["stage_id"].shift(1))].tolist()
        for i in change_idx[1:]:
            ax.axvline(x.iloc[i], color="grey", alpha=0.3, linewidth=1.0)

        ax.set_title("PP monthly price with stage boundaries (Q3)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Price")
        fig.tight_layout()
        fig.savefig(output_dir / "q3_stages_price.png", dpi=200)
        plt.close(fig)
    except Exception:
        return


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (Path("outputs/q3") / args.dataset.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset)
    df["month"] = to_month_period(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    cfg = StageConfig(
        trend_window=args.trend_window,
        vol_window=args.vol_window,
        trend_threshold=args.trend_threshold,
        vol_quantile=args.vol_quantile,
        min_stage_len=args.min_stage_len,
    )

    regimes_df = compute_regimes(df, config=cfg)
    for c in ["trend", "vol", "trend_state", "vol_state", "regime"]:
        df[c] = regimes_df[c].to_numpy()

    merged_regime, stage_id = finalize_regimes_and_stage_id(
        df["regime"], min_stage_len=cfg.min_stage_len, returns=df["y_return"]
    )
    df["regime_merged"] = merged_regime.to_numpy()
    df["stage_id"] = stage_id.to_numpy()

    stage_summary = summarize_stages(
        df,
        stage_id=df["stage_id"],
        regime=df["regime_merged"],
    )
    stage_summary.to_csv(output_dir / "q3_stage_summary.csv", index=False)

    assign_cols = ["month", "stage_id", "regime", "regime_merged", "trend", "vol", "y", "y_return"]
    df_assign = df[assign_cols].copy()
    df_assign["month"] = df_assign["month"].astype(str)
    df_assign.to_csv(output_dir / "q3_stage_assignments.csv", index=False)

    exclude_groups = []
    if not args.include_own_price:
        exclude_groups.append("PP价格")
    if not args.include_calendar:
        exclude_groups.append("Calendar")

    failures: list[dict[str, str]] = []
    group_rows = []
    top_feature_rows = []

    strength_group_rows = []
    strength_top_feature_rows = []

    for stage_id, g in df.groupby("stage_id", sort=True):
        g = g.sort_values("month").reset_index(drop=True)
        if not args.keep_cross_boundary and int(stage_id) != int(df["stage_id"].min()):
            g = g.iloc[1:].reset_index(drop=True)

        if len(g) < args.min_stage_samples:
            failures.append(
                {
                    "task": "stage_skip",
                    "stage_id": str(stage_id),
                    "error_type": "TooShort",
                    "error": f"len={len(g)} < min_stage_samples={args.min_stage_samples}",
                }
            )
            continue

        try:
            fdf, gdf = ridge_stage_factor_weights(
                g,
                alpha=args.ridge_alpha,
                exclude_groups=exclude_groups,
            )
        except Exception as e:
            failures.append(
                {
                    "task": "ridge_group_weights",
                    "stage_id": str(stage_id),
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            continue

        gdf = gdf.copy()
        gdf["stage_id"] = int(stage_id)
        gdf["is_topk"] = False
        if args.top_k > 0 and len(gdf) > 0:
            gdf.loc[gdf.index[: args.top_k], "is_topk"] = True
        group_rows.append(gdf)

        top_groups = gdf.loc[gdf["is_topk"], "group"].tolist()
        fdf_top = fdf[fdf["group"].isin(top_groups)].copy()
        fdf_top = fdf_top.sort_values(["group", "abs_coef"], ascending=[True, False])
        fdf_top["rank_in_group"] = fdf_top.groupby("group")["abs_coef"].rank(
            method="first", ascending=False
        )
        fdf_top = fdf_top[fdf_top["rank_in_group"] <= args.top_features_per_group].copy()
        fdf_top["stage_id"] = int(stage_id)
        top_feature_rows.append(fdf_top)

        if args.also_strength and "y_strength" in g.columns:
            try:
                s_fdf, s_gdf = logreg_stage_factor_weights(
                    g,
                    target_col="y_strength",
                    exclude_groups=exclude_groups,
                )
            except Exception as e:
                failures.append(
                    {
                        "task": "strength_logreg_group_weights",
                        "stage_id": str(stage_id),
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
            else:
                s_gdf = s_gdf.copy()
                s_gdf["stage_id"] = int(stage_id)
                s_gdf["is_topk"] = False
                if args.top_k > 0 and len(s_gdf) > 0:
                    s_gdf.loc[s_gdf.index[: args.top_k], "is_topk"] = True
                strength_group_rows.append(s_gdf)

                s_top_groups = s_gdf.loc[s_gdf["is_topk"], "group"].tolist()
                s_fdf_top = s_fdf[s_fdf["group"].isin(s_top_groups)].copy()
                s_fdf_top = s_fdf_top.sort_values(["group", "coef_mag"], ascending=[True, False])
                s_fdf_top["rank_in_group"] = s_fdf_top.groupby("group")["coef_mag"].rank(
                    method="first", ascending=False
                )
                s_fdf_top = s_fdf_top[s_fdf_top["rank_in_group"] <= args.top_features_per_group].copy()
                s_fdf_top["stage_id"] = int(stage_id)
                strength_top_feature_rows.append(s_fdf_top)

    if group_rows:
        out = pd.concat(group_rows, axis=0, ignore_index=True)
        out.to_csv(output_dir / "q3_ridge_group_weights.csv", index=False)
    if top_feature_rows:
        out = pd.concat(top_feature_rows, axis=0, ignore_index=True)
        out.to_csv(output_dir / "q3_ridge_top_features.csv", index=False)

    if strength_group_rows:
        out = pd.concat(strength_group_rows, axis=0, ignore_index=True)
        out.to_csv(output_dir / "q3_strength_logreg_group_weights.csv", index=False)
    if strength_top_feature_rows:
        out = pd.concat(strength_top_feature_rows, axis=0, ignore_index=True)
        out.to_csv(output_dir / "q3_strength_logreg_top_features.csv", index=False)

    if failures:
        with open(output_dir / "q3_failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

    _try_plot_price_stages(df, output_dir=output_dir)

    print("[OK] wrote Q3 outputs to:", output_dir)
    print(stage_summary.to_string(index=False))


if __name__ == "__main__":
    main()
