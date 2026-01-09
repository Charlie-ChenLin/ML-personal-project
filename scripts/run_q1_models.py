from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pp_forecast.q1_dataset import strength_label
from pp_forecast.q1_models import (
    build_models,
    build_strength_models,
    direction_metrics,
    fit_predict_strength,
    fit_predict_regression,
    predict_baseline,
    regression_metrics,
    strength_metrics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Q1 models and evaluate on a time window.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--test-start", default="2021-01")
    p.add_argument("--test-end", default="2021-07")
    p.add_argument(
        "--futures-mode",
        choices=["restrict", "impute"],
        default="restrict",
        help="If dataset includes futures columns, restrict drops months with missing futures features.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/metrics/<dataset_stem>/",
    )
    p.add_argument(
        "--strong-threshold",
        type=float,
        default=None,
        help="Override strong threshold (e.g. 0.05 means 5%). Default: read from dataset if available.",
    )
    p.add_argument(
        "--flat-threshold",
        type=float,
        default=None,
        help="Override flat threshold (e.g. 0.005 means ±0.5%). Default: read from dataset if available.",
    )
    return p.parse_args()


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.astype(str), freq="M")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (Path("outputs/metrics") / args.dataset.stem)

    df = pd.read_csv(args.dataset)
    df["month"] = _to_period(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    strong_threshold = args.strong_threshold
    if strong_threshold is None and "cfg_strong_threshold" in df.columns:
        vals = df["cfg_strong_threshold"].dropna().unique().tolist()
        if vals:
            strong_threshold = float(vals[0])
    if strong_threshold is None:
        strong_threshold = 0.05

    flat_threshold = args.flat_threshold
    if flat_threshold is None and "cfg_flat_threshold" in df.columns:
        vals = df["cfg_flat_threshold"].dropna().unique().tolist()
        if vals:
            flat_threshold = float(vals[0])
    if flat_threshold is None:
        flat_threshold = 0.005

    futures_cols = [c for c in df.columns if c.startswith("期货价格__")]
    if futures_cols and args.futures_mode == "restrict":
        df = df.dropna(subset=futures_cols).reset_index(drop=True)

    test_start = pd.Period(args.test_start, freq="M")
    test_end = pd.Period(args.test_end, freq="M")
    test_mask = (df["month"] >= test_start) & (df["month"] <= test_end)

    df_train = df.loc[~test_mask].copy()
    df_test = df.loc[test_mask].copy()

    # ensure time order (and no leakage from future months)
    df_train = df_train[df_train["month"] < test_start]

    if df_test.empty:
        raise SystemExit("empty test set (check --test-start/--test-end and dataset coverage)")
    if df_train.empty:
        raise SystemExit("empty train set (check --test-start and dataset coverage)")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []

    # Baselines based on y itself
    y_series = df.set_index("month")["y"]
    baselines = {}
    for name in ["naive", "seasonal_12"]:
        pred = predict_baseline(df, baseline=name)  # aligned to month index
        pred = pred.reindex(y_series.index)
        pred_test = pred.loc[df_test["month"]].to_numpy()
        ok = ~np.isnan(pred_test)
        if ok.sum() == 0:
            continue
        y_true = df_test.loc[ok, "y"].to_numpy()
        y_prev = df_test.loc[ok, "y_prev"].to_numpy()
        y_pred = pred_test[ok]
        dir_true = (y_true - y_prev > 0).astype(int)
        dir_pred = (y_pred - y_prev > 0).astype(int)
        ret_pred = (y_pred - y_prev) / y_prev
        if "y_strength" in df_test.columns:
            strength_true = df_test.loc[ok, "y_strength"].astype(str).to_numpy()
            strength_pred = np.array(
                [
                    strength_label(
                        float(r),
                        strong_threshold=strong_threshold,
                        flat_threshold=flat_threshold,
                    )
                    for r in ret_pred
                ]
            )
            strength_part = strength_metrics(strength_true, strength_pred)
        else:
            strength_part = {}
        baselines[name] = {
            **regression_metrics(y_true, y_pred),
            **direction_metrics(dir_true, dir_pred),
            **strength_part,
        }

    # ML models
    models = build_models()
    model_rows = []
    pred_rows = []

    y_true = df_test["y"].to_numpy()
    y_prev = df_test["y_prev"].to_numpy()
    dir_true = (y_true - y_prev > 0).astype(int)
    ret_true = (
        df_test["y_return"].to_numpy()
        if "y_return" in df_test.columns
        else (y_true - y_prev) / y_prev
    )
    strength_true = (
        df_test["y_strength"].astype(str).to_numpy() if "y_strength" in df_test.columns else None
    )

    for model_name, model in models.items():
        try:
            res = fit_predict_regression(df_train, df_test, model_name=model_name, model=model)
        except Exception as e:
            failures.append(
                {
                    "task": "price_regression",
                    "model": model_name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            continue

        dir_pred = (res.y_pred - y_prev > 0).astype(int)
        ret_pred = (res.y_pred - y_prev) / y_prev

        if strength_true is not None:
            strength_pred = np.array(
                [
                    strength_label(
                        float(r),
                        strong_threshold=strong_threshold,
                        flat_threshold=flat_threshold,
                    )
                    for r in ret_pred
                ]
            )
            strength_part = strength_metrics(strength_true, strength_pred)
        else:
            strength_pred = None
            strength_part = {}

        row = {
            "model": model_name,
            **res.metrics,
            **direction_metrics(dir_true, dir_pred),
            **strength_part,
        }
        model_rows.append(row)

        pred_payload = {
            "month": df_test["month"].astype(str),
            "model": model_name,
            "y_true": y_true,
            "y_pred": res.y_pred,
            "direction_true": dir_true,
            "direction_pred": dir_pred,
            "return_true": ret_true,
            "return_pred": ret_pred,
        }
        if strength_true is not None and strength_pred is not None:
            pred_payload["strength_true"] = strength_true
            pred_payload["strength_pred"] = strength_pred

        pred_rows.append(
            pd.DataFrame(pred_payload)
        )

        try:
            with open(output_dir / f"q1_params_{model_name}.json", "w", encoding="utf-8") as f:
                json.dump(res.params, f, ensure_ascii=False, indent=2)
        except Exception:
            # writing params is non-critical for the main metrics outputs
            pass

    metrics_df = pd.DataFrame(model_rows)
    if not metrics_df.empty and "RMSE" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("RMSE")
    metrics_df.to_csv(output_dir / "q1_model_metrics.csv", index=False)

    if baselines:
        with open(output_dir / "q1_baselines.json", "w", encoding="utf-8") as f:
            json.dump(baselines, f, ensure_ascii=False, indent=2)

    if pred_rows:
        preds = pd.concat(pred_rows, axis=0, ignore_index=True)
        preds.to_csv(output_dir / "q1_test_predictions.csv", index=False)

    # Strength classification models (输出“涨跌幅度区间”的概率)
    if strength_true is not None:
        strength_models = build_strength_models()
        strength_rows = []
        strength_pred_rows = []

        for model_name, model in strength_models.items():
            try:
                res = fit_predict_strength(df_train, df_test, model_name=model_name, model=model)
            except Exception as e:
                failures.append(
                    {
                        "task": "strength_classification",
                        "model": model_name,
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )
                continue

            strength_rows.append({"model": model_name, **res.metrics})

            proba_cols = [f"proba__{c}" for c in res.classes]
            proba_df = pd.DataFrame(res.y_proba, columns=proba_cols)
            pred_df = pd.DataFrame(
                {
                    "month": df_test["month"].astype(str),
                    "model": model_name,
                    "strength_true": strength_true,
                    "strength_pred": res.y_pred,
                }
            )
            strength_pred_rows.append(pd.concat([pred_df, proba_df], axis=1))

            with open(
                output_dir / f"q1_strength_params_{model_name}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(res.params, f, ensure_ascii=False, indent=2)

        strength_metrics_df = pd.DataFrame(strength_rows)
        if not strength_metrics_df.empty and "Strength_F1_macro" in strength_metrics_df.columns:
            strength_metrics_df = strength_metrics_df.sort_values(
                "Strength_F1_macro", ascending=False
            )
        strength_metrics_df.to_csv(output_dir / "q1_strength_model_metrics.csv", index=False)

        if strength_pred_rows:
            strength_preds = pd.concat(strength_pred_rows, axis=0, ignore_index=True)
            strength_preds.to_csv(output_dir / "q1_strength_test_predictions.csv", index=False)

    if failures:
        with open(output_dir / "q1_failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, ensure_ascii=False, indent=2)

    print("[OK] wrote metrics to:", output_dir)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))
    else:
        print("(no successful regression models)")
    if baselines:
        print("\\nBaselines:", json.dumps(baselines, ensure_ascii=False))
    if failures:
        print("\\nFailures:", json.dumps(failures, ensure_ascii=False))


if __name__ == "__main__":
    main()
