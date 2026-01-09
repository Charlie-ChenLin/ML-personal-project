from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from pp_forecast.q1_models import (
    build_models,
    direction_metrics,
    fit_predict_regression,
    predict_baseline,
    regression_metrics,
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
    return p.parse_args()


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(s.astype(str), freq="M")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (Path("outputs/metrics") / args.dataset.stem)

    df = pd.read_csv(args.dataset)
    df["month"] = _to_period(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

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
        baselines[name] = {
            **regression_metrics(y_true, y_pred),
            **direction_metrics(dir_true, dir_pred),
        }

    # ML models
    models = build_models()
    model_rows = []
    pred_rows = []

    for model_name, model in models.items():
        res = fit_predict_regression(df_train, df_test, model_name=model_name, model=model)
        y_true = df_test["y"].to_numpy()
        y_prev = df_test["y_prev"].to_numpy()
        dir_true = (y_true - y_prev > 0).astype(int)
        dir_pred = (res.y_pred - y_prev > 0).astype(int)

        row = {
            "model": model_name,
            **res.metrics,
            **direction_metrics(dir_true, dir_pred),
        }
        model_rows.append(row)

        pred_rows.append(
            pd.DataFrame(
                {
                    "month": df_test["month"].astype(str),
                    "model": model_name,
                    "y_true": y_true,
                    "y_pred": res.y_pred,
                    "direction_true": dir_true,
                    "direction_pred": dir_pred,
                }
            )
        )

        with open(output_dir / f"q1_params_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(res.params, f, ensure_ascii=False, indent=2)

    metrics_df = pd.DataFrame(model_rows).sort_values("RMSE")
    metrics_df.to_csv(output_dir / "q1_model_metrics.csv", index=False)

    if baselines:
        with open(output_dir / "q1_baselines.json", "w", encoding="utf-8") as f:
            json.dump(baselines, f, ensure_ascii=False, indent=2)

    if pred_rows:
        preds = pd.concat(pred_rows, axis=0, ignore_index=True)
        preds.to_csv(output_dir / "q1_test_predictions.csv", index=False)

    print("[OK] wrote metrics to:", output_dir)
    print(metrics_df.to_string(index=False))
    if baselines:
        print("\\nBaselines:", json.dumps(baselines, ensure_ascii=False))


if __name__ == "__main__":
    main()
