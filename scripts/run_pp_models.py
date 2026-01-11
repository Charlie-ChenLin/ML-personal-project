from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from pp_forecast.dataset import strength_label
from pp_forecast.models import (
    build_models,
    build_strength_models,
    direction_metrics,
    fit_predict_strength,
    fit_predict_regression,
    predict_baseline,
    regression_metrics,
    strength_metrics,
)
from pp_forecast.ts_models import fit_predict_prophet, fit_predict_sarimax


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PP models and evaluate on a time window.")
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
    p.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="If >0, run residual bootstrap for the best-CV regression model and output intervals/probabilities.",
    )
    p.add_argument(
        "--bootstrap-alpha",
        type=float,
        default=0.1,
        help="Interval alpha (e.g. 0.1 -> 90%% interval).",
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
    futures_base = [c for c in futures_cols if c in {"期货价格__mean", "期货价格__min", "期货价格__max", "期货价格__last", "期货价格__range"}]
    restrict_cols = futures_base if futures_base else futures_cols
    if futures_cols and args.futures_mode == "restrict":
        df = df.dropna(subset=restrict_cols).reset_index(drop=True)

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
    regression_preds: dict[str, np.ndarray] = {}
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

        regression_preds[model_name] = res.y_pred

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
            with open(output_dir / f"pp_params_{model_name}.json", "w", encoding="utf-8") as f:
                json.dump(res.params, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            # writing params is non-critical for the main metrics outputs
            pass

    # Time-series specific models (optional dependencies)
    for ts_name, ts_fn in [
        ("sarimax", fit_predict_sarimax),
        ("prophet", fit_predict_prophet),
    ]:
        try:
            res = ts_fn(df_train, df_test, model_name=ts_name)
        except Exception as e:
            failures.append(
                {
                    "task": "ts_price_regression",
                    "model": ts_name,
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

        model_rows.append(
            {
                "model": ts_name,
                **res.metrics,
                **direction_metrics(dir_true, dir_pred),
                **strength_part,
            }
        )

        pred_payload = {
            "month": df_test["month"].astype(str),
            "model": ts_name,
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
        pred_rows.append(pd.DataFrame(pred_payload))

        try:
            with open(output_dir / f"pp_params_{ts_name}.json", "w", encoding="utf-8") as f:
                json.dump(res.params, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    # Cross-validated score on training set (for ensemble / model selection)
    def _cv_setup(n: int) -> TimeSeriesSplit | None:
        # allow CV on smaller monthly samples (but keep at least a year to reduce instability)
        if n < 12:
            return None
        test_size = min(6, max(2, n // 5))
        max_splits = (n - 1) // test_size
        n_splits = min(3, max_splits)
        if n_splits < 2:
            return None
        return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    cv = _cv_setup(len(df_train))
    cv_rows = []
    cv_rmse: dict[str, float] = {}
    if cv is not None:
        for model_name, model in models.items():
            rmses = []
            for tr_idx, va_idx in cv.split(df_train):
                df_tr = df_train.iloc[tr_idx].copy()
                df_va = df_train.iloc[va_idx].copy()
                try:
                    res = fit_predict_regression(
                        df_tr, df_va, model_name=model_name, model=clone(model)
                    )
                except Exception as e:
                    failures.append(
                        {
                            "task": "cv_price_regression",
                            "model": model_name,
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )
                    rmses = []
                    break
                rmses.append(float(res.metrics["RMSE"]))

            if rmses:
                mean_rmse = float(np.mean(rmses))
                cv_rmse[model_name] = mean_rmse
                cv_rows.append(
                    {
                        "model": model_name,
                        "cv_n_splits": cv.n_splits,
                        "cv_test_size": getattr(cv, "test_size", None),
                        "cv_RMSE_mean": mean_rmse,
                        "cv_RMSE_std": float(np.std(rmses, ddof=0)),
                    }
                )

        if cv_rows:
            pd.DataFrame(cv_rows).sort_values("cv_RMSE_mean").to_csv(
                output_dir / "pp_model_cv.csv", index=False
            )

    # Simple ensembles across successful regressors
    if regression_preds:
        try:
            pred_mat = np.column_stack([regression_preds[k] for k in regression_preds.keys()])
            ens_mean = pred_mat.mean(axis=1)
            ens_models = list(regression_preds.keys())

            def _append_ens(
                name: str, y_pred_ens: np.ndarray, *, ensemble_of: list[str] | None = None
            ) -> None:
                used_models = ensemble_of or ens_models
                dir_pred = (y_pred_ens - y_prev > 0).astype(int)
                ret_pred = (y_pred_ens - y_prev) / y_prev

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

                model_rows.append(
                    {
                        "model": name,
                        **regression_metrics(y_true, y_pred_ens),
                        **direction_metrics(dir_true, dir_pred),
                        **strength_part,
                    }
                )

                pred_payload = {
                    "month": df_test["month"].astype(str),
                    "model": name,
                    "y_true": y_true,
                    "y_pred": y_pred_ens,
                    "direction_true": dir_true,
                    "direction_pred": dir_pred,
                    "return_true": ret_true,
                    "return_pred": ret_pred,
                }
                if strength_true is not None and strength_pred is not None:
                    pred_payload["strength_true"] = strength_true
                    pred_payload["strength_pred"] = strength_pred
                pred_rows.append(pd.DataFrame(pred_payload))

                with open(output_dir / f"pp_params_{name}.json", "w", encoding="utf-8") as f:
                    json.dump({"ensemble_of": used_models}, f, ensure_ascii=False, indent=2)

            _append_ens("ensemble_mean", ens_mean)

            # Robust alternatives (less sensitive to weak models)
            try:
                ens_median = np.median(pred_mat, axis=1)
                _append_ens("ensemble_median", ens_median)

                n_models = pred_mat.shape[1]
                trim_n = 1 if n_models >= 5 else 0
                if trim_n > 0 and n_models >= (2 * trim_n + 2):
                    sorted_mat = np.sort(pred_mat, axis=1)
                    ens_trimmed = sorted_mat[:, trim_n:-trim_n].mean(axis=1)
                    _append_ens("ensemble_trimmed_mean", ens_trimmed)
            except Exception:
                # robust ensembles are optional
                pass

            # CV-weighted ensemble (if CV is available)
            if cv_rmse:
                use = [m for m in ens_models if m in cv_rmse]
                if len(use) >= 2:
                    w = np.array([1.0 / (cv_rmse[m] ** 2) for m in use], dtype=float)
                    w = w / w.sum()
                    pred_mat_w = np.column_stack([regression_preds[m] for m in use])
                    ens_w = pred_mat_w @ w
                    _append_ens("ensemble_cv_weighted", ens_w, ensemble_of=use)

                # Top-k (by CV RMSE) ensembles
                ranked = [m for m, _ in sorted(cv_rmse.items(), key=lambda kv: kv[1]) if m in ens_models]
                k = min(5, len(ranked))
                use_k = ranked[:k]
                if len(use_k) >= 2:
                    pred_mat_k = np.column_stack([regression_preds[m] for m in use_k])
                    _append_ens("ensemble_topk_mean", pred_mat_k.mean(axis=1), ensemble_of=use_k)

                    w = np.array([1.0 / (cv_rmse[m] ** 2) for m in use_k], dtype=float)
                    w = w / w.sum()
                    _append_ens(
                        "ensemble_topk_cv_weighted", pred_mat_k @ w, ensemble_of=use_k
                    )
        except Exception as e:
            failures.append(
                {
                    "task": "ensemble_regression",
                    "model": "ensemble",
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )

    metrics_df = pd.DataFrame(model_rows)
    if not metrics_df.empty and "RMSE" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("RMSE")
    metrics_df.to_csv(output_dir / "pp_model_metrics.csv", index=False)

    if baselines:
        with open(output_dir / "pp_baselines.json", "w", encoding="utf-8") as f:
            json.dump(baselines, f, ensure_ascii=False, indent=2)

    if pred_rows:
        preds = pd.concat(pred_rows, axis=0, ignore_index=True)
        preds.to_csv(output_dir / "pp_test_predictions.csv", index=False)

    # Strength classification models (输出“涨跌幅度区间”的概率)
    if strength_true is not None:
        strength_models = build_strength_models()
        strength_rows = []
        strength_pred_rows = []
        strength_probas: dict[str, tuple[list[str], np.ndarray]] = {}

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
            strength_probas[model_name] = (res.classes, res.y_proba)

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
            # Align indexes before column-wise concat (df_test keeps original row indexes).
            pred_df = pred_df.reset_index(drop=True)
            proba_df = proba_df.reset_index(drop=True)
            strength_pred_rows.append(pd.concat([pred_df, proba_df], axis=1))

            with open(
                output_dir / f"pp_strength_params_{model_name}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(res.params, f, ensure_ascii=False, indent=2, default=str)

        if strength_pred_rows:
            strength_preds = pd.concat(strength_pred_rows, axis=0, ignore_index=True)
            strength_preds.to_csv(output_dir / "pp_strength_test_predictions.csv", index=False)

        # Optional: CV on training set (for weighted/top-k ensembling)
        strength_cv: dict[str, float] = {}
        strength_cv_rows = []
        if cv is not None:
            for model_name, model in strength_models.items():
                f1s = []
                for tr_idx, va_idx in cv.split(df_train):
                    df_tr = df_train.iloc[tr_idx].copy()
                    df_va = df_train.iloc[va_idx].copy()
                    try:
                        res = fit_predict_strength(
                            df_tr, df_va, model_name=model_name, model=clone(model)
                        )
                    except Exception as e:
                        failures.append(
                            {
                                "task": "cv_strength_classification",
                                "model": model_name,
                                "error_type": type(e).__name__,
                                "error": str(e),
                            }
                        )
                        f1s = []
                        break
                    f1s.append(float(res.metrics.get("Strength_F1_macro", np.nan)))

                if f1s:
                    mean_f1 = float(np.nanmean(f1s))
                    strength_cv[model_name] = mean_f1
                    strength_cv_rows.append(
                        {
                            "model": model_name,
                            "cv_n_splits": cv.n_splits,
                            "cv_test_size": getattr(cv, "test_size", None),
                            "cv_Strength_F1_macro_mean": mean_f1,
                            "cv_Strength_F1_macro_std": float(np.nanstd(f1s, ddof=0)),
                        }
                    )

            if strength_cv_rows:
                pd.DataFrame(strength_cv_rows).sort_values(
                    "cv_Strength_F1_macro_mean", ascending=False
                ).to_csv(output_dir / "pp_strength_model_cv.csv", index=False)

        # Ensembling (soft voting)
        if strength_probas:
            strength_ensemble_pred_rows: list[pd.DataFrame] = []

            def _aligned_proba_matrix(
                probs: dict[str, tuple[list[str], np.ndarray]]
            ) -> tuple[list[str], dict[str, np.ndarray]]:
                all_classes = sorted({c for classes, _ in probs.values() for c in classes})
                mats: dict[str, np.ndarray] = {}
                for name, (classes, proba) in probs.items():
                    dfp = pd.DataFrame(proba, columns=[str(c) for c in classes])
                    dfp = dfp.reindex(columns=all_classes, fill_value=0.0)
                    mats[name] = dfp.to_numpy(dtype=float)
                return all_classes, mats

            try:
                all_classes, mats = _aligned_proba_matrix(strength_probas)
                proba_mean = np.mean(np.stack(list(mats.values()), axis=0), axis=0)
                pred_idx = np.argmax(proba_mean, axis=1)
                y_pred_ens = np.array([all_classes[i] for i in pred_idx], dtype=object)

                strength_rows.append(
                    {"model": "ensemble_proba_mean", **strength_metrics(strength_true, y_pred_ens)}
                )

                proba_cols = [f"proba__{c}" for c in all_classes]
                proba_df = pd.DataFrame(proba_mean, columns=proba_cols)
                pred_df = pd.DataFrame(
                    {
                        "month": df_test["month"].astype(str),
                        "model": "ensemble_proba_mean",
                        "strength_true": strength_true,
                        "strength_pred": y_pred_ens,
                    }
                )
                pred_df = pred_df.reset_index(drop=True)
                proba_df = proba_df.reset_index(drop=True)
                strength_ensemble_pred_rows.append(pd.concat([pred_df, proba_df], axis=1))

                with open(
                    output_dir / "pp_strength_params_ensemble_proba_mean.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        {"ensemble_of": list(strength_probas.keys()), "classes": all_classes},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception as e:
                failures.append(
                    {
                        "task": "ensemble_strength",
                        "model": "ensemble_proba_mean",
                        "error_type": type(e).__name__,
                        "error": str(e),
                    }
                )

            # CV-weighted / top-k CV ensemble (if CV is available)
            if strength_cv:
                try:
                    eligible = [m for m in strength_probas.keys() if m in strength_cv]
                    eligible = [m for m in eligible if np.isfinite(strength_cv[m])]
                    eligible = sorted(eligible, key=lambda m: strength_cv[m], reverse=True)
                    k = min(5, len(eligible))
                    use_k = eligible[:k]

                    if len(use_k) >= 2:
                        all_classes, mats = _aligned_proba_matrix(
                            {m: strength_probas[m] for m in use_k}
                        )
                        w = np.array([max(0.0, float(strength_cv[m])) for m in use_k], dtype=float)
                        if w.sum() > 0:
                            w = w / w.sum()
                            proba_w = np.zeros_like(next(iter(mats.values())))
                            for i, m in enumerate(use_k):
                                proba_w += mats[m] * w[i]

                            pred_idx = np.argmax(proba_w, axis=1)
                            y_pred_ens = np.array([all_classes[i] for i in pred_idx], dtype=object)

                            strength_rows.append(
                                {
                                    "model": "ensemble_proba_topk_cv_weighted",
                                    **strength_metrics(strength_true, y_pred_ens),
                                }
                            )

                            proba_cols = [f"proba__{c}" for c in all_classes]
                            proba_df = pd.DataFrame(proba_w, columns=proba_cols)
                            pred_df = pd.DataFrame(
                                {
                                    "month": df_test["month"].astype(str),
                                    "model": "ensemble_proba_topk_cv_weighted",
                                    "strength_true": strength_true,
                                    "strength_pred": y_pred_ens,
                                }
                            )
                            pred_df = pred_df.reset_index(drop=True)
                            proba_df = proba_df.reset_index(drop=True)
                            strength_ensemble_pred_rows.append(pd.concat([pred_df, proba_df], axis=1))

                            with open(
                                output_dir / "pp_strength_params_ensemble_proba_topk_cv_weighted.json",
                                "w",
                                encoding="utf-8",
                            ) as f:
                                json.dump(
                                    {"ensemble_of": use_k, "weights": dict(zip(use_k, w.tolist()))},
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                except Exception as e:
                    failures.append(
                        {
                            "task": "ensemble_strength",
                            "model": "ensemble_proba_topk_cv_weighted",
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )

            if strength_ensemble_pred_rows:
                pd.concat(strength_ensemble_pred_rows, axis=0, ignore_index=True).to_csv(
                    output_dir / "pp_strength_ensemble_test_predictions.csv", index=False
                )

        strength_metrics_df = pd.DataFrame(strength_rows)
        if not strength_metrics_df.empty and "Strength_F1_macro" in strength_metrics_df.columns:
            strength_metrics_df = strength_metrics_df.sort_values(
                "Strength_F1_macro", ascending=False
            )
        strength_metrics_df.to_csv(output_dir / "pp_strength_model_metrics.csv", index=False)

    # Residual bootstrap for the best-CV regression model (optional)
    if args.bootstrap and args.bootstrap > 0 and regression_preds:
        try:
            # pick best model by CV RMSE if available; otherwise fall back to best test RMSE
            if cv_rmse:
                boot_model = min(cv_rmse.items(), key=lambda kv: kv[1])[0]
            else:
                # only allow base regression models (so we can refit + compute residuals)
                candidate_rows = [r for r in model_rows if r.get("model") in regression_preds]
                if not candidate_rows:
                    raise ValueError("no eligible base models for bootstrap")
                best_row = pd.DataFrame(candidate_rows).sort_values("RMSE").iloc[0]
                boot_model = str(best_row["model"])

            if boot_model in models:
                boot_model_obj = models[boot_model]
            else:
                raise ValueError(f"bootstrap model not found in base models: {boot_model}")

            # train fit to get residuals
            train_fit = fit_predict_regression(
                df_train, df_train, model_name=boot_model, model=clone(boot_model_obj)
            )
            resid = df_train["y"].to_numpy() - train_fit.y_pred

            # test point prediction already computed
            y_hat = regression_preds[boot_model]
            rng = np.random.default_rng(42)
            n_boot = int(args.bootstrap)
            n_test = len(df_test)
            draws = rng.choice(resid, size=(n_boot, n_test), replace=True)
            y_sim = y_hat[None, :] + draws

            lo_q = args.bootstrap_alpha / 2.0
            hi_q = 1.0 - lo_q
            y_lo = np.quantile(y_sim, lo_q, axis=0)
            y_hi = np.quantile(y_sim, hi_q, axis=0)

            y_prev_test = y_prev
            ret_sim = (y_sim - y_prev_test[None, :]) / y_prev_test[None, :]
            ret_lo = np.quantile(ret_sim, lo_q, axis=0)
            ret_hi = np.quantile(ret_sim, hi_q, axis=0)
            p_up = np.mean((y_sim - y_prev_test[None, :]) > 0, axis=0)

            # strength probabilities from simulated returns
            buckets = ["big_up", "small_up", "flat", "small_down", "big_down"]
            strength_probs = {b: np.zeros(n_test, dtype=float) for b in buckets}
            for i in range(n_test):
                labs = [
                    strength_label(
                        float(r),
                        strong_threshold=strong_threshold,
                        flat_threshold=flat_threshold,
                    )
                    for r in ret_sim[:, i]
                ]
                vc = pd.Series(labs).value_counts()
                for b in buckets:
                    strength_probs[b][i] = float(vc.get(b, 0.0)) / float(n_boot)

            out = pd.DataFrame(
                {
                    "month": df_test["month"].astype(str),
                    "model": boot_model,
                    "bootstrap_n": n_boot,
                    "y_pred": y_hat,
                    "y_lo": y_lo,
                    "y_hi": y_hi,
                    "return_pred": (y_hat - y_prev_test) / y_prev_test,
                    "return_lo": ret_lo,
                    "return_hi": ret_hi,
                    "p_up": p_up,
                    **{f"proba_strength__{b}": strength_probs[b] for b in buckets},
                }
            )
            out.to_csv(output_dir / "pp_bootstrap_predictions.csv", index=False)
        except Exception as e:
            failures.append(
                {
                    "task": "bootstrap",
                    "model": "bootstrap",
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )

    if failures:
        with open(output_dir / "pp_failures.json", "w", encoding="utf-8") as f:
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
