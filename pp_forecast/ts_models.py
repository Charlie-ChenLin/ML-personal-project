from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .models import ModelResult, _numeric_features, regression_metrics


def _select_topk_features(
    df_train: pd.DataFrame, *, max_features: int | None
) -> list[str]:
    feats = _numeric_features(df_train)
    feats = [c for c in feats if df_train[c].notna().any()]
    if not feats:
        return []
    if max_features is None or max_features <= 0 or len(feats) <= max_features:
        return feats

    # Simple univariate relevance: absolute correlation with y on training window.
    corr = df_train[feats].corrwith(df_train["y"]).abs().dropna()
    if corr.empty:
        return feats[:max_features]
    top = corr.sort_values(ascending=False).head(max_features).index.tolist()
    return top


def _prep_exog(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    X_train = df_train[features].to_numpy()
    X_test = df_test[features].to_numpy()

    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    meta = {
        "features": features,
        "imputer": {"strategy": "median"},
        "scaler": {"type": "StandardScaler"},
    }
    return X_train_scaled, X_test_scaled, meta


def fit_predict_sarimax(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    model_name: str = "sarimax",
    max_features: int | None = 20,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_period: int = 12,
) -> ModelResult:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "statsmodels is required for SARIMAX; install statsmodels in your environment."
        ) from e

    features = _select_topk_features(df_train, max_features=max_features)
    if not features:
        raise ValueError("no usable numeric features (all NaN in train)")

    X_train, X_test, meta = _prep_exog(df_train, df_test, features=features)
    y_train = df_train["y"].to_numpy(dtype=float)

    use_seasonal = len(df_train) >= (2 * seasonal_period)
    seasonal_order = (1, 0, 1, seasonal_period) if use_seasonal else (0, 0, 0, 0)

    mod = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = mod.fit(disp=False)

    fcst = res.get_forecast(steps=len(df_test), exog=X_test)
    y_pred = np.asarray(fcst.predicted_mean, dtype=float)

    params = {
        "type": "sarimax",
        "order": order,
        "seasonal_order": seasonal_order,
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
        **meta,
    }

    return ModelResult(
        name=model_name,
        y_pred=y_pred,
        metrics=regression_metrics(df_test["y"].to_numpy(), y_pred),
        params=params,
    )


def fit_predict_prophet(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    model_name: str = "prophet",
    max_regressors: int | None = 10,
) -> ModelResult:
    try:
        from prophet import Prophet  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("prophet is required; install prophet in your environment.") from e

    # ds: use month-start timestamp for stability
    ds_train = pd.PeriodIndex(df_train["month"].astype(str), freq="M").to_timestamp()
    ds_test = pd.PeriodIndex(df_test["month"].astype(str), freq="M").to_timestamp()

    regressors = _select_topk_features(df_train, max_features=max_regressors)
    if not regressors:
        # Prophet without extra regressors (trend + yearly seasonality)
        regressors = []

    # Prophet cannot handle NaNs in regressors; impute with train median
    imputer = SimpleImputer(strategy="median")
    if regressors:
        X_train = imputer.fit_transform(df_train[regressors].to_numpy())
        X_test = imputer.transform(df_test[regressors].to_numpy())
        dfp_train = pd.DataFrame(X_train, columns=regressors)
        dfp_test = pd.DataFrame(X_test, columns=regressors)
    else:
        dfp_train = pd.DataFrame(index=range(len(df_train)))
        dfp_test = pd.DataFrame(index=range(len(df_test)))

    dfp_train.insert(0, "ds", ds_train)
    dfp_train.insert(1, "y", df_train["y"].to_numpy(dtype=float))
    dfp_test.insert(0, "ds", ds_test)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
    )
    for r in regressors:
        m.add_regressor(r)

    m.fit(dfp_train)
    forecast = m.predict(dfp_test)
    y_pred = forecast["yhat"].to_numpy(dtype=float)

    params: dict[str, Any] = {
        "type": "prophet",
        "regressors": regressors,
        "seasonality_mode": "additive",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False,
        "imputer": {"strategy": "median"} if regressors else None,
    }
    # Prophet object is not JSON-serializable; only record the config above.

    return ModelResult(
        name=model_name,
        y_pred=y_pred,
        metrics=regression_metrics(df_test["y"].to_numpy(), y_pred),
        params=params,
    )
