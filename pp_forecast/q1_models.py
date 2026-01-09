from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


def direction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


@dataclass(frozen=True)
class ModelResult:
    name: str
    y_pred: np.ndarray
    metrics: dict[str, float]
    params: dict[str, Any]


def _numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {"month", "y", "y_prev", "y_direction"}
    return [c for c in df.columns if c not in exclude]


def build_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {}

    models["ridge"] = Ridge(alpha=1.0, random_state=random_state)
    models["rf"] = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )

    models["gbr"] = GradientBoostingRegressor(
        random_state=random_state,
        learning_rate=0.05,
        n_estimators=500,
        max_depth=3,
    )

    return models


def fit_predict_regression(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    model_name: str,
    model: Any,
) -> ModelResult:
    features = _numeric_features(df_train)

    X_train = df_train[features]
    y_train = df_train["y"].to_numpy()
    X_test = df_test[features]

    # preprocessing (all numeric)
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return ModelResult(
        name=model_name,
        y_pred=y_pred,
        metrics=regression_metrics(df_test["y"].to_numpy(), y_pred),
        params=getattr(model, "get_params", lambda: {})(),
    )


BaselineName = Literal["naive", "seasonal_12"]


def predict_baseline(df_all: pd.DataFrame, *, baseline: BaselineName) -> pd.Series:
    s = df_all.set_index("month")["y"]
    s.index = pd.PeriodIndex(s.index, freq="M")
    if baseline == "naive":
        pred = s.shift(1)  # y(t) ~= y(t-1)
    elif baseline == "seasonal_12":
        pred = s.shift(12)
    else:
        raise ValueError("unknown baseline")
    return pred.rename(f"pred_{baseline}")
