from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


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


def strength_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Strength_Accuracy": float(accuracy_score(y_true, y_pred)),
        "Strength_Precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Strength_Recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Strength_F1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


@dataclass(frozen=True)
class ModelResult:
    name: str
    y_pred: np.ndarray
    metrics: dict[str, float]
    params: dict[str, Any]


def _numeric_features(df: pd.DataFrame) -> list[str]:
    exclude = {
        "month",
        "y",
        "y_prev",
        "y_direction",
        "y_return",
        "y_strength",
        "cfg_engineer_features",
        "cfg_strong_threshold",
        "cfg_flat_threshold",
    }
    return [c for c in df.columns if c not in exclude]


def build_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {}

    models["ridge"] = Ridge(alpha=1.0, random_state=random_state)
    models["lasso"] = Lasso(alpha=0.001, random_state=random_state, max_iter=20000)
    models["elasticnet"] = ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        random_state=random_state,
        max_iter=20000,
    )
    models["bayes_ridge"] = BayesianRidge()
    models["huber"] = HuberRegressor(max_iter=2000, epsilon=1.35, alpha=0.0001)

    models["knn"] = KNeighborsRegressor(n_neighbors=8, weights="distance")
    models["svr_rbf"] = SVR(C=10.0, gamma="scale", epsilon=0.1)

    models["rf"] = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    models["extra_trees"] = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=8,
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
    models["ada"] = AdaBoostRegressor(
        n_estimators=500,
        learning_rate=0.05,
        random_state=random_state,
    )
    models["bagging_tree"] = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=random_state),
        n_estimators=300,
        bootstrap=True,
        random_state=random_state,
        n_jobs=1,
    )

    return models


def build_strength_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {}

    models["logreg"] = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
    )
    models["knn_clf"] = KNeighborsClassifier(n_neighbors=10, weights="distance")
    models["nb"] = GaussianNB()
    models["svc_rbf"] = SVC(
        C=5.0,
        gamma="scale",
        probability=True,
        random_state=random_state,
    )
    models["rf_clf"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    models["extra_trees_clf"] = ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=8,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,
    )
    models["gbr_clf"] = GradientBoostingClassifier(
        random_state=random_state,
        learning_rate=0.05,
        n_estimators=500,
        max_depth=3,
    )
    models["ada_clf"] = AdaBoostClassifier(
        n_estimators=500,
        learning_rate=0.05,
        random_state=random_state,
    )
    models["bagging_tree_clf"] = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=4, random_state=random_state),
        n_estimators=300,
        bootstrap=True,
        random_state=random_state,
        n_jobs=1,
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
    features = [c for c in features if df_train[c].notna().any()]
    if not features:
        raise ValueError("no usable numeric features (all NaN in train)")

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


@dataclass(frozen=True)
class ClassifierResult:
    name: str
    y_pred: np.ndarray
    y_proba: np.ndarray
    classes: list[str]
    metrics: dict[str, float]
    params: dict[str, Any]


def fit_predict_strength(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    model_name: str,
    model: Any,
) -> ClassifierResult:
    features = _numeric_features(df_train)
    features = [c for c in features if df_train[c].notna().any()]
    if not features:
        raise ValueError("no usable numeric features (all NaN in train)")

    X_train = df_train[features]
    y_train = df_train["y_strength"].astype(str).to_numpy()
    X_test = df_test[features]
    y_true = df_test["y_strength"].astype(str).to_numpy()

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

    model_step = pipe.named_steps["model"]
    if not hasattr(model_step, "predict_proba"):
        raise TypeError(f"{model_name} does not support predict_proba")
    y_proba = pipe.predict_proba(X_test)
    classes = [str(c) for c in getattr(model_step, "classes_", [])]

    return ClassifierResult(
        name=model_name,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=classes,
        metrics=strength_metrics(y_true, y_pred),
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
