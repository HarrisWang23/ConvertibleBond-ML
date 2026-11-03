"""A compact demonstration of the paper's ML workflow."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATE = "date"
ID = "bond_id"
TARGET = "ret_fwd"


def rank_features(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rank predictors cross-sectionally by date and map ranks to [-1, 1]."""
    out = panel.copy()
    for col in features:
        out[col] = out.groupby(DATE)[col].rank(pct=True, method="average") * 2 - 1
    return out


def get_models(seed: int = 7) -> dict[str, tuple[Any, list[dict[str, Any]]]]:
    """Return 13 model families and small, readable validation grids."""
    models: dict[str, tuple[Any, list[dict[str, Any]]]] = {
        "OLS": (make_pipeline(StandardScaler(), LinearRegression()), [{}]),
        "Lasso": (make_pipeline(StandardScaler(), Lasso(max_iter=10000)), [{"lasso__alpha": [0.0001, 0.001, 0.01]}]),
        "Ridge": (make_pipeline(StandardScaler(), Ridge()), [{"ridge__alpha": [0.1, 1.0, 10.0]}]),
        "Enet": (make_pipeline(StandardScaler(), ElasticNet(max_iter=10000)), [{"elasticnet__alpha": [0.001, 0.01], "elasticnet__l1_ratio": [0.25, 0.5, 0.75]}]),
        "PCR": (make_pipeline(StandardScaler(), PCA(), LinearRegression()), [{"pca__n_components": [0.8, 0.9, 0.95]}]),
        "PLS": (make_pipeline(StandardScaler(), PLSRegression()), [{"plsregression__n_components": [2, 5, 10]}]),
        "GBRT": (GradientBoostingRegressor(random_state=seed), [{"n_estimators": [100], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]}]),
        "RF": (RandomForestRegressor(random_state=seed, n_jobs=-1), [{"n_estimators": [200], "max_depth": [4, 8, None], "min_samples_leaf": [5, 20]}]),
    }
    for layers in range(1, 6):
        models[f"NN{layers}"] = (
            make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(32,) * layers, solver="adam", max_iter=500,
                early_stopping=True, random_state=seed,
            )),
            [{"mlpregressor__alpha": [0.0001, 0.001], "mlpregressor__learning_rate_init": [0.0001, 0.001]}],
        )
    return models


def oos_r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Out-of-sample R-squared against the paper's zero-return benchmark."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    den = np.sum(y ** 2)
    return float(1 - np.sum((y - pred) ** 2) / den) if den else np.nan


def _choose_model(model: Any, grid: list[dict[str, Any]], x: pd.DataFrame,
                  y: pd.Series, xv: pd.DataFrame, yv: pd.Series) -> Any:
    """Select the candidate with the smallest validation MSE."""
    best = None
    best_mse = np.inf
    for values in grid:
        candidate = clone(model).set_params(**values)
        try:
            candidate.fit(x, y)
            pred = np.asarray(candidate.predict(xv)).reshape(-1)
            score = mean_squared_error(yv, pred)
        except (ValueError, TypeError):
            continue
        if score < best_mse:
            best, best_mse = candidate, score
    if best is None:
        raise ValueError("No candidate model fit successfully; check sample size and features.")
    return best


def run_study(
    panel: pd.DataFrame,
    train_end: str = "2018-12-31",
    valid_start: str = "2019-01-01",
    valid_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
    test_end: str = "2022-12-31",
    features: list[str] | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Fit 13 models, select settings on validation data, and predict the test set."""
    need = {DATE, ID, TARGET}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = panel.copy()
    data[DATE] = pd.to_datetime(data[DATE])
    if features is None:
        features = [c for c in data.columns if c not in need]
    if not features:
        raise ValueError("Provide at least one predictor column.")
    data = rank_features(data, features).dropna(subset=features + [TARGET]).sort_values(DATE)

    train = data[data[DATE] <= pd.Timestamp(train_end)]
    valid = data[data[DATE].between(pd.Timestamp(valid_start), pd.Timestamp(valid_end))]
    test = data[data[DATE].between(pd.Timestamp(test_start), pd.Timestamp(test_end))]
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Train, validation, and test periods must all contain rows.")

    x = train[features]
    y = train[TARGET]
    xv = valid[features]
    yv = valid[TARGET]
    xt = test[features]
    yt = test[TARGET]
    combined_x = pd.concat([x, xv], ignore_index=True)
    combined_y = pd.concat([y, yv], ignore_index=True)

    rows = []
    preds = test[[DATE, ID, TARGET]].copy()
    for name, (model, grid) in get_models(seed).items():
        selected = _choose_model(model, grid, x, y, xv, yv)
        selected.fit(combined_x, combined_y)
        p = np.asarray(selected.predict(xt)).reshape(-1)
        preds[name] = p
        rows.append({"model": name, "oos_r2": oos_r2(yt.to_numpy(), p),
                     "test_mse": mean_squared_error(yt, p)})

    return {"metrics": pd.DataFrame(rows).sort_values("oos_r2", ascending=False),
            "predictions": preds, "features": features}


def long_short_summary(predictions: pd.DataFrame, model: str,
                       groups: int = 5) -> pd.DataFrame:
    """Summarize a simple equal-weighted top-minus-bottom portfolio by week."""
    if groups < 2:
        raise ValueError("groups must be at least 2")
    rows = []
    for date, part in predictions.groupby(DATE):
        part = part.dropna(subset=[model, TARGET]).copy()
        if len(part) < groups * 2:
            continue
        part["group"] = pd.qcut(part[model].rank(method="first"), groups, labels=False)
        top = part.loc[part["group"] == groups - 1, TARGET].mean()
        bottom = part.loc[part["group"] == 0, TARGET].mean()
        rows.append({DATE: date, "top_return": top, "bottom_return": bottom,
                     "top_minus_bottom": top - bottom})
    return pd.DataFrame(rows)
