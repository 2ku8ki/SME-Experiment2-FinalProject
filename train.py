"""
train.py
PCA-Kernel Ridge Regression 기반 2D 위치 추정 모델 학습 파일

실행 방법:
    python train.py

출력:
    model.pkl

주의:
    DH_FR1.mat 또는 InF_DH_FR1.mat 파일이 train.py와 같은 폴더에 있어야 한다.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import scipy.io as sio
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler
from sklearn.model_selection import cross_validate

RANDOM_STATE = 42
MODEL_PATH = "model.pkl"


def find_data_path() -> str:
    candidates = ["DH_FR1.mat", "InF_DH_FR1.mat"]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "DH_FR1.mat 또는 InF_DH_FR1.mat 파일을 train.py와 같은 폴더에 넣어주세요."
    )


def load_dataset(path: str):
    mat = sio.loadmat(path, squeeze_me=False)
    d_hat = np.asarray(mat["d_hat"], dtype=float)
    p = np.asarray(mat["p"], dtype=float)
    bs_positions = np.asarray(mat["BS_positions"], dtype=float)

    X = d_hat.T
    Y = p.T
    return X, Y, bs_positions


def position_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred, axis=1)


def mean_position_error(y_true, y_pred):
    return float(np.mean(position_error(y_true, y_pred)))


def rmse_position_error(y_true, y_pred):
    errors = position_error(y_true, y_pred)
    return float(np.sqrt(np.mean(errors ** 2)))


def p90_position_error(y_true, y_pred):
    return float(np.percentile(position_error(y_true, y_pred), 90))


def max_position_error(y_true, y_pred):
    return float(np.max(position_error(y_true, y_pred)))


class WLSRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, bs_positions, weight_power=2.0):
        self.bs_positions = bs_positions
        self.weight_power = weight_power

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array([self._predict_one(row) for row in X], dtype=float)

    def _predict_one(self, distances):
        bs = np.asarray(self.bs_positions, dtype=float)
        bx = bs[0]
        by = bs[1]

        ref = int(np.argmin(distances))
        idx = [i for i in range(len(distances)) if i != ref]

        A = []
        b = []
        for i in idx:
            A.append([2.0 * (bx[i] - bx[ref]), 2.0 * (by[i] - by[ref])])
            b.append(
                bx[i] ** 2 - bx[ref] ** 2
                + by[i] ** 2 - by[ref] ** 2
                + distances[ref] ** 2 - distances[i] ** 2
            )

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        w = 1.0 / (np.maximum(distances[idx], 1e-6) ** self.weight_power)
        sqrt_w = np.sqrt(w)
        Aw = A * sqrt_w[:, None]
        bw = b * sqrt_w

        try:
            pred = np.linalg.lstsq(Aw, bw, rcond=None)[0]
        except np.linalg.LinAlgError:
            pred = np.array([0.0, 0.0])

        return pred


def evaluate_model(name, model, X, Y, cv):
    scorer = make_scorer(p90_position_error, greater_is_better=False)
    scores = cross_val_score(model, X, Y, scoring=scorer, cv=cv, n_jobs=-1)
    p90 = -float(np.mean(scores))
    print(f"{name:14s}: {p90:8.4f} m")
    return p90


def tune_and_evaluate_baseline(name, estimator, param_grid, X, Y, cv):
    scorer = make_scorer(p90_position_error, greater_is_better=False)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, Y)
    best_p90 = -float(search.best_score_)
    print(f"{name:14s}: {best_p90:8.4f} m")
    print(f"  best params: {search.best_params_}")
    return best_p90


def main():
    data_path = find_data_path()
    X, Y, bs_positions = load_dataset(data_path)

    print(f"[Data] path = {data_path}")
    print(f"[Data] X shape = {X.shape}, Y shape = {Y.shape}")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n[Baseline comparison: 5-Fold CV P90 position error]")

    baseline_results = {}

    baseline_results["WLS"] = evaluate_model(
        "WLS",
        WLSRegressor(bs_positions=bs_positions, weight_power=2.0),
        X,
        Y,
        cv,
    )

    knn_matched = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=18, whiten=True)),
        ("knn", KNeighborsRegressor()),
    ])

    baseline_results["KNN-matched"] = tune_and_evaluate_baseline(
        "KNN-matched",
        knn_matched,
        {
            "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "knn__weights": ["uniform", "distance"],
        },
        X,
        Y,
        cv,
    )

    ridge_matched = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=18, whiten=True)),
        ("ridge", Ridge()),
    ])

    baseline_results["Ridge-matched"] = tune_and_evaluate_baseline(
        "Ridge-matched",
        ridge_matched,
        {
            "ridge__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
        X,
        Y,
        cv,
    )

    print("\n[Hyperparameter search: PCA-KRR]")

    pca_krr = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("krr", KernelRidge(kernel="rbf")),
    ])

    param_dist = {
        "log": [
            FunctionTransformer(np.log1p),
            FunctionTransformer(),
        ],
        "scaler": [StandardScaler(), RobustScaler()],
        "pca__n_components": [6, 8, 10, 12, 14, 16, 18],
        "pca__whiten": [True, False],
        "krr__alpha": loguniform(1e-4, 100),
        "krr__gamma": loguniform(1e-4, 0.5),
    }

    scorer = make_scorer(p90_position_error, greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=pca_krr,
        param_distributions=param_dist,
        n_iter=150,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        random_state=RANDOM_STATE,
    )
    search.fit(X, Y)

    best_model = search.best_estimator_
    best_cv_p90 = -float(search.best_score_)
    baseline_results["PCA-KRR"] = best_cv_p90

    print("\n[Best PCA-KRR]")
    print(f"Best CV P90 position error: {best_cv_p90:.4f} m")
    print("Best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    train_pred = best_model.predict(X)
    train_mean = mean_position_error(Y, train_pred)
    train_rmse = rmse_position_error(Y, train_pred)
    train_p90 = p90_position_error(Y, train_pred)
    train_max = max_position_error(Y, train_pred)

    print("\n[Train-set diagnostic error of final refit model]")
    print(f"Train mean position error: {train_mean:.4f} m")
    print(f"Train RMSE position error: {train_rmse:.4f} m")
    print(f"Train P90 position error : {train_p90:.4f} m")
    print(f"Train max position error : {train_max:.4f} m")

    multi_scorers = {
        "mean": make_scorer(mean_position_error, greater_is_better=False),
        "p90": make_scorer(p90_position_error, greater_is_better=False),
    }

    cv_eval = cross_validate(
        best_model,
        X,
        Y,
        scoring=multi_scorers,
        cv=cv,
        n_jobs=-1,
    )

    print("\n[Selected PCA-KRR additional CV metrics]")
    print(f"CV mean position error: {-np.mean(cv_eval['test_mean']):.4f} m")
    print(f"CV P90 position error : {-np.mean(cv_eval['test_p90']):.4f} m")

    print("\n[Report table values (CV P90)]")
    for name, value in baseline_results.items():
        print(f"{name}: {value:.4f} m")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"\nSaved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()