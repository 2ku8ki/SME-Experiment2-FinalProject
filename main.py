"""
main.py
학습된 PCA-KRR pipeline을 이용하여 사용자 위치 p_hat을 반환한다.

채점기 요구사항:
    - main() 함수가 있어야 한다.
    - main()은 numpy array 형태의 p_hat을 반환해야 한다.
    - p_hat shape은 (2, num_user)이어야 한다.
    - 사용자 수는 d_hat.shape[1]에서 동적으로 받아야 한다.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import scipy.io as sio


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "DH_FR1.mat")

_model = None


def _load_model():
    global _model
    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def run(d_hat: np.ndarray) -> np.ndarray:
    model = _load_model()

    d_hat = np.asarray(d_hat, dtype=float)

    if d_hat.ndim != 2:
        raise ValueError("d_hat must be a 2D array with shape (18, num_user).")
    if d_hat.shape[0] != 18:
        raise ValueError(f"Expected d_hat shape (18, num_user), got {d_hat.shape}.")

    X = d_hat.T
    y_hat = model.predict(X)
    p_hat = y_hat.T

    return np.asarray(p_hat, dtype=float)


def main() -> np.ndarray:
    data = sio.loadmat(DATA_PATH, squeeze_me=False)
    d_hat = np.asarray(data["d_hat"], dtype=float)

    p_hat = run(d_hat)

    num_user = d_hat.shape[1]
    if p_hat.shape != (2, num_user):
        raise ValueError(f"p_hat shape must be (2, {num_user}), got {p_hat.shape}.")

    return p_hat


if __name__ == "__main__":
    p_hat = main()
    print(p_hat.shape)

    data = sio.loadmat(DATA_PATH, squeeze_me=False)
    if "p" in data:
        p_true = np.asarray(data["p"], dtype=float)
        errors = np.linalg.norm(p_hat - p_true, axis=0)
        print(f"Mean position error: {np.mean(errors):.4f} m")
        print(f"RMSE position error: {np.sqrt(np.mean(errors ** 2)):.4f} m")
        print(f"P90 position error : {np.percentile(errors, 90):.4f} m")
        print(f"Max position error : {np.max(errors):.4f} m")