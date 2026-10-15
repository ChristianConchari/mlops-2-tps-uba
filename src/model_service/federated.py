"""Aprendizaje federado sobre el dataset de `iris-rf` (Mini-TP 5).

FedAvg necesita un modelo cuyos parámetros se puedan promediar, y un
RandomForest no los tiene. Por eso se federa un clasificador softmax sobre las
mismas 4 features y el mismo split (80/20, `random_state=42`) que usa
`train.py`: el RF queda como referencia centralizada y el softmax es lo que
viaja entre clientes. Solo se intercambian pesos, nunca filas del dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

N_CLS = 3
N_FEAT = 4


def load_data():
    """Mismo split que train.py; el scaler se ajusta solo con el train."""
    X, y = load_iris(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def init_model():
    return np.zeros((N_FEAT, N_CLS)), np.zeros(N_CLS)


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


def sgd_step(W, b, X, y, lr):
    D = _softmax(X @ W + b)
    D[np.arange(len(y)), y] -= 1
    D /= len(y)
    return W - lr * (X.T @ D), b - lr * D.sum(0)


def accuracy(W, b, X, y):
    return float((np.argmax(X @ W + b, 1) == y).mean())


def train_centralized(X, y, epochs=200, lr=0.5):
    W, b = init_model()
    for _ in range(epochs):
        W, b = sgd_step(W, b, X, y, lr)
    return W, b


def partition_iid(X, y, k, rng):
    idx = rng.permutation(len(X))
    return [(X[c], y[c]) for c in np.array_split(idx, k)]


def partition_non_iid(X, y, k, rng, classes_per_client=1):
    """Ordena por clase y reparte shards contiguos: cada cliente ve pocas clases."""
    shards = np.array_split(np.argsort(y, kind="stable"), k * classes_per_client)
    rng.shuffle(shards)
    out = []
    for i in range(k):
        idx = np.concatenate(shards[i * classes_per_client:(i + 1) * classes_per_client])
        out.append((X[idx], y[idx]))
    return out


def local_update(W, b, X, y, epochs, lr):
    for _ in range(epochs):
        W, b = sgd_step(W, b, X, y, lr)
    return W, b


def fedavg(updates):
    """Promedio ponderado por n_k: w = sum(n_k / n * w_k)."""
    n = sum(nk for _, _, nk in updates)
    return (
        sum(nk / n * Wk for Wk, _, nk in updates),
        sum(nk / n * bk for _, bk, nk in updates),
    )


def _clip(dW, db, S):
    norm = np.sqrt((dW**2).sum() + (db**2).sum())
    f = min(1.0, S / (norm + 1e-12))
    return dW * f, db * f


def run_federated(clients, Xte, yte, rng, rounds=20, epochs=10, lr=0.5,
                  frac=1.0, clip=None, sigma=0.0):
    """Corre FedAvg y devuelve (W, b, accuracy por ronda).

    Con `clip` y `sigma` es DP-FedAvg: cada update (delta respecto del modelo
    global) se recorta a norma L2 `clip` y al promedio se le suma ruido
    gaussiano de desvío `sigma * clip / m`, con m clientes por ronda.
    """
    W, b = init_model()
    m = max(1, int(frac * len(clients)))
    hist = []
    for _ in range(rounds):
        sel = rng.choice(len(clients), m, replace=False)
        ups = []
        for i in sel:
            Xc, yc = clients[i]
            Wk, bk = local_update(W, b, Xc, yc, epochs, lr)
            if clip is not None:
                dW, db = _clip(Wk - W, bk - b, clip)
                Wk, bk = W + dW, b + db
            ups.append((Wk, bk, len(Xc)))
        W, b = fedavg(ups)
        if sigma > 0:
            std = sigma * (clip or 1.0) / m
            W = W + rng.normal(0, std, W.shape)
            b = b + rng.normal(0, std, b.shape)
        hist.append(accuracy(W, b, Xte, yte))
    return W, b, hist
