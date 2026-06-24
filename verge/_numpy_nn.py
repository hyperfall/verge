"""A minimal numpy NN substrate for the learned *seams* (standard-backprop baseline).

These are the small learned components the spec actually calls for — projection heads,
the router's salience weights, linear transitions — NOT foundation models (constraint
§2.1: never train a foundation model from scratch). Keeping them in numpy lets the whole
stack run and be tested on CPU with no GPU and no heavy deps. When torch is present, the
same interfaces can be backed by it; the contract is identical.

Everything here is deliberately small, explicit, and unit-tested.
"""
from __future__ import annotations

import numpy as np


def unit_norm(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class Linear:
    """A linear map y = X W^T (+ b) with explicit MSE gradient steps (plain backprop).

    Initialized near-isometric (scaled Gaussian) so an untrained head already produces
    well-spread, full-rank outputs — the no-collapse floor holds before any training.
    """

    def __init__(self, d_in: int, d_out: int, *, seed: int = 0, bias: bool = False):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_out, d_in)).astype(np.float64) / np.sqrt(d_in)
        self.b = np.zeros(d_out) if bias else None

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = X @ self.W.T
        if self.b is not None:
            y = y + self.b
        return y

    __call__ = forward

    def apply_output_grad(self, X: np.ndarray, dy: np.ndarray, lr: float = 0.1) -> None:
        """Backprop an arbitrary upstream gradient dy = dL/d(forward(X)) into W.
        dL/dW = dy^T @ X / n. Lets callers combine multiple loss terms (alignment +
        VICReg) computed in output space."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        dy = np.atleast_2d(np.asarray(dy, dtype=np.float64))
        n = X.shape[0]
        self.W -= lr * (dy.T @ X / n)
        if self.b is not None:
            self.b -= lr * dy.mean(axis=0)

    def mse_step(self, X: np.ndarray, target: np.ndarray, lr: float = 0.1) -> float:
        """One gradient step of 0.5*||forward(X) - target||^2. Returns the loss before
        the step. Gradient: dL/dW = (pred - target)^T @ X / n."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        target = np.atleast_2d(np.asarray(target, dtype=np.float64))
        pred = self.forward(X)
        err = pred - target
        loss = 0.5 * float(np.mean(np.sum(err ** 2, axis=1)))
        n = X.shape[0]
        gW = err.T @ X / n
        self.W -= lr * gW
        if self.b is not None:
            self.b -= lr * err.mean(axis=0)
        return loss


def vicreg_variance_push(y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gradient of the VICReg variance hinge wrt outputs y (n, d): pushes each dim's std
    up toward gamma. Used to keep a head off the collapsed manifold during training."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    std = np.sqrt(y.var(axis=0) + 1e-4)
    active = (std < gamma).astype(np.float64)            # only dims below the target
    centered = y - y.mean(axis=0, keepdims=True)
    # d std_j / d y_ij = centered_ij / (n * std_j); hinge gradient is -active * d std.
    return -active * centered / (n * std)
