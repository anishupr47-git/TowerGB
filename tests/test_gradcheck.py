"""Tests to check that gradient math matches step-by-step numbers."""

from __future__ import annotations

import numpy as np

from towergb._engine import cross_entropy_loss_and_grad


def _numerical_gradient(X: np.ndarray, Y_one_hot: np.ndarray, W: np.ndarray, temperature: float, h: float = 1e-5) -> np.ndarray:
    D, K = W.shape
    num_grad = np.zeros_like(W)

    for i in range(D):
        for j in range(K):
            W_plus = W.copy()
            W_plus[i, j] += h
            loss_plus, _ = cross_entropy_loss_and_grad(X, Y_one_hot, W_plus, temperature)

            W_minus = W.copy()
            W_minus[i, j] -= h
            loss_minus, _ = cross_entropy_loss_and_grad(X, Y_one_hot, W_minus, temperature)

            num_grad[i, j] = (loss_plus - loss_minus) / (2.0 * h)

    return num_grad


class TestGradientCheck:
    """Make sure analytical formula matches the manual slope checks."""

    def test_gradient_matches_numerical_temperature_1(self) -> None:
        rng = np.random.RandomState(42)
        N, D, K = 20, 5, 3

        X = rng.randn(N, D).astype(np.float64)
        W = (rng.randn(D, K) * 0.1).astype(np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        temperature = 1.0
        _, analytical_grad = cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature)
        numerical_grad = _numerical_gradient(X, Y_one_hot, W, temperature)

        max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
        assert max_diff < 1e-7

    def test_gradient_matches_numerical_temperature_1_5(self) -> None:
        rng = np.random.RandomState(123)
        N, D, K = 30, 8, 4

        X = rng.randn(N, D).astype(np.float64)
        W = (rng.randn(D, K) * 0.05).astype(np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        temperature = 1.5
        _, analytical_grad = cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature)
        numerical_grad = _numerical_gradient(X, Y_one_hot, W, temperature)

        max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
        assert max_diff < 1e-7

    def test_gradient_matches_numerical_temperature_0_5(self) -> None:
        rng = np.random.RandomState(7)
        N, D, K = 15, 4, 2

        X = rng.randn(N, D).astype(np.float64)
        W = (rng.randn(D, K) * 0.1).astype(np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        temperature = 0.5
        _, analytical_grad = cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature)
        numerical_grad = _numerical_gradient(X, Y_one_hot, W, temperature)

        max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
        assert max_diff < 1e-7

    def test_gradient_zero_at_perfect_fit(self) -> None:
        rng = np.random.RandomState(99)
        N, D, K = 10, 3, 2

        X = np.eye(N, D, dtype=np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        W = np.zeros((D, K), dtype=np.float64)
        for i in range(min(N, D)):
            W[i, y_idx[i]] = 100.0

        _, grad = cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature=1.0)
        assert np.max(np.abs(grad)) < 1e-6
