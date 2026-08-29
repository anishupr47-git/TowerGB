"""Finite-difference numerical gradient verification for TowerGB engine.

Verifies that the analytical gradient ∂L/∂W produced by
``cross_entropy_loss_and_grad`` matches the numerical central-difference
approximation to within machine-precision tolerances.

Central difference formula:

    ∂L/∂W_{ij} ≈ [L(W + h·e_{ij}) − L(W − h·e_{ij})] / (2h)

Error analysis:
    • Truncation error: O(h²) for central differences
    • Rounding error: O(ε_mach / h)
    • Optimal h ≈ ε_mach^{1/3} ≈ 6×10⁻⁶ for float64
    • With h = 10⁻⁵, expected accuracy ≈ 10⁻¹⁰
"""

from __future__ import annotations

import numpy as np

from towergb._engine import cross_entropy_loss_and_grad


# ── Helpers ────────────────────────────────────────────────────────────
def _numerical_gradient(
    X: np.ndarray,
    Y_one_hot: np.ndarray,
    W: np.ndarray,
    temperature: float,
    h: float = 1e-5,
) -> np.ndarray:
    """Compute numerical gradient via central finite differences.

    Iterates over each element of W — acceptable here because this is a
    verification routine, not production code, and D×K is small.
    """
    D, K = W.shape
    num_grad: np.ndarray = np.zeros_like(W)

    for i in range(D):
        for j in range(K):
            W_plus: np.ndarray = W.copy()
            W_plus[i, j] += h
            loss_plus, _ = cross_entropy_loss_and_grad(
                X, Y_one_hot, W_plus, temperature
            )

            W_minus: np.ndarray = W.copy()
            W_minus[i, j] -= h
            loss_minus, _ = cross_entropy_loss_and_grad(
                X, Y_one_hot, W_minus, temperature
            )

            num_grad[i, j] = (loss_plus - loss_minus) / (2.0 * h)

    return num_grad


# ── Tests ──────────────────────────────────────────────────────────────
class TestGradientCheck:
    """Suite of gradient verification tests."""

    def test_gradient_matches_numerical_temperature_1(self) -> None:
        """Analytical gradient matches numerical at T=1.0 within 1e-7."""
        rng = np.random.RandomState(42)
        N, D, K = 20, 5, 3

        X = rng.randn(N, D).astype(np.float64)
        W = (rng.randn(D, K) * 0.1).astype(np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        temperature = 1.0
        _, analytical_grad = cross_entropy_loss_and_grad(
            X, Y_one_hot, W, temperature
        )
        numerical_grad = _numerical_gradient(X, Y_one_hot, W, temperature)

        max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
        assert max_diff < 1e-7, (
            f"Gradient check failed at T={temperature}: "
            f"max |analytical − numerical| = {max_diff:.2e}"
        )

    def test_gradient_matches_numerical_temperature_1_5(self) -> None:
        """Analytical gradient matches numerical at T=1.5 within 1e-7."""
        rng = np.random.RandomState(123)
        N, D, K = 30, 8, 4

        X = rng.randn(N, D).astype(np.float64)
        W = (rng.randn(D, K) * 0.05).astype(np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        temperature = 1.5
        _, analytical_grad = cross_entropy_loss_and_grad(
            X, Y_one_hot, W, temperature
        )
        numerical_grad = _numerical_gradient(X, Y_one_hot, W, temperature)

        max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
        assert max_diff < 1e-7, (
            f"Gradient check failed at T={temperature}: "
            f"max |analytical − numerical| = {max_diff:.2e}"
        )

    def test_gradient_matches_numerical_temperature_0_5(self) -> None:
        """Analytical gradient matches numerical at T=0.5 (sharp softmax)."""
        rng = np.random.RandomState(7)
        N, D, K = 15, 4, 2

        X = rng.randn(N, D).astype(np.float64)
        W = (rng.randn(D, K) * 0.1).astype(np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        temperature = 0.5
        _, analytical_grad = cross_entropy_loss_and_grad(
            X, Y_one_hot, W, temperature
        )
        numerical_grad = _numerical_gradient(X, Y_one_hot, W, temperature)

        max_diff = float(np.max(np.abs(analytical_grad - numerical_grad)))
        assert max_diff < 1e-7, (
            f"Gradient check failed at T={temperature}: "
            f"max |analytical − numerical| = {max_diff:.2e}"
        )

    def test_gradient_zero_at_perfect_fit(self) -> None:
        """Gradient should approach zero when predictions equal targets."""
        rng = np.random.RandomState(99)
        N, D, K = 10, 3, 2

        # Construct W so that X @ W gives perfect one-hot logits
        X = np.eye(N, D, dtype=np.float64)
        y_idx = rng.randint(0, K, size=N)
        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        # Make logits strongly point to the correct class
        W = np.zeros((D, K), dtype=np.float64)
        for i in range(min(N, D)):
            W[i, y_idx[i]] = 100.0  # Very large → softmax ≈ one-hot

        _, grad = cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature=1.0)

        # Gradient should be near-zero (saturated softmax)
        assert np.max(np.abs(grad)) < 1e-6, (
            f"Gradient not near-zero at perfect fit: max |grad| = "
            f"{np.max(np.abs(grad)):.2e}"
        )
