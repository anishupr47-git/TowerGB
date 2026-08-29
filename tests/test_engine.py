"""Unit tests for the TowerGB core computational engine.

Covers:
    • Softmax numerical stability under extreme inputs (Z > 10⁴)
    • Softmax mathematical properties (row-stochastic, monotonic)
    • Cross-entropy loss correctness
    • Risk metric computation
    • Pareto arbiter weighting sanity
"""

from __future__ import annotations

import numpy as np

from towergb._engine import (
    compute_risk_metrics,
    cross_entropy_loss_and_grad,
    pareto_arbiter,
    stable_softmax,
)


# ══════════════════════════════════════════════════════════════════════
# Softmax Tests
# ══════════════════════════════════════════════════════════════════════
class TestStableSoftmax:
    """Verify softmax stability and mathematical properties."""

    def test_extreme_positive_logits(self) -> None:
        """Softmax should not overflow for Z > 10⁴."""
        Z = np.array([[1e4, 1e4 + 1.0, 1e4 - 1.0]], dtype=np.float64)
        P = stable_softmax(Z)

        assert np.all(np.isfinite(P)), "Non-finite values in softmax output"
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        # Monotonicity: largest logit → largest probability
        assert P[0, 1] > P[0, 0] > P[0, 2]

    def test_extreme_negative_logits(self) -> None:
        """Softmax should not underflow for Z < -10⁴."""
        Z = np.array([[-1e4, -1e4 + 1.0, -1e4 - 1.0]], dtype=np.float64)
        P = stable_softmax(Z)

        assert np.all(np.isfinite(P)), "Non-finite values in softmax output"
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_mixed_extreme_logits(self) -> None:
        """Softmax handles mixed extreme magnitudes."""
        Z = np.array([[-1e4, 0.0, 1e4]], dtype=np.float64)
        P = stable_softmax(Z)

        assert np.all(np.isfinite(P))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        # The max-logit class should dominate
        assert P[0, 2] > 0.99

    def test_uniform_logits(self) -> None:
        """Equal logits should produce uniform distribution."""
        Z = np.ones((5, 4), dtype=np.float64) * 42.0
        P = stable_softmax(Z)

        np.testing.assert_allclose(P, 0.25, atol=1e-12)

    def test_row_stochastic(self) -> None:
        """Every row must sum to 1."""
        rng = np.random.RandomState(42)
        Z = rng.randn(100, 10).astype(np.float64)
        P = stable_softmax(Z)

        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(P >= 0.0), "Negative probabilities detected"

    def test_temperature_scaling(self) -> None:
        """Higher temperature → softer distribution (closer to uniform)."""
        Z = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)

        P_cold = stable_softmax(Z, temperature=0.1)
        P_warm = stable_softmax(Z, temperature=1.0)
        P_hot = stable_softmax(Z, temperature=10.0)

        # Entropy should increase with temperature
        # Max probability should decrease with temperature
        assert P_cold.max() > P_warm.max() > P_hot.max()

    def test_single_sample(self) -> None:
        """Works correctly for a single sample."""
        Z = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        P = stable_softmax(Z)

        assert P.shape == (1, 3)
        np.testing.assert_allclose(P.sum(), 1.0, atol=1e-12)


# ══════════════════════════════════════════════════════════════════════
# Cross-Entropy Loss Tests
# ══════════════════════════════════════════════════════════════════════
class TestCrossEntropyLoss:
    """Verify loss and gradient computation."""

    def test_loss_is_positive(self) -> None:
        """Cross-entropy loss must always be non-negative."""
        rng = np.random.RandomState(42)
        N, D, K = 50, 10, 5
        X = rng.randn(N, D).astype(np.float64)
        W = rng.randn(D, K).astype(np.float64) * 0.1
        y_idx = rng.randint(0, K, size=N)
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), y_idx] = 1.0

        loss, _ = cross_entropy_loss_and_grad(X, Y, W)
        assert loss >= 0.0, f"Negative loss: {loss}"

    def test_loss_decreases_with_training(self) -> None:
        """Gradient descent should decrease loss."""
        rng = np.random.RandomState(42)
        N, D, K = 30, 5, 3
        X = rng.randn(N, D).astype(np.float64)
        W = rng.randn(D, K).astype(np.float64) * 0.1
        y_idx = rng.randint(0, K, size=N)
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), y_idx] = 1.0

        initial_loss, _ = cross_entropy_loss_and_grad(X, Y, W)

        # 50 steps of gradient descent
        for _ in range(50):
            _, grad = cross_entropy_loss_and_grad(X, Y, W)
            W -= 0.1 * grad

        final_loss, _ = cross_entropy_loss_and_grad(X, Y, W)
        assert final_loss < initial_loss, (
            f"Loss didn't decrease: {initial_loss:.6f} → {final_loss:.6f}"
        )

    def test_gradient_shape(self) -> None:
        """Gradient must have same shape as weight matrix."""
        N, D, K = 20, 7, 4
        rng = np.random.RandomState(42)
        X = rng.randn(N, D).astype(np.float64)
        W = rng.randn(D, K).astype(np.float64)
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), rng.randint(0, K, size=N)] = 1.0

        _, grad = cross_entropy_loss_and_grad(X, Y, W)
        assert grad.shape == (D, K)


# ══════════════════════════════════════════════════════════════════════
# Risk Metrics Tests
# ══════════════════════════════════════════════════════════════════════
class TestRiskMetrics:
    """Verify Tower 2 risk metric computation."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions → log_loss ≈ 0, brier ≈ 0, var ≈ 0."""
        N, K = 10, 3
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), np.arange(N) % K] = 1.0

        # Near-perfect predictions (not exactly one-hot to avoid log(0))
        P = Y * (1.0 - 1e-10) + (1.0 - Y) * (1e-10 / (K - 1))

        metrics = compute_risk_metrics(P, Y)

        assert metrics["log_loss"] < 1e-8
        assert metrics["brier"] < 1e-8
        assert metrics["loss_var"] < 1e-14

    def test_uniform_predictions(self) -> None:
        """Uniform predictions should have log_loss = ln(K)."""
        N, K = 100, 4
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), np.arange(N) % K] = 1.0

        P = np.ones((N, K), dtype=np.float64) / K

        metrics = compute_risk_metrics(P, Y)

        expected_log_loss = np.log(K)
        np.testing.assert_allclose(
            metrics["log_loss"], expected_log_loss, rtol=1e-10
        )
        # Variance should be zero since all samples get same loss
        np.testing.assert_allclose(metrics["loss_var"], 0.0, atol=1e-14)

    def test_all_metrics_finite(self) -> None:
        """Metrics should be finite for valid probability matrices."""
        rng = np.random.RandomState(42)
        N, K = 50, 5
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), rng.randint(0, K, size=N)] = 1.0

        # Random softmax probabilities
        P = stable_softmax(rng.randn(N, K))

        metrics = compute_risk_metrics(P, Y)
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(v >= 0 for v in metrics.values())


# ══════════════════════════════════════════════════════════════════════
# Pareto Arbiter Tests
# ══════════════════════════════════════════════════════════════════════
class TestParetoArbiter:
    """Verify arbiter weighting behavior."""

    def test_weights_sum_to_one(self) -> None:
        """Ensemble weights must be a valid probability vector."""
        metrics_list = [
            {"accuracy": 0.8, "log_loss": 0.5, "brier": 0.3, "loss_var": 0.1},
            {"accuracy": 0.7, "log_loss": 0.6, "brier": 0.4, "loss_var": 0.2},
            {"accuracy": 0.9, "log_loss": 0.3, "brier": 0.2, "loss_var": 0.05},
        ]
        weights = {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}

        ensemble_w = pareto_arbiter(metrics_list, weights)

        np.testing.assert_allclose(ensemble_w.sum(), 1.0, atol=1e-12)
        assert np.all(ensemble_w >= 0.0)

    def test_best_pass_gets_highest_weight(self) -> None:
        """The pass with best metrics should receive the most weight."""
        metrics_list = [
            {"accuracy": 0.5, "log_loss": 1.0, "brier": 0.8, "loss_var": 0.5},
            {"accuracy": 0.95, "log_loss": 0.1, "brier": 0.05, "loss_var": 0.01},
            {"accuracy": 0.6, "log_loss": 0.8, "brier": 0.6, "loss_var": 0.3},
        ]
        weights = {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}

        ensemble_w = pareto_arbiter(metrics_list, weights)

        assert np.argmax(ensemble_w) == 1, (
            f"Best pass (index 1) should have highest weight, "
            f"got argmax={np.argmax(ensemble_w)}"
        )

    def test_identical_passes_equal_weights(self) -> None:
        """Identical passes should receive equal weights."""
        metrics = {"accuracy": 0.8, "log_loss": 0.4, "brier": 0.3, "loss_var": 0.1}
        metrics_list = [metrics.copy() for _ in range(5)]
        weights = {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}

        ensemble_w = pareto_arbiter(metrics_list, weights)

        np.testing.assert_allclose(ensemble_w, 0.2, atol=1e-12)

    def test_single_pass(self) -> None:
        """Single pass should get weight 1.0."""
        metrics_list = [
            {"accuracy": 0.9, "log_loss": 0.2, "brier": 0.1, "loss_var": 0.05}
        ]
        weights = {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}

        ensemble_w = pareto_arbiter(metrics_list, weights)

        np.testing.assert_allclose(ensemble_w, [1.0], atol=1e-12)
