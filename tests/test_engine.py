"""Tests for the engine math functions."""

from __future__ import annotations

import numpy as np

from towergb._engine import (
    compute_risk_metrics,
    cross_entropy_loss_and_grad,
    pareto_arbiter,
    stable_softmax,
)


class TestStableSoftmax:
    """Tests for softmax probability calculation."""

    def test_extreme_positive_logits(self) -> None:
        # Should not crash or overflow when numbers are very large
        Z = np.array([[1e4, 1e4 + 1.0, 1e4 - 1.0]], dtype=np.float64)
        P = stable_softmax(Z)

        assert np.all(np.isfinite(P)), "Non-finite values in softmax output"
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        # Bigger input score should give bigger probability
        assert P[0, 1] > P[0, 0] > P[0, 2]

    def test_extreme_negative_logits(self) -> None:
        # Should not crash when numbers are very small negative
        Z = np.array([[-1e4, -1e4 + 1.0, -1e4 - 1.0]], dtype=np.float64)
        P = stable_softmax(Z)

        assert np.all(np.isfinite(P)), "Non-finite values in softmax output"
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_mixed_extreme_logits(self) -> None:
        # Should handle mixed big and small numbers
        Z = np.array([[-1e4, 0.0, 1e4]], dtype=np.float64)
        P = stable_softmax(Z)

        assert np.all(np.isfinite(P))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert P[0, 2] > 0.99

    def test_uniform_logits(self) -> None:
        # Equal numbers should give equal chances
        Z = np.ones((5, 4), dtype=np.float64) * 42.0
        P = stable_softmax(Z)

        np.testing.assert_allclose(P, 0.25, atol=1e-12)

    def test_row_stochastic(self) -> None:
        # Each row should always add up to 1
        rng = np.random.RandomState(42)
        Z = rng.randn(100, 10).astype(np.float64)
        P = stable_softmax(Z)

        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(P >= 0.0), "Negative probabilities detected"

    def test_temperature_scaling(self) -> None:
        # Higher temperature makes chances more spread out
        Z = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)

        P_cold = stable_softmax(Z, temperature=0.1)
        P_warm = stable_softmax(Z, temperature=1.0)
        P_hot = stable_softmax(Z, temperature=10.0)

        assert P_cold.max() > P_warm.max() > P_hot.max()

    def test_single_sample(self) -> None:
        # Should work for one row
        Z = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        P = stable_softmax(Z)

        assert P.shape == (1, 3)
        np.testing.assert_allclose(P.sum(), 1.0, atol=1e-12)


class TestCrossEntropyLoss:
    """Tests for loss and gradient calculation."""

    def test_loss_is_positive(self) -> None:
        # Mistake score must never be negative
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
        # Mistake score should go down as we train
        rng = np.random.RandomState(42)
        N, D, K = 30, 5, 3
        X = rng.randn(N, D).astype(np.float64)
        W = rng.randn(D, K).astype(np.float64) * 0.1
        y_idx = rng.randint(0, K, size=N)
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), y_idx] = 1.0

        initial_loss, _ = cross_entropy_loss_and_grad(X, Y, W)

        # Take 50 training steps
        for _ in range(50):
            _, grad = cross_entropy_loss_and_grad(X, Y, W)
            W -= 0.1 * grad

        final_loss, _ = cross_entropy_loss_and_grad(X, Y, W)
        assert final_loss < initial_loss, (
            f"Loss didn't decrease: {initial_loss:.6f} -> {final_loss:.6f}"
        )

    def test_gradient_shape(self) -> None:
        # Gradient size must match weight matrix size
        N, D, K = 20, 7, 4
        rng = np.random.RandomState(42)
        X = rng.randn(N, D).astype(np.float64)
        W = rng.randn(D, K).astype(np.float64)
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), rng.randint(0, K, size=N)] = 1.0

        _, grad = cross_entropy_loss_and_grad(X, Y, W)
        assert grad.shape == (D, K)


class TestRiskMetrics:
    """Tests for risk metrics."""

    def test_perfect_predictions(self) -> None:
        # When guesses are perfect, mistakes should be close to zero
        N, K = 10, 3
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), np.arange(N) % K] = 1.0

        P = Y * (1.0 - 1e-10) + (1.0 - Y) * (1e-10 / (K - 1))

        metrics = compute_risk_metrics(P, Y)

        assert metrics["log_loss"] < 1e-8
        assert metrics["brier"] < 1e-8
        assert metrics["loss_var"] < 1e-14

    def test_uniform_predictions(self) -> None:
        # Equal guessing should give predictable log loss
        N, K = 100, 4
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), np.arange(N) % K] = 1.0

        P = np.ones((N, K), dtype=np.float64) / K

        metrics = compute_risk_metrics(P, Y)

        expected_log_loss = np.log(K)
        np.testing.assert_allclose(
            metrics["log_loss"], expected_log_loss, rtol=1e-10
        )
        np.testing.assert_allclose(metrics["loss_var"], 0.0, atol=1e-14)

    def test_all_metrics_finite(self) -> None:
        # Metrics should always be valid real numbers
        rng = np.random.RandomState(42)
        N, K = 50, 5
        Y = np.zeros((N, K), dtype=np.float64)
        Y[np.arange(N), rng.randint(0, K, size=N)] = 1.0

        P = stable_softmax(rng.randn(N, K))

        metrics = compute_risk_metrics(P, Y)
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(v >= 0 for v in metrics.values())


class TestParetoArbiter:
    """Tests for arbiter round weighting."""

    def test_weights_sum_to_one(self) -> None:
        # All round voting percentages must add up to 1
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
        # The best model round should get the biggest vote
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
        # Equal rounds should get equal votes
        metrics = {"accuracy": 0.8, "log_loss": 0.4, "brier": 0.3, "loss_var": 0.1}
        metrics_list = [metrics.copy() for _ in range(5)]
        weights = {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}

        ensemble_w = pareto_arbiter(metrics_list, weights)

        np.testing.assert_allclose(ensemble_w, 0.2, atol=1e-12)

    def test_single_pass(self) -> None:
        # One round gets 100% of the vote
        metrics_list = [
            {"accuracy": 0.9, "log_loss": 0.2, "brier": 0.1, "loss_var": 0.05}
        ]
        weights = {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}

        ensemble_w = pareto_arbiter(metrics_list, weights)

        np.testing.assert_allclose(ensemble_w, [1.0], atol=1e-12)
