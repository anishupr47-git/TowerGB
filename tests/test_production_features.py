"""Tests for production math features: polynomial expansion, Adam optimizer,
early stopping, gradient clipping, normalization, vector calibration,
epistemic uncertainty, and NaN handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from towergb import TowerGBClassifier


class TestProductionFeatures:
    """Test suite for the 8 enterprise math features."""

    def test_polynomial_expansion_xor(self) -> None:
        """Verify degree-2 polynomial expansion solves the non-linear XOR problem."""
        X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float64)
        y = np.array([0, 1, 1, 0])

        clf = TowerGBClassifier(
            interaction_degree=2,
            max_iter=500,
            learning_rate=0.2,
            random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        np.testing.assert_array_equal(preds, y)

    def test_builtin_normalization(self) -> None:
        """Verify built-in normalization handles wildly unscaled features."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        X[:, 0] *= 1e6  # unscaled large feature
        y = (X[:, 0] / 1e6 + X[:, 1] > 0).astype(int)

        clf = TowerGBClassifier(normalize=True, random_state=42)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.85

    def test_nan_handling(self) -> None:
        """Verify automatic NaN imputation and indicator creation."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Inject NaNs
        X[5, 0] = np.nan
        X[12, 2] = np.nan

        clf = TowerGBClassifier(handle_missing=True, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == 100

        # Predict on test set with NaNs
        X_test = rng.randn(20, 4)
        X_test[2, 0] = np.nan
        proba = clf.predict_proba(X_test)
        assert proba.shape == (20, 2)
        assert not np.isnan(proba).any()

    def test_gradient_clipping(self) -> None:
        """Verify max_grad_norm prevents exploding gradients on extreme inputs."""
        X = np.array([[1e5, 1e5], [-1e5, -1e5]], dtype=np.float64)
        y = np.array([1, 0])

        clf = TowerGBClassifier(
            max_grad_norm=1.0,
            max_iter=10,
            learning_rate=0.01,
            random_state=42,
        )
        clf.fit(X, y)
        assert np.all(np.isfinite(clf._W_ensemble))

    def test_early_stopping(self) -> None:
        """Verify validation-based early stopping stops before max_iter."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = rng.randint(0, 2, size=200)

        clf = TowerGBClassifier(
            max_iter=500,
            early_stop_fraction=0.2,
            tol=1e-3,
            random_state=42,
        )
        clf.fit(X, y)
        assert np.all(clf.n_iter_ < 500)

    def test_vector_calibration(self) -> None:
        """Verify vector (Platt-style per-class) calibration."""
        rng = np.random.RandomState(42)
        X = rng.randn(150, 4)
        y = rng.randint(0, 3, size=150)

        clf = TowerGBClassifier(n_passes=3, random_state=42)
        clf.fit(X[:100], y[:100])

        clf.calibrate(X[100:], y[100:], method="vector")
        assert clf._calibration_method_ == "vector"
        proba = clf.predict_proba(X[100:])
        assert proba.shape == (50, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

    def test_epistemic_uncertainty(self) -> None:
        """Verify epistemic uncertainty returns non-negative per-sample scores."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        y = rng.randint(0, 2, size=100)

        clf = TowerGBClassifier(n_passes=5, random_state=42)
        clf.fit(X, y)

        uncertainty = clf.predict_uncertainty(X)
        assert uncertainty.shape == (100,)
        assert np.all(uncertainty >= 0.0)
        assert np.all(np.isfinite(uncertainty))
