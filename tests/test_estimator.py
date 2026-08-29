"""Estimator-level integration tests for TowerGBClassifier.

Covers:
    • Full sklearn.utils.estimator_checks.check_estimator compliance
    • Multi-class training and prediction
    • String target label round-trips
    • DataFrame input handling
    • Serialization (pickle / joblib)
    • Post-hoc temperature calibration
    • Reproducibility via random_state
"""

from __future__ import annotations

import pickle
import tempfile

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from towergb import TowerGBClassifier
from towergb.calibration import compute_ece


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def binary_data() -> tuple:
    """Binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2,
        n_informative=6, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def multiclass_data() -> tuple:
    """5-class classification dataset."""
    X, y = make_classification(
        n_samples=300, n_features=15, n_classes=5,
        n_informative=10, n_clusters_per_class=1, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ══════════════════════════════════════════════════════════════════════
# sklearn Estimator Compliance
# ══════════════════════════════════════════════════════════════════════
class TestSklearnCompliance:
    """Validate full check_estimator compliance."""

    def test_check_estimator(self) -> None:
        """Run sklearn's comprehensive estimator checks."""
        from sklearn.utils.estimator_checks import check_estimator

        # Use small n_passes and max_iter for speed
        clf = TowerGBClassifier(
            n_passes=2, max_iter=30, random_state=42
        )
        check_estimator(clf)


# ══════════════════════════════════════════════════════════════════════
# Core Functionality
# ══════════════════════════════════════════════════════════════════════
class TestFitPredict:
    """Test fit/predict/predict_proba pipeline."""

    def test_binary_classification(self, binary_data: tuple) -> None:
        """Train and predict on binary data."""
        X_train, X_test, y_train, y_test = binary_data

        clf = TowerGBClassifier(n_passes=3, max_iter=100, random_state=42)
        result = clf.fit(X_train, y_train)

        # fit returns self
        assert result is clf

        # Predictions have correct shape and type
        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert set(y_pred).issubset(set(clf.classes_))
        assert clf.score(X_test, y_test) > 0.5

        # Probabilities are valid
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(proba >= 0.0)

    def test_multiclass_classification(self, multiclass_data: tuple) -> None:
        """Train and predict on 5-class data."""
        X_train, X_test, y_train, y_test = multiclass_data

        clf = TowerGBClassifier(n_passes=5, max_iter=150, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        assert len(clf.classes_) == 5
        assert set(y_pred).issubset(set(clf.classes_))
        assert proba.shape == (len(X_test), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

        # Accuracy should be above random (20%) for informative features
        accuracy = clf.score(X_test, y_test)
        assert accuracy > 0.2, f"Accuracy {accuracy:.3f} ≤ random chance"

    def test_score_method(self, binary_data: tuple) -> None:
        """score() should return value in [0, 1]."""
        X_train, X_test, y_train, y_test = binary_data

        clf = TowerGBClassifier(n_passes=2, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        assert 0.0 <= score <= 1.0

    def test_classes_attribute(self) -> None:
        """classes_ should contain sorted unique labels."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([2, 0, 1, 2])

        clf = TowerGBClassifier(n_passes=2, max_iter=20, random_state=42)
        clf.fit(X, y)

        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_n_features_in(self) -> None:
        """n_features_in_ should match input dimensionality."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 7)
        y = rng.randint(0, 3, size=50)

        clf = TowerGBClassifier(n_passes=2, max_iter=20, random_state=42)
        clf.fit(X, y)

        assert clf.n_features_in_ == 7


# ══════════════════════════════════════════════════════════════════════
# String / Categorical Target Labels
# ══════════════════════════════════════════════════════════════════════
class TestStringTargets:
    """Verify correct handling of non-numeric labels."""

    def test_string_labels_roundtrip(self) -> None:
        """String labels should be preserved through fit → predict."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        labels = ["cat", "dog", "bird"]
        y = np.array([labels[i % 3] for i in range(60)])

        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert all(p in labels for p in y_pred), (
            f"Predicted labels {set(y_pred)} not subset of {set(labels)}"
        )
        assert clf.classes_.dtype.kind in ("U", "O")  # string or object

    def test_integer_string_labels(self) -> None:
        """String-encoded integers should work."""
        rng = np.random.RandomState(42)
        X = rng.randn(40, 4)
        y = np.array(["1", "2", "3", "1", "2", "3"] * 6 + ["1", "2", "3", "1"])

        clf = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert set(y_pred).issubset({"1", "2", "3"})


# ══════════════════════════════════════════════════════════════════════
# DataFrame Input
# ══════════════════════════════════════════════════════════════════════
class TestDataFrameInput:
    """Verify DataFrame/Series extraction."""

    def test_pandas_dataframe_input(self) -> None:
        """Should accept pandas DataFrame if pandas is installed."""
        pytest.importorskip("pandas")
        import pandas as pd

        rng = np.random.RandomState(42)
        X_arr = rng.randn(50, 5)
        y_arr = rng.randint(0, 3, size=50)

        X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
        y_series = pd.Series(y_arr, name="target")

        clf = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        clf.fit(X_df, y_series)

        # Predict with DataFrame
        y_pred = clf.predict(X_df)
        assert y_pred.shape == (50,)

        proba = clf.predict_proba(X_df)
        assert proba.shape == (50, 3)


# ══════════════════════════════════════════════════════════════════════
# Serialization
# ══════════════════════════════════════════════════════════════════════
class TestSerialization:
    """Verify pickle and joblib serialization."""

    def test_pickle_roundtrip(self, binary_data: tuple) -> None:
        """Pickled model should produce identical predictions."""
        X_train, X_test, y_train, _ = binary_data

        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        original_pred = clf.predict(X_test)
        original_proba = clf.predict_proba(X_test)

        # Pickle round-trip
        serialized = pickle.dumps(clf)
        clf_loaded = pickle.loads(serialized)

        np.testing.assert_array_equal(clf_loaded.predict(X_test), original_pred)
        np.testing.assert_allclose(
            clf_loaded.predict_proba(X_test), original_proba, atol=1e-14
        )

    def test_joblib_roundtrip(self, binary_data: tuple) -> None:
        """Joblib-serialized model should produce identical predictions."""
        joblib = pytest.importorskip("joblib")
        X_train, X_test, y_train, _ = binary_data

        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        original_proba = clf.predict_proba(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            joblib.dump(clf, f.name)
            clf_loaded = joblib.load(f.name)

        np.testing.assert_allclose(
            clf_loaded.predict_proba(X_test), original_proba, atol=1e-14
        )


# ══════════════════════════════════════════════════════════════════════
# Temperature Calibration
# ══════════════════════════════════════════════════════════════════════
class TestCalibration:
    """Verify post-hoc temperature calibration."""

    def test_calibrate_updates_temperature(self, binary_data: tuple) -> None:
        """calibrate() should update temperature_ attribute."""
        X_train, X_test, y_train, y_test = binary_data

        clf = TowerGBClassifier(n_passes=3, max_iter=100, random_state=42)
        clf.fit(X_train, y_train)

        assert clf.temperature_ == clf.temperature
        clf.calibrate(X_test, y_test)

        # Temperature should have been updated
        assert isinstance(clf.temperature_, float)
        assert clf.temperature_ > 0.0

    def test_calibrate_returns_self(self, binary_data: tuple) -> None:
        """calibrate() should return self for method chaining."""
        X_train, X_test, y_train, y_test = binary_data

        clf = TowerGBClassifier(n_passes=2, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)

        result = clf.calibrate(X_test, y_test)
        assert result is clf

    def test_calibration_reduces_or_maintains_ece(
        self, binary_data: tuple
    ) -> None:
        """Calibration should not worsen ECE significantly."""
        X_train, X_test, y_train, y_test = binary_data

        clf = TowerGBClassifier(n_passes=5, max_iter=150, random_state=42)
        clf.fit(X_train, y_train)

        proba_before = clf.predict_proba(X_test)
        ece_before = compute_ece(y_test, proba_before)

        clf.calibrate(X_test, y_test)

        proba_after = clf.predict_proba(X_test)
        ece_after = compute_ece(y_test, proba_after)

        # ECE should not increase significantly (allow small tolerance
        # for edge cases where optimization landscape is flat)
        assert ece_after <= ece_before + 0.05, (
            f"ECE worsened: {ece_before:.4f} → {ece_after:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════
class TestReproducibility:
    """Verify deterministic behavior with fixed random_state."""

    def test_same_seed_same_predictions(self) -> None:
        """Two fits with the same seed should yield identical predictions."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        y = rng.randint(0, 3, size=100)

        clf1 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=99)
        clf1.fit(X, y)

        clf2 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=99)
        clf2.fit(X, y)

        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))
        np.testing.assert_allclose(
            clf1.predict_proba(X), clf2.predict_proba(X), atol=1e-14
        )

    def test_different_seed_different_predictions(self) -> None:
        """Different seeds should (very likely) produce different weights."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        y = rng.randint(0, 3, size=100)

        clf1 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=1)
        clf1.fit(X, y)

        clf2 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=2)
        clf2.fit(X, y)

        # Ensemble weight matrices should differ
        assert not np.allclose(clf1._W_ensemble, clf2._W_ensemble, atol=1e-10)


# ══════════════════════════════════════════════════════════════════════
# Error Handling
# ══════════════════════════════════════════════════════════════════════
class TestErrorHandling:
    """Verify proper error messages for invalid inputs."""

    def test_predict_before_fit(self) -> None:
        """predict() on unfitted estimator should raise."""
        clf = TowerGBClassifier()
        with pytest.raises(NotFittedError):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_regression_target_raises(self) -> None:
        """Continuous float targets should raise ValueError."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        y = np.array([0.1, 0.5, 0.9])

        clf = TowerGBClassifier()
        with pytest.raises(ValueError, match="Unknown label type"):
            clf.fit(X, y)
