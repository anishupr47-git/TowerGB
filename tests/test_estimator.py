"""Tests for the TowerGBClassifier model."""

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


@pytest.fixture
def binary_data() -> tuple:
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, n_informative=6, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def multiclass_data() -> tuple:
    X, y = make_classification(
        n_samples=300, n_features=15, n_classes=5,
        n_informative=10, n_clusters_per_class=1, random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestSklearnCompliance:
    """Make sure TowerGB follows all standard scikit-learn rules."""

    def test_check_estimator(self) -> None:
        from sklearn.utils.estimator_checks import check_estimator
        clf = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        check_estimator(clf)


class TestFitPredict:
    """Test training and predicting."""

    def test_binary_classification(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        clf = TowerGBClassifier(n_passes=3, max_iter=100, random_state=42)
        result = clf.fit(X_train, y_train)

        assert result is clf
        y_pred = clf.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert set(y_pred).issubset(set(clf.classes_))
        assert clf.score(X_test, y_test) > 0.5

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert np.all(proba >= 0.0)

    def test_multiclass_classification(self, multiclass_data: tuple) -> None:
        X_train, X_test, y_train, y_test = multiclass_data
        clf = TowerGBClassifier(n_passes=5, max_iter=150, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)

        assert len(clf.classes_) == 5
        assert set(y_pred).issubset(set(clf.classes_))
        assert proba.shape == (len(X_test), 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert clf.score(X_test, y_test) > 0.2

    def test_score_method(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        clf = TowerGBClassifier(n_passes=2, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        assert 0.0 <= score <= 1.0

    def test_classes_attribute(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y = np.array([2, 0, 1, 2])
        clf = TowerGBClassifier(n_passes=2, max_iter=20, random_state=42)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])

    def test_n_features_in(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, 7)
        y = rng.randint(0, 3, size=50)
        clf = TowerGBClassifier(n_passes=2, max_iter=20, random_state=42)
        clf.fit(X, y)
        assert clf.n_features_in_ == 7


class TestFeatureImportancesAndCoef:
    """Test feature importance ranking and weight coefficients."""

    def test_feature_importances_shape_and_sum(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)

        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_.shape == (10,)
        np.testing.assert_allclose(clf.feature_importances_.sum(), 1.0, atol=1e-6)
        assert np.all(clf.feature_importances_ >= 0.0)

    def test_coef_attribute(self, binary_data: tuple, multiclass_data: tuple) -> None:
        X_bin, _, y_bin, _ = binary_data
        clf_bin = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        clf_bin.fit(X_bin, y_bin)
        assert hasattr(clf_bin, "coef_")
        assert clf_bin.coef_.shape == (1, 10)

        X_multi, _, y_multi, _ = multiclass_data
        clf_multi = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        clf_multi.fit(X_multi, y_multi)
        assert clf_multi.coef_.shape == (5, 15)


class TestClassWeightAndSampleWeight:
    """Test handling imbalanced classes and sample weights."""

    def test_class_weight_balanced(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        y = np.array([0] * 90 + [1] * 10)
        clf = TowerGBClassifier(n_passes=3, max_iter=50, class_weight="balanced", random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == 100

    def test_sample_weight_fit(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        weights = np.ones(len(y_train), dtype=np.float64)
        weights[:20] = 5.0
        clf = TowerGBClassifier(n_passes=2, max_iter=50, random_state=42)
        clf.fit(X_train, y_train, sample_weight=weights)
        assert clf.score(X_test, y_test) > 0.4


class TestL2Regularization:
    """Test weight shrinkage with L2 regularization."""

    def test_l2_regularization_effect(self, binary_data: tuple) -> None:
        X_train, _, y_train, _ = binary_data
        clf_noreg = TowerGBClassifier(n_passes=3, max_iter=100, l2_reg=0.0, random_state=42)
        clf_noreg.fit(X_train, y_train)

        clf_reg = TowerGBClassifier(n_passes=3, max_iter=100, l2_reg=1.0, random_state=42)
        clf_reg.fit(X_train, y_train)

        norm_noreg = float(np.linalg.norm(clf_noreg._W_ensemble))
        norm_reg = float(np.linalg.norm(clf_reg._W_ensemble))
        assert norm_reg < norm_noreg


class TestParallelTraining:
    """Test parallel bootstrap training with n_jobs."""

    def test_n_jobs_parallel(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        clf = TowerGBClassifier(n_passes=4, max_iter=50, n_jobs=2, random_state=42)
        clf.fit(X_train, y_train)
        assert clf.score(X_test, y_test) > 0.5


class TestStringTargets:
    """Test using text words as labels like cat, dog, bird."""

    def test_string_labels_roundtrip(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(60, 5)
        labels = ["cat", "dog", "bird"]
        y = np.array([labels[i % 3] for i in range(60)])

        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert all(p in labels for p in y_pred)
        assert clf.classes_.dtype.kind in ("U", "O")

    def test_integer_string_labels(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(40, 4)
        y = np.array(["1", "2", "3", "1", "2", "3"] * 6 + ["1", "2", "3", "1"])

        clf = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert set(y_pred).issubset({"1", "2", "3"})


class TestDataFrameInput:
    """Test using pandas tables as input."""

    def test_pandas_dataframe_input(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        rng = np.random.RandomState(42)
        X_arr = rng.randn(50, 5)
        y_arr = rng.randint(0, 3, size=50)

        X_df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
        y_series = pd.Series(y_arr, name="target")

        clf = TowerGBClassifier(n_passes=2, max_iter=30, random_state=42)
        clf.fit(X_df, y_series)

        y_pred = clf.predict(X_df)
        assert y_pred.shape == (50,)
        proba = clf.predict_proba(X_df)
        assert proba.shape == (50, 3)


class TestSerialization:
    """Test saving and loading the model with pickle and joblib."""

    def test_pickle_roundtrip(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, _ = binary_data
        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        original_pred = clf.predict(X_test)
        original_proba = clf.predict_proba(X_test)

        serialized = pickle.dumps(clf)
        clf_loaded = pickle.loads(serialized)

        np.testing.assert_array_equal(clf_loaded.predict(X_test), original_pred)
        np.testing.assert_allclose(clf_loaded.predict_proba(X_test), original_proba, atol=1e-14)

    def test_joblib_roundtrip(self, binary_data: tuple) -> None:
        joblib = pytest.importorskip("joblib")
        X_train, X_test, y_train, _ = binary_data

        clf = TowerGBClassifier(n_passes=3, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        original_proba = clf.predict_proba(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            joblib.dump(clf, f.name)
            clf_loaded = joblib.load(f.name)

        np.testing.assert_allclose(clf_loaded.predict_proba(X_test), original_proba, atol=1e-14)


class TestCalibration:
    """Test confidence tuning."""

    def test_calibrate_updates_temperature(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        clf = TowerGBClassifier(n_passes=3, max_iter=100, random_state=42)
        clf.fit(X_train, y_train)

        assert clf.temperature_ == clf.temperature
        clf.calibrate(X_test, y_test)
        assert isinstance(clf.temperature_, float)
        assert clf.temperature_ > 0.0

    def test_calibrate_returns_self(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        clf = TowerGBClassifier(n_passes=2, max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        assert clf.calibrate(X_test, y_test) is clf

    def test_calibration_reduces_or_maintains_ece(self, binary_data: tuple) -> None:
        X_train, X_test, y_train, y_test = binary_data
        clf = TowerGBClassifier(n_passes=5, max_iter=150, random_state=42)
        clf.fit(X_train, y_train)

        proba_before = clf.predict_proba(X_test)
        ece_before = compute_ece(y_test, proba_before)

        clf.calibrate(X_test, y_test)
        proba_after = clf.predict_proba(X_test)
        ece_after = compute_ece(y_test, proba_after)

        assert ece_after <= ece_before + 0.05


class TestReproducibility:
    """Test that the same random seed gives the exact same result."""

    def test_same_seed_same_predictions(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        y = rng.randint(0, 3, size=100)

        clf1 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=99)
        clf1.fit(X, y)

        clf2 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=99)
        clf2.fit(X, y)

        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))
        np.testing.assert_allclose(clf1.predict_proba(X), clf2.predict_proba(X), atol=1e-14)

    def test_different_seed_different_predictions(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8)
        y = rng.randint(0, 3, size=100)

        clf1 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=1)
        clf1.fit(X, y)

        clf2 = TowerGBClassifier(n_passes=3, max_iter=50, random_state=2)
        clf2.fit(X, y)

        assert not np.allclose(clf1._W_ensemble, clf2._W_ensemble, atol=1e-10)


class TestErrorHandling:
    """Test that the model shows clear errors when misused."""

    def test_predict_before_fit(self) -> None:
        clf = TowerGBClassifier()
        with pytest.raises(NotFittedError):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_regression_target_raises(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        y = np.array([0.1, 0.5, 0.9])
        clf = TowerGBClassifier()
        with pytest.raises(ValueError, match="Unknown label type"):
            clf.fit(X, y)


class TestParameterValidation:
    """Test that invalid hyperparameters raise clear errors."""

    def test_invalid_n_passes(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(n_passes=0)
        with pytest.raises(ValueError, match="n_passes"):
            clf.fit(X, y)

    def test_invalid_max_iter(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(max_iter=0)
        with pytest.raises(ValueError, match="max_iter"):
            clf.fit(X, y)

    def test_invalid_learning_rate(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(learning_rate=-0.1)
        with pytest.raises(ValueError, match="learning_rate"):
            clf.fit(X, y)

    def test_invalid_subsample_ratio(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(subsample_ratio=1.5)
        with pytest.raises(ValueError, match="subsample_ratio"):
            clf.fit(X, y)

    def test_invalid_l2_reg(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(l2_reg=-0.01)
        with pytest.raises(ValueError, match="l2_reg"):
            clf.fit(X, y)

    def test_invalid_temperature(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            clf.fit(X, y)

    def test_invalid_early_stop_fraction(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(early_stop_fraction=1.0)
        with pytest.raises(ValueError, match="early_stop_fraction"):
            clf.fit(X, y)

    def test_zero_sample_weight_raises(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier()
        with pytest.raises(ValueError, match="zero"):
            clf.fit(X, y, sample_weight=np.zeros(2))

    def test_wrong_sample_weight_shape_raises(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier()
        with pytest.raises(ValueError, match="sample_weight"):
            clf.fit(X, y, sample_weight=np.ones(5))


class TestAdversarialAndSecurityInputs:
    """Test robustness against extreme, adversarial, and edge-case inputs."""

    def test_all_nan_column(self) -> None:
        """Model should handle a column that is entirely NaN."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        X[:, 2] = np.nan  # Entire column missing
        y = (X[:, 0] > 0).astype(int)
        clf = TowerGBClassifier(handle_missing=True, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(np.isfinite(proba))

    def test_extreme_feature_values(self) -> None:
        """Model should not produce NaN/Inf on very large feature values."""
        X = np.array([[1e10, -1e10], [-1e10, 1e10]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(max_grad_norm=1.0, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(np.isfinite(proba))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)

    def test_single_sample_per_class(self) -> None:
        """Minimum viable training: exactly one sample per class."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        y = np.array([0, 1])
        clf = TowerGBClassifier(n_passes=2, max_iter=20, random_state=42)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert len(pred) == 2

    def test_feature_mismatch_at_predict(self) -> None:
        """Predict with wrong number of features should raise ValueError."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 5)
        y = rng.randint(0, 2, size=50)
        clf = TowerGBClassifier(n_passes=2, max_iter=20, random_state=42)
        clf.fit(X_train, y)
        with pytest.raises(ValueError):
            clf.predict(rng.randn(10, 3))

    def test_constant_features(self) -> None:
        """All-constant features (zero std) should not cause division by zero."""
        X = np.ones((50, 3), dtype=np.float64)
        y = np.array([0] * 25 + [1] * 25)
        clf = TowerGBClassifier(normalize=True, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(np.isfinite(proba))

    def test_large_n_classes(self) -> None:
        """Ensure stability with many classes."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.randint(0, 20, size=200)
        clf = TowerGBClassifier(n_passes=2, max_iter=50, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (200, 20)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
