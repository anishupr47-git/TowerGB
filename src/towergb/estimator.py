"""TowerGB classifier model."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted

try:
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:
    from sklearn.utils.validation import check_array, check_X_y
    _validate_data = None

from towergb._engine import (
    compute_risk_metrics,
    cross_entropy_loss_and_grad,
    pareto_arbiter,
    stable_softmax,
)
from towergb.calibration import optimize_temperature


class TowerGBClassifier(ClassifierMixin, BaseEstimator):
    """Calibrated ensemble classifier for tabular data."""

    def __init__(self, n_passes: int = 5, learning_rate: float = 0.1, max_iter: int = 300, temperature: float = 1.0, subsample_ratio: float = 0.8, tol: float = 1e-6, random_state: int | None = None, arbiter_weights: dict[str, float] | None = None) -> None:
        self.n_passes = n_passes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.temperature = temperature
        self.subsample_ratio = subsample_ratio
        self.tol = tol
        self.random_state = random_state
        self.arbiter_weights = arbiter_weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> TowerGBClassifier:
        """Learn patterns from training data and labels."""
        if _validate_data is not None:
            X, y = _validate_data(
                self, X, y, accept_sparse=False, dtype=np.float64,
                multi_output=False, ensure_2d=True, reset=True,
            )
        else:
            X, y = check_X_y(
                X, y, accept_sparse=False, dtype=np.float64,
                multi_output=False, ensure_2d=True,
            )

        target_type = type_of_target(y)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"Unknown label type: '{target_type}'. "
                "TowerGBClassifier requires discrete target labels."
            )

        X = np.ascontiguousarray(X, dtype=np.float64)
        self.classes_ = unique_labels(y)
        K = len(self.classes_)

        self._label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_idx = np.array([self._label_to_idx[label] for label in y], dtype=np.intp)

        N, D = X.shape
        self.n_features_in_ = D

        Y_one_hot = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        rng = np.random.RandomState(self.random_state)
        arb_weights = (
            dict(self.arbiter_weights) if self.arbiter_weights is not None
            else {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}
        )

        pass_weights = []
        metrics_list = []
        n_iter_per_pass = []

        for _m in range(self.n_passes):
            n_sub = max(1, int(N * self.subsample_ratio))
            indices = rng.choice(N, size=n_sub, replace=True)
            X_sub = X[indices]
            Y_sub = Y_one_hot[indices]

            std = np.sqrt(2.0 / (D + K)) if (D + K) > 0 else 0.01
            W = (rng.randn(D, K) * std).astype(np.float64)

            prev_loss = np.inf
            n_iters = 0
            for _it in range(self.max_iter):
                loss, grad = cross_entropy_loss_and_grad(X_sub, Y_sub, W, self.temperature)
                W -= self.learning_rate * grad
                n_iters = _it + 1

                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

            pass_weights.append(W.copy())
            n_iter_per_pass.append(n_iters)

            logits_full = X @ W
            P_full = stable_softmax(logits_full, self.temperature)
            preds = np.argmax(P_full, axis=1)
            accuracy = float(np.mean(preds == y_idx))

            risk = compute_risk_metrics(P_full, Y_one_hot)
            risk["accuracy"] = accuracy
            metrics_list.append(risk)

        self._pass_weights = pass_weights
        self._ensemble_weights = pareto_arbiter(metrics_list, arb_weights)
        self._pass_metrics = metrics_list
        self.n_iter_ = np.array(n_iter_per_pass, dtype=np.intp)

        W_stack = np.stack(self._pass_weights, axis=0)
        alpha = self._ensemble_weights.reshape(-1, 1, 1)
        self._W_ensemble = np.sum(alpha * W_stack, axis=0)
        self.temperature_ = self.temperature

        return self

    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if _validate_data is not None:
            X = _validate_data(self, X, accept_sparse=False, dtype=np.float64, reset=False)
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features as input."
                )
        X = np.ascontiguousarray(X)
        return np.asarray(X @ self._W_ensemble, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability (0% to 100%) for each category."""
        return stable_softmax(self._raw_logits(X), self.temperature_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely category for each row."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return np.asarray(self.classes_[indices])

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> TowerGBClassifier:
        """Tune confidence temperature on validation data so probabilities match real accuracy."""
        check_is_fitted(self)
        if _validate_data is not None:
            X_val = _validate_data(self, X_val, accept_sparse=False, dtype=np.float64, reset=False)
        else:
            from sklearn.utils.validation import check_array
            X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)

        y_val_arr = np.asarray(y_val)
        y_idx = np.array([self._label_to_idx[label] for label in y_val_arr], dtype=np.intp)
        logits = self._raw_logits(X_val)

        self.temperature_ = optimize_temperature(logits, y_idx, len(self.classes_))
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        """Calculate classification accuracy score."""
        predictions = self.predict(X)
        correct = predictions == np.asarray(y)
        if sample_weight is not None:
            return float(np.average(correct, weights=np.asarray(sample_weight)))
        return float(np.mean(correct))
