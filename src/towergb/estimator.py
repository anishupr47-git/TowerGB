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


def _train_single_pass(
    pass_idx: int,
    X: np.ndarray,
    Y_one_hot: np.ndarray,
    y_idx: np.ndarray,
    sample_weight: np.ndarray | None,
    subsample_ratio: float,
    max_iter: int,
    learning_rate: float,
    temperature: float,
    l2_reg: float,
    tol: float,
    seed: int | None,
) -> tuple[np.ndarray, int, dict[str, float]]:
    N, D = X.shape
    K = Y_one_hot.shape[1]
    pass_seed = None if seed is None else (seed + pass_idx * 10007)
    rng = np.random.RandomState(pass_seed)

    X_ext = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    D_ext = D + 1

    if subsample_ratio >= 1.0:
        X_sub = X_ext
        Y_sub = Y_one_hot
        sw_sub = sample_weight
    else:
        n_sub = max(1, int(N * subsample_ratio))
        indices = rng.choice(N, size=n_sub, replace=True)
        X_sub = X_ext[indices]
        Y_sub = Y_one_hot[indices]
        sw_sub = sample_weight[indices] if sample_weight is not None else None

    std = np.sqrt(2.0 / (D_ext + K)) if (D_ext + K) > 0 else 0.01
    W = (rng.randn(D_ext, K) * std).astype(np.float64)

    prev_loss = np.inf
    n_iters = 0
    for _it in range(max_iter):
        loss, grad = cross_entropy_loss_and_grad(X_sub, Y_sub, W, temperature, l2_reg, sw_sub)
        W -= learning_rate * grad
        n_iters = _it + 1

        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    logits_full = X_ext @ W
    P_full = stable_softmax(logits_full, temperature)
    preds = np.argmax(P_full, axis=1)

    if sample_weight is not None:
        sw = sample_weight.ravel()
        w_sum = float(np.sum(sw)) if np.sum(sw) > 0 else float(len(sw))
        accuracy = float(np.sum(sw * (preds == y_idx)) / w_sum)
    else:
        accuracy = float(np.mean(preds == y_idx))

    risk = compute_risk_metrics(P_full, Y_one_hot, sample_weight)
    risk["accuracy"] = accuracy
    return W, n_iters, risk


class TowerGBClassifier(ClassifierMixin, BaseEstimator):
    """Calibrated ensemble classifier for tabular data with feature importances and class weighting."""

    def __init__(
        self,
        n_passes: int = 5,
        learning_rate: float = 0.1,
        max_iter: int = 300,
        temperature: float = 1.0,
        l2_reg: float = 1e-4,
        subsample_ratio: float = 1.0,
        class_weight: str | dict[object, float] | None = None,
        tol: float = 1e-6,
        random_state: int | None = None,
        arbiter_weights: dict[str, float] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_passes = n_passes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.temperature = temperature
        self.l2_reg = l2_reg
        self.subsample_ratio = subsample_ratio
        self.class_weight = class_weight
        self.tol = tol
        self.random_state = random_state
        self.arbiter_weights = arbiter_weights
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> TowerGBClassifier:
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

        sw = None
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim != 1 or len(sample_weight) != N:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if np.all(sample_weight == 0) or np.sum(sample_weight) == 0:
                raise ValueError("Sample weights sum to zero.")
            sw = sample_weight

        if self.class_weight is not None:
            from sklearn.utils.class_weight import compute_sample_weight
            cw = compute_sample_weight(self.class_weight, y)
            sw = cw if sw is None else sw * cw
            if np.all(sw == 0) or np.sum(sw) == 0:
                raise ValueError("Sample weights sum to zero.")

        arb_weights = (
            dict(self.arbiter_weights) if self.arbiter_weights is not None
            else {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}
        )

        if self.n_jobs is not None and self.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_train_single_pass)(
                        m, X, Y_one_hot, y_idx, sw,
                        self.subsample_ratio, self.max_iter, self.learning_rate,
                        self.temperature, self.l2_reg, self.tol, self.random_state,
                    )
                    for m in range(self.n_passes)
                )
            except (ImportError, RuntimeError, OSError):
                results = [
                    _train_single_pass(
                        m, X, Y_one_hot, y_idx, sw,
                        self.subsample_ratio, self.max_iter, self.learning_rate,
                        self.temperature, self.l2_reg, self.tol, self.random_state,
                    )
                    for m in range(self.n_passes)
                ]
        else:
            results = [
                _train_single_pass(
                    m, X, Y_one_hot, y_idx, sw,
                    self.subsample_ratio, self.max_iter, self.learning_rate,
                    self.temperature, self.l2_reg, self.tol, self.random_state,
                )
                for m in range(self.n_passes)
            ]

        self._pass_weights = [r[0] for r in results]
        self.n_iter_ = np.array([r[1] for r in results], dtype=np.intp)
        self._pass_metrics = [r[2] for r in results]

        self._ensemble_weights = pareto_arbiter(self._pass_metrics, arb_weights)

        W_stack = np.stack(self._pass_weights, axis=0)
        alpha = self._ensemble_weights.reshape(-1, 1, 1)
        self._W_ensemble = np.sum(alpha * W_stack, axis=0)
        self.temperature_ = self.temperature

        W_weights = self._W_ensemble[:-1]
        W_bias = self._W_ensemble[-1]

        if K == 2:
            self.coef_ = W_weights.T[1:2]
            self.intercept_ = W_bias[1:2]
        else:
            self.coef_ = W_weights.T
            self.intercept_ = W_bias

        raw_imp = np.mean(np.abs(W_weights), axis=1)
        imp_sum = float(np.sum(raw_imp))
        self.feature_importances_ = raw_imp / imp_sum if imp_sum > 0 else np.full(D, 1.0 / D)

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
        X_ext = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        return np.asarray(X_ext @ self._W_ensemble, dtype=np.float64)

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
