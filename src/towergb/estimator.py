"""TowerGB classifier model.

This model learns from table data and predicts which group a new row belongs to.
It works just like any standard scikit-learn classifier.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted

# Check if newer validate_data function exists in scikit-learn
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
    """Classifier that trains multiple rounds and combines them.

    Tower 1 checks accuracy (how many guesses are right).
    Tower 2 checks risk and confidence stability.
    The Arbiter gives higher voting weight to the best rounds.
    Calibration tunes the confidence score so it matches real accuracy.
    """

    def __init__(
        self,
        n_passes: int = 5,
        learning_rate: float = 0.1,
        max_iter: int = 300,
        temperature: float = 1.0,
        subsample_ratio: float = 0.8,
        tol: float = 1e-6,
        random_state: int | None = None,
        arbiter_weights: dict[str, float] | None = None,
    ) -> None:
        # Save setup settings
        self.n_passes = n_passes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.temperature = temperature
        self.subsample_ratio = subsample_ratio
        self.tol = tol
        self.random_state = random_state
        self.arbiter_weights = arbiter_weights

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TowerGBClassifier:
        """Learn patterns from training data and labels."""
        # Check input data format
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

        # Make sure labels are categories and not continuous numbers
        target_type: str = type_of_target(y)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"Unknown label type: '{target_type}'. "
                "TowerGBClassifier requires discrete target labels."
            )

        # Put numbers cleanly into memory
        X = np.ascontiguousarray(X, dtype=np.float64)

        # Find all unique categories
        self.classes_: np.ndarray = unique_labels(y)
        K: int = len(self.classes_)

        # Map each category name to a simple number (0, 1, 2, ...)
        self._label_to_idx: dict[object, int] = {
            label: idx for idx, label in enumerate(self.classes_)
        }
        y_idx: np.ndarray = np.array(
            [self._label_to_idx[label] for label in y], dtype=np.intp
        )

        N: int = X.shape[0]
        D: int = X.shape[1]
        self.n_features_in_: int = D

        # Create one-hot answers table
        Y_one_hot: np.ndarray = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        # Random number generator for repeatability
        rng: np.random.RandomState = np.random.RandomState(self.random_state)

        # Use default arbiter scoring weights if none provided
        arb_weights: dict[str, float] = (
            dict(self.arbiter_weights) if self.arbiter_weights is not None
            else {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}
        )

        # Train multiple rounds
        pass_weights: list[np.ndarray] = []
        metrics_list: list[dict[str, float]] = []
        n_iter_per_pass: list[int] = []

        for _m in range(self.n_passes):
            # Take a random sample of rows for this round
            n_sub: int = max(1, int(N * self.subsample_ratio))
            indices: np.ndarray = rng.choice(N, size=n_sub, replace=True)
            X_sub: np.ndarray = X[indices]
            Y_sub: np.ndarray = Y_one_hot[indices]

            # Start with small random weights
            std: float = np.sqrt(2.0 / (D + K)) if (D + K) > 0 else 0.01
            W: np.ndarray = (rng.randn(D, K) * std).astype(np.float64)

            # Train with gradient descent step by step
            prev_loss: float = np.inf
            n_iters: int = 0
            for _it in range(self.max_iter):
                loss, grad = cross_entropy_loss_and_grad(
                    X_sub, Y_sub, W, self.temperature
                )
                # Adjust weights in the direction that lowers mistakes
                W -= self.learning_rate * grad
                n_iters = _it + 1

                # Stop early if mistake score stopped changing
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

            pass_weights.append(W.copy())
            n_iter_per_pass.append(n_iters)

            # Tower 1: measure accuracy on full data
            logits_full: np.ndarray = X @ W
            P_full: np.ndarray = stable_softmax(logits_full, self.temperature)
            preds: np.ndarray = np.argmax(P_full, axis=1)
            accuracy: float = float(np.mean(preds == y_idx))

            # Tower 2: measure mistake size and risk
            risk: dict[str, float] = compute_risk_metrics(P_full, Y_one_hot)
            risk["accuracy"] = accuracy
            metrics_list.append(risk)

        # Combine all rounds using Arbiter weights
        self._pass_weights: list[np.ndarray] = pass_weights
        self._ensemble_weights: np.ndarray = pareto_arbiter(
            metrics_list, arb_weights
        )
        self._pass_metrics: list[dict[str, float]] = metrics_list

        # Save iteration counts per pass
        self.n_iter_: np.ndarray = np.array(n_iter_per_pass, dtype=np.intp)

        # Pre-calculate combined weight matrix for fast predictions
        W_stack: np.ndarray = np.stack(self._pass_weights, axis=0)
        alpha: np.ndarray = self._ensemble_weights.reshape(-1, 1, 1)
        self._W_ensemble: np.ndarray = np.sum(alpha * W_stack, axis=0)

        # Set starting temperature
        self.temperature_: float = self.temperature

        return self

    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        # Calculate raw scores before converting to probabilities
        check_is_fitted(self)
        if _validate_data is not None:
            X = _validate_data(
                self, X, accept_sparse=False, dtype=np.float64, reset=False,
            )
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features as input."
                )
        X = np.ascontiguousarray(X)
        logits: np.ndarray = np.asarray(X @ self._W_ensemble, dtype=np.float64)
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the probability (0% to 100%) for each category."""
        logits: np.ndarray = self._raw_logits(X)
        return stable_softmax(logits, self.temperature_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely category for each row."""
        proba: np.ndarray = self.predict_proba(X)
        indices: np.ndarray = np.argmax(proba, axis=1)
        return np.asarray(self.classes_[indices])

    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TowerGBClassifier:
        """Tune confidence temperature on test data so probabilities are accurate."""
        check_is_fitted(self)
        if _validate_data is not None:
            X_val = _validate_data(
                self, X_val, accept_sparse=False, dtype=np.float64, reset=False,
            )
        else:
            from sklearn.utils.validation import check_array
            X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)

        # Map validation labels to numbers
        y_val_arr: np.ndarray = np.asarray(y_val)
        y_idx: np.ndarray = np.array(
            [self._label_to_idx[label] for label in y_val_arr], dtype=np.intp
        )

        # Get raw scores
        logits: np.ndarray = self._raw_logits(X_val)

        # Find best temperature
        self.temperature_ = optimize_temperature(
            logits, y_idx, len(self.classes_)
        )

        return self

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Calculate accuracy (percentage of correct guesses)."""
        predictions: np.ndarray = self.predict(X)
        correct: np.ndarray = predictions == np.asarray(y)
        if sample_weight is not None:
            return float(np.average(correct, weights=np.asarray(sample_weight)))
        return float(np.mean(correct))
