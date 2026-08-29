"""TowerGB estimator — scikit-learn compliant dual-tower classifier.

Implements the full ``TowerGBClassifier`` with:

    • ``.fit(X, y)``           — Dual-tower M-pass bootstrap training
    • ``.predict(X)``          — Hard class predictions
    • ``.predict_proba(X)``    — Calibrated soft probabilities
    • ``.calibrate(X_val, y_val)`` — Post-hoc temperature scaling
    • ``.score(X, y)``         — Classification accuracy

The estimator is fully compliant with ``sklearn.utils.estimator_checks``
and can be used seamlessly in scikit-learn pipelines, grid searches,
and cross-validation workflows.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted

# sklearn >= 1.6 exposes validate_data as a standalone function;
# older versions expose it as a method on BaseEstimator.
try:
    from sklearn.utils.validation import validate_data as _validate_data  # >= 1.6
except ImportError:  # pragma: no cover – older sklearn
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
    r"""Dual-tower calibrated ensemble meta-learner for tabular classification.

    TowerGB trains :math:`M` independent linear models on bootstrap
    sub-samples, evaluating each through two complementary towers:

    **Tower 1 (Accuracy & Ranking):**
        Measures empirical accuracy on the full training set.

    **Tower 2 (Loss Calibration & Risk):**
        Computes log-loss, Brier score, and sample-wise loss variance
        :math:`\operatorname{Var}(\mathcal{L}^{(i)})` to penalize unstable
        or overconfident passes.

    **Cross-Tower Pareto Arbiter:**
        Fuses both signals into a composite fitness score and derives
        softmax-normalized ensemble weights for soft probability
        combination.

    **Post-Hoc Temperature Calibration:**
        Optimizes a scalar temperature :math:`T > 0` via golden-section
        search to minimize Expected Calibration Error (ECE).

    Parameters
    ----------
    n_passes : int, default 5
        Number of bootstrap training passes :math:`M`.
    learning_rate : float, default 0.1
        Step size :math:`\eta` for gradient descent weight updates.
    max_iter : int, default 300
        Maximum gradient descent iterations per pass.
    temperature : float, default 1.0
        Initial softmax temperature :math:`T > 0` for logit scaling.
    subsample_ratio : float, default 0.8
        Fraction of training samples drawn (with replacement) per
        bootstrap pass.
    tol : float, default 1e-6
        Convergence tolerance on loss change between iterations.
    random_state : int or None, default None
        Random seed for reproducibility of bootstrap sampling and
        Xavier weight initialization.
    arbiter_weights : dict or None, default None
        Custom Pareto arbiter coefficients.
        Keys: ``'accuracy'``, ``'log_loss'``, ``'brier'``, ``'loss_var'``.
        Defaults to ``{accuracy: 1.0, log_loss: 0.5, brier: 0.5, loss_var: 0.3}``.

    Attributes
    ----------
    classes_ : np.ndarray, shape ``(K,)``
        Unique class labels discovered during ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    temperature_ : float
        Active temperature (updated by ``calibrate``).

    Examples
    --------
    >>> from towergb import TowerGBClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, n_features=10,
    ...                            n_classes=3, n_informative=8,
    ...                            random_state=42)
    >>> clf = TowerGBClassifier(n_passes=3, random_state=42)
    >>> clf.fit(X, y)
    TowerGBClassifier(n_passes=3, random_state=42)
    >>> clf.predict(X[:3])
    array([...])
    """

    # ── Constructor (only sets params — no computation) ────────────
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
        self.n_passes = n_passes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.temperature = temperature
        self.subsample_ratio = subsample_ratio
        self.tol = tol
        self.random_state = random_state
        self.arbiter_weights = arbiter_weights

    # ── fit ─────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TowerGBClassifier:
        """Train the dual-tower ensemble.

        Parameters
        ----------
        X : array-like, shape ``(N, D)``
            Training feature matrix.
        y : array-like, shape ``(N,)``
            Target labels (integer, string, or any discrete type).

        Returns
        -------
        self
            Fitted estimator.
        """
        # ── Input validation ───────────────────────────────────────
        # Use validate_data (sklearn >= 1.6) or check_X_y (older)
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

        # Reject regression (continuous) targets
        target_type: str = type_of_target(y)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"Unknown label type: '{target_type}'. "
                "TowerGBClassifier requires discrete target labels."
            )

        # Enforce C-contiguous layout for BLAS / cache optimization
        X = np.ascontiguousarray(X, dtype=np.float64)

        # ── Label encoding ─────────────────────────────────────────
        self.classes_: np.ndarray = unique_labels(y)
        K: int = len(self.classes_)

        # Map labels → integer indices (supports strings, ints, etc.)
        self._label_to_idx: dict[object, int] = {
            label: idx for idx, label in enumerate(self.classes_)
        }
        y_idx: np.ndarray = np.array(
            [self._label_to_idx[label] for label in y], dtype=np.intp
        )

        N: int = X.shape[0]
        D: int = X.shape[1]
        self.n_features_in_: int = D

        # ── One-hot encoding — O(N·K) space ───────────────────────
        Y_one_hot: np.ndarray = np.zeros((N, K), dtype=np.float64)
        Y_one_hot[np.arange(N), y_idx] = 1.0

        # ── RNG for reproducibility ───────────────────────────────
        rng: np.random.RandomState = np.random.RandomState(self.random_state)

        # Resolve arbiter weights (never mutate constructor param)
        arb_weights: dict[str, float] = (
            dict(self.arbiter_weights) if self.arbiter_weights is not None
            else {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}
        )

        # ── Multi-pass training ───────────────────────────────────
        pass_weights: list[np.ndarray] = []
        metrics_list: list[dict[str, float]] = []
        n_iter_per_pass: list[int] = []

        for _m in range(self.n_passes):
            # ── Bootstrap / sub-sample ─────────────────────────────
            n_sub: int = max(1, int(N * self.subsample_ratio))
            indices: np.ndarray = rng.choice(N, size=n_sub, replace=True)
            X_sub: np.ndarray = X[indices]
            Y_sub: np.ndarray = Y_one_hot[indices]

            # ── Xavier initialization: std = √(2 / (D + K)) ───────
            std: float = np.sqrt(2.0 / (D + K)) if (D + K) > 0 else 0.01
            W: np.ndarray = (rng.randn(D, K) * std).astype(np.float64)

            # ── Gradient descent training loop ─────────────────────
            prev_loss: float = np.inf
            n_iters: int = 0
            for _it in range(self.max_iter):
                loss, grad = cross_entropy_loss_and_grad(
                    X_sub, Y_sub, W, self.temperature
                )
                # In-place weight update: W ← W − η·∇L
                W -= self.learning_rate * grad
                n_iters = _it + 1

                # Convergence check (loss plateau)
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

            pass_weights.append(W.copy())
            n_iter_per_pass.append(n_iters)

            # ── Tower 1: Accuracy computation ──────────────────────
            logits_full: np.ndarray = X @ W  # (N, K)
            P_full: np.ndarray = stable_softmax(logits_full, self.temperature)
            preds: np.ndarray = np.argmax(P_full, axis=1)
            accuracy: float = float(np.mean(preds == y_idx))

            # ── Tower 2: Risk metrics ──────────────────────────────
            risk: dict[str, float] = compute_risk_metrics(P_full, Y_one_hot)
            risk["accuracy"] = accuracy
            metrics_list.append(risk)

        # ── Pareto arbiter → ensemble weights ──────────────────────
        self._pass_weights: list[np.ndarray] = pass_weights
        self._ensemble_weights: np.ndarray = pareto_arbiter(
            metrics_list, arb_weights
        )
        self._pass_metrics: list[dict[str, float]] = metrics_list

        # Iteration counts per pass (sklearn requires n_iter_ for max_iter estimators)
        self.n_iter_: np.ndarray = np.array(n_iter_per_pass, dtype=np.intp)

        # ── Pre-compute ensemble weight matrix for O(N·D·K) inference
        # W_ens = Σ_m α_m · W_m   →   logits = X @ W_ens
        W_stack: np.ndarray = np.stack(self._pass_weights, axis=0)  # (M, D, K)
        alpha: np.ndarray = self._ensemble_weights.reshape(-1, 1, 1)  # (M, 1, 1)
        self._W_ensemble: np.ndarray = np.sum(alpha * W_stack, axis=0)  # (D, K)

        # Initialize calibrated temperature to the constructor value
        self.temperature_: float = self.temperature

        return self

    # ── Raw logits (pre-softmax) ───────────────────────────────────
    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble logits: X @ W_ensemble.

        Uses the pre-computed weighted-average weight matrix for
        single-matmul O(N·D·K) inference.
        """
        check_is_fitted(self)
        # Validate and check n_features_in_ consistency
        if _validate_data is not None:
            X = _validate_data(
                self, X, accept_sparse=False, dtype=np.float64, reset=False,
            )
        else:
            X = check_array(X, accept_sparse=False, dtype=np.float64)
            # Manual feature count check for older sklearn
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features as input."
                )
        X = np.ascontiguousarray(X)
        logits: np.ndarray = np.asarray(X @ self._W_ensemble, dtype=np.float64)
        return logits  # (N, K)

    # ── predict_proba ──────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with temperature-calibrated softmax.

        Parameters
        ----------
        X : array-like, shape ``(N, D)``
            Feature matrix.

        Returns
        -------
        P : np.ndarray, shape ``(N, K)``
            Row-stochastic probability matrix.
        """
        logits: np.ndarray = self._raw_logits(X)
        return stable_softmax(logits, self.temperature_)

    # ── predict ────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via argmax of calibrated probabilities.

        Parameters
        ----------
        X : array-like, shape ``(N, D)``
            Feature matrix.

        Returns
        -------
        y_pred : np.ndarray, shape ``(N,)``
            Predicted class labels (same dtype as training ``y``).
        """
        proba: np.ndarray = self.predict_proba(X)
        indices: np.ndarray = np.argmax(proba, axis=1)
        return np.asarray(self.classes_[indices])

    # ── calibrate ──────────────────────────────────────────────────
    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TowerGBClassifier:
        """Post-hoc temperature calibration via ECE minimization.

        Runs golden-section search on a validation set to find the
        optimal temperature :math:`T^*` that minimizes Expected
        Calibration Error.

        Parameters
        ----------
        X_val : array-like, shape ``(N_val, D)``
            Validation feature matrix.
        y_val : array-like, shape ``(N_val,)``
            Validation labels.

        Returns
        -------
        self
            Estimator with updated ``temperature_``.
        """
        check_is_fitted(self)
        # Validate with feature count check
        if _validate_data is not None:
            X_val = _validate_data(
                self, X_val, accept_sparse=False, dtype=np.float64, reset=False,
            )
        else:
            from sklearn.utils.validation import check_array
            X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)

        # Encode validation labels to indices
        y_val_arr: np.ndarray = np.asarray(y_val)
        y_idx: np.ndarray = np.array(
            [self._label_to_idx[label] for label in y_val_arr], dtype=np.intp
        )

        # Compute raw ensemble logits (unscaled)
        logits: np.ndarray = self._raw_logits(X_val)

        # Golden-section optimization of temperature
        self.temperature_ = optimize_temperature(
            logits, y_idx, len(self.classes_)
        )

        return self

    # ── score ──────────────────────────────────────────────────────
    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Compute classification accuracy on test data.

        Parameters
        ----------
        X : array-like, shape ``(N, D)``
            Test feature matrix.
        y : array-like, shape ``(N,)``
            True labels.
        sample_weight : array-like or None, default None
            Optional per-sample weights.

        Returns
        -------
        accuracy : float
            Mean accuracy ∈ ``[0, 1]``.
        """
        predictions: np.ndarray = self.predict(X)
        correct: np.ndarray = predictions == np.asarray(y)
        if sample_weight is not None:
            return float(np.average(correct, weights=np.asarray(sample_weight)))
        return float(np.mean(correct))
