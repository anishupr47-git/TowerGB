"""TowerGB classifier model estimator.

This module provides the main `TowerGBClassifier` class fully compliant with scikit-learn.
It combines:
1. Multi-pass ensemble training with Adam optimization and He initialization.
2. Built-in feature standardization, NaN imputation, and polynomial interaction expansion.
3. Ensemble weight collapsing into a single matrix for sub-millisecond inference.
4. Feature importance attribution mapped back to original input columns.
5. Epistemic uncertainty estimation via weighted ensemble disagreement.

Every class method and helper function is documented with both a simple plain-English explanation
and the exact mathematical formulas.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:
    _validate_data = None


# Cache the finite keyword name once at import time (deterministic per process)
_FINITE_KW_NAME: str = (
    "ensure_all_finite"
    if "ensure_all_finite" in inspect.signature(check_array).parameters
    else "force_all_finite"
)


def _get_finite_kwarg(allow_nan: bool) -> dict[str, str | bool]:
    """Helper to maintain backward/forward compatibility across all scikit-learn versions.

    SIMPLE EXPLANATION:
    -------------------
    Scikit-Learn renamed its validation keyword from 'force_all_finite' to 'ensure_all_finite'.
    This function uses a module-level cached lookup to pass the correct keyword,
    preventing unexpected runtime crashes.

    MATH LOGIC:
    -----------
    If 'ensure_all_finite' in signature(check_array):
        return {"ensure_all_finite": "allow-nan" if allow_nan else True}
    Else:
        return {"force_all_finite": "allow-nan" if allow_nan else True}
    """
    val = "allow-nan" if allow_nan else True
    return {_FINITE_KW_NAME: val}


from towergb._engine import (
    _safe_weight_sum,
    adam_step,
    clip_gradient,
    compute_risk_metrics,
    cross_entropy_loss_and_grad,
    pareto_arbiter,
    stable_softmax,
)
from towergb._preprocessing import (
    apply_standardizer,
    compute_nan_stats,
    expand_polynomial,
    fit_standardizer,
    impute_nans,
)
from towergb.calibration import optimize_temperature, vector_scale_calibration

# Minimum training samples required before validation early stopping can be activated
_MIN_SAMPLES_EARLY_STOP = 30
# Number of consecutive checks without loss improvement before stopping
_EARLY_STOP_PATIENCE = 10
# Interval (in iterations) between validation loss checks
_VAL_CHECK_INTERVAL = 5


def _train_single_pass(
    pass_idx: int, X: np.ndarray, Y_one_hot: np.ndarray, y_idx: np.ndarray,
    sample_weight: np.ndarray | None, subsample_ratio: float, max_iter: int,
    learning_rate: float, temperature: float, l2_reg: float, tol: float,
    seed: int | None, max_grad_norm: float | None, early_stop_fraction: float,
) -> tuple[np.ndarray, int, dict[str, float]]:
    """Train a single linear model pass in the ensemble using Adam optimization.

    SIMPLE EXPLANATION:
    -------------------
    This function trains one helper model ('pass') out of the ensemble:
    1. Appends a constant 1.0 bias feature to the data so the model can learn offsets.
    2. Initializes weights using He Normal Random Initialization (scaled by feature count).
    3. If subsampling < 1.0, draws a random bootstrap subset of rows.
    4. If early stopping is enabled, splits off a validation portion to detect when overfitting begins.
    5. Runs the Adam optimization loop: computes loss/gradients, clips large gradients, updates weights.
    6. Returns the trained weight matrix W, total iterations run, and risk metrics on full data.

    MATH LOGIC:
    -----------
    - Extended Feature Matrix:  X_ext = [ X  |  1 ]  (shape N x (D+1))
    - He Normal Initialization: W_ij ~ N(0, σ^2)  where σ = sqrt(2 / ((D+1) + K))
    - Optimization Loop (per iteration t):
        1. Loss & Grad: (loss_t, grad_t) = cross_entropy_loss_and_grad(X_tr, Y_tr, W, T, λ, w_tr)
        2. Gradient Clipping: If ||grad_t||_2 > max_norm, grad_t = grad_t * (max_norm / ||grad_t||_2)
        3. Adam Update: (step_t, m_t, v_t) = adam_step(grad_t, m, v, t, lr)
                        W = W - step_t
    - Early Stopping Check: Every 5 iterations, evaluate val_loss(W) on validation set.
        If val_loss doesn't decrease by > tol for 10 checks, restore best_W and terminate loop.
    """
    N, D = X.shape
    K = Y_one_hot.shape[1]
    pass_seed = None if seed is None else (seed + pass_idx * 10007)
    rng = np.random.RandomState(pass_seed)

    X_ext = np.hstack([X, np.ones((N, 1), dtype=np.float64)])
    D_ext = D + 1

    # --- Subsampling / Bootstrap -------------------------------------------
    if subsample_ratio >= 1.0:
        X_sub, Y_sub, sw_sub = X_ext, Y_one_hot, sample_weight
    else:
        n_sub = max(1, int(N * subsample_ratio))
        indices = rng.choice(N, size=n_sub, replace=True)
        X_sub, Y_sub = X_ext[indices], Y_one_hot[indices]
        sw_sub = sample_weight[indices] if sample_weight is not None else None

    # --- Validation split for early stopping --------------------------------
    N_sub = X_sub.shape[0]
    use_early_stop = early_stop_fraction > 0.0 and N_sub >= _MIN_SAMPLES_EARLY_STOP
    if use_early_stop:
        n_val = max(1, int(N_sub * early_stop_fraction))
        perm = rng.permutation(N_sub)
        tr_idx, va_idx = perm[n_val:], perm[:n_val]
        X_tr, Y_tr = X_sub[tr_idx], Y_sub[tr_idx]
        X_va, Y_va = X_sub[va_idx], Y_sub[va_idx]
        sw_tr = sw_sub[tr_idx] if sw_sub is not None else None
        sw_va = sw_sub[va_idx] if sw_sub is not None else None
    else:
        X_tr, Y_tr, sw_tr = X_sub, Y_sub, sw_sub

    # --- He Initialization --------------------------------------------------
    std = np.sqrt(2.0 / (D_ext + K)) if (D_ext + K) > 0 else 0.01
    W = (rng.randn(D_ext, K) * std).astype(np.float64)

    # --- Adam Optimizer State -----------------------------------------------
    m_state = np.zeros_like(W)
    v_state = np.zeros_like(W)

    best_W = W.copy()
    best_val_loss = np.inf
    patience_ctr = 0
    prev_loss = np.inf
    n_iters = 0

    for _it in range(max_iter):
        loss, grad = cross_entropy_loss_and_grad(
            X_tr, Y_tr, W, temperature, l2_reg, sw_tr,
        )

        # Gradient clipping
        if max_grad_norm is not None:
            grad = clip_gradient(grad, max_grad_norm)

        # Adam parameter update
        t = _it + 1
        step, m_state, v_state = adam_step(grad, m_state, v_state, t, learning_rate)
        W -= step
        n_iters = t

        # --- Early Stopping Evaluation ---------------------------------------
        if use_early_stop:
            if t % _VAL_CHECK_INTERVAL == 0:
                val_loss, _ = cross_entropy_loss_and_grad(
                    X_va, Y_va, W, temperature, l2_reg, sw_va,
                )
                if val_loss < best_val_loss - tol:
                    best_val_loss = val_loss
                    best_W = W.copy()
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= _EARLY_STOP_PATIENCE:
                        W = best_W
                        break
        else:
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

    if use_early_stop:
        W = best_W

    # --- Risk Evaluation on Full Dataset ------------------------------------
    logits_full = X_ext @ W
    P_full = stable_softmax(logits_full, temperature)
    preds = np.argmax(P_full, axis=1)

    if sample_weight is not None:
        sw = sample_weight.ravel()
        w_sum = _safe_weight_sum(sw)
        accuracy = float(np.sum(sw * (preds == y_idx)) / w_sum)
    else:
        accuracy = float(np.mean(preds == y_idx))

    risk = compute_risk_metrics(P_full, Y_one_hot, sample_weight)
    risk["accuracy"] = accuracy
    return W, n_iters, risk


class TowerGBClassifier(ClassifierMixin, BaseEstimator):
    """Calibrated ensemble classifier for tabular classification tasks.

    SIMPLE EXPLANATION:
    -------------------
    TowerGB is an enterprise-ready model built for tabular data. It trains multiple
    helper passes, evaluates their accuracy and stability using a multi-objective
    Pareto referee (Arbiter), and combines all models into ONE SINGLE mathematical matrix.
    This gives sub-millisecond prediction speeds with high accuracy and honest probabilities.

    KEY MATHEMATICAL FEATURES:
    --------------------------
    - Adam Optimizer + Gradient Clipping (fast, stable optimization)
    - Built-in Z-Score Standardization (equal feature scaling)
    - Degree-2 Polynomial Expansion (captures non-linear interaction terms x_i * x_j)
    - Native NaN Imputation + Missing Indicators (handles missing data gracefully)
    - Pareto Arbiter Ensemble Collapse (sums passes into W_ensemble = Σ α_m W^(m))
    - Temperature & Vector (Platt) Calibration (honest confidence tuning)
    - Epistemic Uncertainty Estimation (quantifies model disagreement across passes)
    """

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
        # Enterprise Math Parameters
        interaction_degree: int = 1,
        normalize: bool = True,
        max_grad_norm: float | None = 5.0,
        early_stop_fraction: float = 0.0,
        handle_missing: bool = True,
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
        self.interaction_degree = interaction_degree
        self.normalize = normalize
        self.max_grad_norm = max_grad_norm
        self.early_stop_fraction = early_stop_fraction
        self.handle_missing = handle_missing

    def __sklearn_tags__(self) -> object:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = bool(self.handle_missing)
        return tags

    def _get_tags(self) -> dict[str, bool]:
        tags = super()._get_tags() if hasattr(super(), "_get_tags") else {}
        tags["allow_nan"] = bool(self.handle_missing)
        return tags

    # -------------------------------------------------------------------
    # Internal Preprocessing Pipeline
    # -------------------------------------------------------------------

    def _preprocess(
        self, X: np.ndarray, *, fit: bool = False, sample_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Execute feature engineering pipeline: Impute NaN -> Standardize -> Expand Polynomial -> Append NaN Flags.

        SIMPLE EXPLANATION:
        -------------------
        Transforms raw data tables into clean, scaled mathematical matrices:
        1. Fills missing cells (NaNs) with training-set column averages.
        2. Standardizes features to Z-scores (mean 0, std 1).
        3. Generates polynomial interaction terms (like x_1^2 or x_1 * x_2) if requested.
        4. Appends binary indicator flags showing which cells were originally missing.

        MATH LOGIC:
        -----------
        1. Impute NaNs:        X = impute_nans(X, μ_nan)
        2. Z-Score:            X = (X - μ_feat) / σ_feat
        3. Polynomial:         X_poly = [ X | X^2 | X_i * X_j ]  for i < j
        4. Missing Indicators: X_final = [ X_poly | 𝕀(X_orig == NaN) ]
        """
        X = np.array(X, dtype=np.float64)
        nan_mask = None

        # 1. Impute NaNs
        if self.handle_missing:
            if fit:
                self._nan_means_, self._nan_cols_ = compute_nan_stats(X, sample_weight=sample_weight)
            X, nan_mask = impute_nans(X, self._nan_means_)

        # 2. Z-Score Standardization
        if self.normalize:
            if fit:
                self._feat_mean_, self._feat_std_ = fit_standardizer(X, sample_weight=sample_weight)
            X = apply_standardizer(X, self._feat_mean_, self._feat_std_)

        # 3. Polynomial Feature Expansion
        if self.interaction_degree >= 2:
            if fit:
                X, self._feature_map_ = expand_polynomial(X, self.interaction_degree)
            else:
                X, _ = expand_polynomial(X, self.interaction_degree)
        elif fit:
            self._feature_map_ = [(i,) for i in range(X.shape[1])]

        # 4. Append NaN Indicator Columns
        if self.handle_missing and hasattr(self, "_nan_cols_") and self._nan_cols_.any():
            nan_col_indices = np.where(self._nan_cols_)[0]
            if nan_mask is not None:
                indicators = nan_mask[:, nan_col_indices].astype(np.float64)
            else:
                indicators = np.zeros(
                    (X.shape[0], len(nan_col_indices)), dtype=np.float64,
                )
            X = np.hstack([X, indicators])
            if fit:
                for j in nan_col_indices:
                    self._feature_map_.append((int(j),))

        return np.ascontiguousarray(X, dtype=np.float64)

    # -------------------------------------------------------------------
    # Training (fit)
    # -------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> TowerGBClassifier:
        """Train the ensemble classifier on feature matrix X and target labels y.

        SIMPLE EXPLANATION:
        -------------------
        1. Validates input data and converts class labels into discrete integers.
        2. Encodes y into a One-Hot binary matrix (e.g. Class 2 of 3 -> [0, 0, 1]).
        3. Preprocesses features (NaN imputation, standardization, polynomial expansion).
        4. Trains `n_passes` helper models in parallel or sequence using Adam.
        5. Calls the Pareto Arbiter to calculate voting weight α_m for each pass based on fitness.
        6. COLLAPSES all passes into ONE single combined matrix W_ensemble = Σ α_m W^(m).
        7. Aggregates feature importance scores back to the original input columns.

        MATH LOGIC:
        -----------
        1. One-Hot Matrix: Y_{i,k} = 1.0 if y_i == class_k else 0.0  (shape N x K)
        2. Pass Training: Get weight matrices {W^(1), W^(2), ..., W^(P)} each of shape (D_ext x K)
        3. Pareto Arbiter Weights: α = pareto_arbiter(metrics_list, weights)  where Σ α_m = 1.0
        4. Ensemble Collapsing:
               W_ensemble = Σ_{m=1}^P  α_m * W^(m)   (shape D_ext x K)
           Notice that for any future input X_ext:
               Σ α_m (X_ext @ W^(m)) = X_ext @ (Σ α_m W^(m)) = X_ext @ W_ensemble
           This single-step matrix multiplication is what enables sub-millisecond inference!
        5. Feature Importances:
               Raw importance over expanded features: I_k = mean_j |W_{k,j}|
               Map back to original feature j: Import_j = Σ_{k ∈ map(j)} (I_k / |map(k)|)
               Normalize: feature_importances_ = Import_j / Σ Import
        """
        finite_kw = _get_finite_kwarg(self.handle_missing)
        if _validate_data is not None:
            X, y = _validate_data(
                self, X, y,
                accept_sparse=False, dtype=np.float64,
                multi_output=False, ensure_2d=True, reset=True,
                **finite_kw,
            )
        else:
            X, y = check_X_y(
                X, y,
                accept_sparse=False, dtype=np.float64,
                multi_output=False, ensure_2d=True,
                **finite_kw,
            )

        target_type = type_of_target(y)
        if target_type not in ("binary", "multiclass"):
            raise ValueError(
                f"Unknown label type: '{target_type}'. "
                "TowerGBClassifier requires discrete target labels."
            )

        # --- Validate hyperparameters ----------------------------------------
        if self.n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {self.n_passes}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if not 0.0 < self.learning_rate <= 10.0:
            raise ValueError(
                f"learning_rate must be in (0, 10], got {self.learning_rate}"
            )
        if not 0.0 < self.subsample_ratio <= 1.0:
            raise ValueError(
                f"subsample_ratio must be in (0, 1], got {self.subsample_ratio}"
            )
        if self.l2_reg < 0.0:
            raise ValueError(f"l2_reg must be >= 0, got {self.l2_reg}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not 0.0 <= self.early_stop_fraction < 1.0:
            raise ValueError(
                f"early_stop_fraction must be in [0, 1), got {self.early_stop_fraction}"
            )

        self.classes_ = unique_labels(y)
        K = len(self.classes_)
        N = X.shape[0]
        self.n_features_in_ = X.shape[1]

        # Vectorized label encoding via binary search
        y_idx = np.searchsorted(self.classes_, y).astype(np.intp)
        Y_one_hot = np.eye(K, dtype=np.float64)[y_idx]

        # --- Sample Weight Computation ---------------------------------------
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

        # --- Internal Preprocessing -----------------------------------------
        X = self._preprocess(X, fit=True, sample_weight=sw)

        arb_weights = (
            dict(self.arbiter_weights)
            if self.arbiter_weights is not None
            else {"accuracy": 1.0, "log_loss": 0.5, "brier": 0.5, "loss_var": 0.3}
        )

        # --- Parallel / Sequential Pass Training -----------------------------
        pass_kwargs: dict[str, Any] = {
            "subsample_ratio": self.subsample_ratio,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "temperature": self.temperature,
            "l2_reg": self.l2_reg,
            "tol": self.tol,
            "seed": self.random_state,
            "max_grad_norm": self.max_grad_norm,
            "early_stop_fraction": self.early_stop_fraction,
        }

        if self.n_jobs is not None and self.n_jobs != 1:
            try:
                from joblib import Parallel, delayed
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(_train_single_pass)(
                        m, X, Y_one_hot, y_idx, sw, **pass_kwargs,
                    )
                    for m in range(self.n_passes)
                )
            except (ImportError, RuntimeError, OSError):
                results = [
                    _train_single_pass(m, X, Y_one_hot, y_idx, sw, **pass_kwargs)
                    for m in range(self.n_passes)
                ]
        else:
            results = [
                _train_single_pass(m, X, Y_one_hot, y_idx, sw, **pass_kwargs)
                for m in range(self.n_passes)
            ]

        self._pass_weights = [r[0] for r in results]
        self.n_iter_ = np.array([r[1] for r in results], dtype=np.intp)
        self._pass_metrics = [r[2] for r in results]
        self._ensemble_weights = pareto_arbiter(self._pass_metrics, arb_weights)

        # --- Collapse Ensemble Weights ---------------------------------------
        W_stack = np.stack(self._pass_weights, axis=0)
        alpha = self._ensemble_weights.reshape(-1, 1, 1)
        self._W_ensemble = np.sum(alpha * W_stack, axis=0)
        self.temperature_ = self.temperature
        self._calibration_method_ = "temperature"

        # Feature coefficients & bias intercepts
        W_weights, W_bias = self._W_ensemble[:-1], self._W_ensemble[-1]
        self.coef_ = W_weights.T[1:2] if K == 2 else W_weights.T
        self.intercept_ = W_bias[1:2] if K == 2 else W_bias

        # --- Aggregate Feature Importances -----------------------------------
        D_orig = self.n_features_in_
        raw_imp_expanded = np.mean(np.abs(W_weights), axis=1)
        raw_imp = np.zeros(D_orig, dtype=np.float64)
        for k, feat_indices in enumerate(self._feature_map_):
            if k >= len(raw_imp_expanded):
                break
            share = raw_imp_expanded[k] / len(feat_indices)
            for j in feat_indices:
                if j < D_orig:
                    raw_imp[j] += share
        imp_sum = raw_imp.sum()
        self.feature_importances_ = (
            raw_imp / imp_sum if imp_sum > 0.0 else np.full(D_orig, 1.0 / D_orig)
        )

        return self

    # -------------------------------------------------------------------
    # Inference (predict & predict_proba)
    # -------------------------------------------------------------------

    def _raw_logits(self, X: np.ndarray) -> np.ndarray:
        """Validate input, preprocess features, and compute raw uncalibrated logits.

        MATH LOGIC:
        -----------
        Z = [ X_processed  |  1 ] @ W_ensemble
        """
        check_is_fitted(self)
        finite_kw = _get_finite_kwarg(self.handle_missing)
        if _validate_data is not None:
            X = _validate_data(
                self, X,
                accept_sparse=False, dtype=np.float64, reset=False,
                **finite_kw,
            )
        else:
            X = check_array(
                X, accept_sparse=False, dtype=np.float64,
                **finite_kw,
            )
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features."
                )
        X = self._preprocess(X)
        X_ext = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        return np.asarray(X_ext @ self._W_ensemble, dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability estimates (0.0 to 1.0) for each class label.

        SIMPLE EXPLANATION:
        -------------------
        Computes raw scores for each class and applies softmax (with temperature or vector scaling)
        to return calibrated probability percentages for every row.

        MATH LOGIC:
        -----------
        If Temperature Calibration:  P = softmax(RawLogits / T)
        If Vector Calibration:       P = softmax(RawLogits * a + b)
        """
        logits = self._raw_logits(X)
        if self._calibration_method_ == "vector":
            logits = logits * self._cal_scale_ + self._cal_bias_
            return stable_softmax(logits, temperature=1.0)
        return stable_softmax(logits, self.temperature_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the single most likely class label for each sample row.

        SIMPLE EXPLANATION:
        -------------------
        Picks the category that gets the highest predicted score.
        For temperature calibration, softmax is monotonic, so argmax(Logits) == argmax(Softmax(Logits)).
        We take argmax directly on raw logits, skipping exponentiation for maximum speed!

        MATH LOGIC:
        -----------
        y_pred_i = classes[ argmax_k (Z_{ik}) ]
        """
        if (
            hasattr(self, "_calibration_method_")
            and self._calibration_method_ == "vector"
        ):
            indices = np.argmax(self.predict_proba(X), axis=1)
        else:
            indices = np.argmax(self._raw_logits(X), axis=1)
        return np.asarray(self.classes_[indices])

    # -------------------------------------------------------------------
    # Calibration Method
    # -------------------------------------------------------------------

    def calibrate(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "temperature",
    ) -> TowerGBClassifier:
        """Tune model probability confidence using holdout validation data.

        SIMPLE EXPLANATION:
        -------------------
        Post-hoc calibration tunes the confidence dial on new validation data:
        - `method='temperature'`: Finds 1 single global temperature dial T using Golden Section Search.
        - `method='vector'`: Learns per-class multipliers a_k and shifts b_k (Platt scaling).

        MATH LOGIC:
        -----------
        Temperature: Minimize ECE(y_val, softmax(Z_val / T)) via Golden Ratio Search.
        Vector: Minimize NLL(y_val, softmax(a * Z_val + b)) via Gradient Descent.
        """
        check_is_fitted(self)
        y_val_arr = np.asarray(y_val)
        y_idx = np.searchsorted(self.classes_, y_val_arr).astype(np.intp)
        logits = self._raw_logits(X_val)

        if method == "vector":
            self._cal_scale_, self._cal_bias_ = vector_scale_calibration(
                logits, y_idx, len(self.classes_),
            )
            self._calibration_method_ = "vector"
        else:
            self.temperature_ = optimize_temperature(
                logits, y_idx, len(self.classes_),
            )
            self._calibration_method_ = "temperature"

        return self

    # -------------------------------------------------------------------
    # Epistemic Uncertainty Estimation
    # -------------------------------------------------------------------

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Quantify model epistemic (knowledge) uncertainty via ensemble disagreement.

        SIMPLE EXPLANATION:
        -------------------
        Calibrated probability tells you 'How sure am I?'.
        Epistemic uncertainty tells you 'How much do my ensemble helpers disagree on this row?'.
        If all 5 passes give the exact same prediction, uncertainty is 0.0 (high knowledge).
        If the passes give wildly different predictions, uncertainty is high (unseen data region!).

        MATH LOGIC:
        -----------
        1. Pass Predictions: P^(m) = softmax(X_ext @ W^(m) / T)  for m in [1, P]
        2. Ensemble Mean:    P̄ = Σ α_m P^(m)
        3. Weighted Variance: Var_i = Σ_k Σ_m α_m * (P_{ik}^(m) - P̄_{ik})^2

        Returns per-sample scalar uncertainty of shape (n_samples,).
        """
        check_is_fitted(self)
        finite_kw = _get_finite_kwarg(self.handle_missing)
        if _validate_data is not None:
            X = _validate_data(
                self, X,
                accept_sparse=False, dtype=np.float64, reset=False,
                **finite_kw,
            )
        else:
            X = check_array(
                X, accept_sparse=False, dtype=np.float64,
                **finite_kw,
            )
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but "
                    f"{self.__class__.__name__} is expecting "
                    f"{self.n_features_in_} features."
                )

        X_proc = self._preprocess(X)
        X_ext = np.hstack(
            [X_proc, np.ones((X_proc.shape[0], 1), dtype=np.float64)],
        )

        pass_probas = np.stack(
            [
                stable_softmax(X_ext @ W_p, self.temperature_)
                for W_p in self._pass_weights
            ],
            axis=0,
        )

        mean_proba = np.tensordot(
            self._ensemble_weights, pass_probas, axes=([0], [0]),
        )

        diff = pass_probas - mean_proba[None, :, :]
        weighted_var = np.tensordot(
            self._ensemble_weights, diff ** 2, axes=([0], [0]),
        )

        return np.sum(weighted_var, axis=1)

    # -------------------------------------------------------------------
    # Accuracy Score
    # -------------------------------------------------------------------

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        """Calculate mean classification accuracy score on test data.

        SIMPLE EXPLANATION:
        -------------------
        Compares predicted labels against actual true labels and calculates the accuracy ratio
        between 0.0 (0% correct) and 1.0 (100% correct).

        MATH LOGIC:
        -----------
        Accuracy = (1 / N) * Σ 𝕀(y_pred_i == y_true_i)
        If sample_weight: Accuracy = Σ (w_i * 𝕀(y_pred_i == y_true_i)) / Σ w_i
        """
        predictions = self.predict(X)
        correct = predictions == np.asarray(y)
        if sample_weight is not None:
            return float(np.average(correct, weights=np.asarray(sample_weight)))
        return float(np.mean(correct))
