"""Probability calibration tools for TowerGB.

This module provides two probability calibration algorithms:
1. Temperature Scaling: Tunes a single global temperature (T) using Golden Section Search
   to minimize Expected Calibration Error (ECE).
2. Vector Scaling (Platt Scaling): Learns per-class scale (a_k) and shift (b_k) parameters
   via gradient descent on Negative Log-Likelihood (NLL).

Every function is documented with both a simple plain-English explanation
and the exact mathematical formulas.
"""

from __future__ import annotations

import numpy as np

from towergb._engine import stable_softmax


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute the Expected Calibration Error (ECE) metric.

    SIMPLE EXPLANATION:
    -------------------
    If a weather forecaster says 'There is an 80% chance of rain' on 100 different days, it SHOULD
    rain on exactly 80 of those days. If it only rains 50 times, the forecaster is overconfident!
    ECE measures how honest/calibrated an AI's confidence percentages are.
    It splits predictions into confidence buckets (e.g. 70%-80% confidence bin), checks actual
    accuracy in each bucket, and calculates the weighted average difference (gap).
    A score of 0.0 means perfect honesty!

    MATH LOGIC:
    -----------
    1. For each sample i, find predicted class ŷ_i = argmax(P_i) and confidence c_i = max(P_i).
    2. Divide interval [0, 1] into M equal bins B_1, B_2, ..., B_M (width = 1/M).
    3. For each bin B_m with count |B_m| > 0:
          Bin Accuracy:   acc(B_m)  = (1 / |B_m|) * Σ_{i ∈ B_m} 𝕀(ŷ_i == y_i)
          Bin Confidence: conf(B_m) = (1 / |B_m|) * Σ_{i ∈ B_m} c_i
    4. Expected Calibration Error:
          ECE = Σ_{m=1}^M  (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    N = len(y_true)
    if N == 0:
        return 0.0

    y_true_arr = np.asarray(y_true).ravel()
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)

    if y_true_arr.dtype.kind not in ("i", "u", "b"):
        _, y_true_idx = np.unique(y_true_arr, return_inverse=True)
    else:
        y_true_idx = y_true_arr.astype(np.intp)

    correct = (predictions == y_true_idx).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    counts = np.bincount(bin_indices, minlength=n_bins)
    correct_sums = np.bincount(bin_indices, weights=correct, minlength=n_bins)
    conf_sums = np.bincount(bin_indices, weights=confidences, minlength=n_bins)

    has_samples = counts > 0
    acc_b = correct_sums[has_samples] / counts[has_samples]
    conf_b = conf_sums[has_samples] / counts[has_samples]
    return float(np.sum((counts[has_samples] / N) * np.abs(acc_b - conf_b)))


def optimize_temperature(
    logits: np.ndarray, y_indices: np.ndarray, n_classes: int, n_bins: int = 15,
) -> float:
    """Find the optimal softmax temperature (T) that minimizes ECE using Golden Section Search.

    SIMPLE EXPLANATION:
    -------------------
    Temperature scaling turns the 'confidence dial' up or down. If the model is overconfident,
    we increase T to soften probabilities. If underconfident, we lower T.
    Golden Section Search is a fast mathematical optimization algorithm that plays a high-low
    guessing game using the Golden Ratio (φ ≈ 1.618). It finds the best temperature between 0.1 and 10.0
    in just ~40 steps without needing derivatives.

    MATH LOGIC:
    -----------
    Target function: f(T) = compute_ece(y, softmax(Z / T))
    Interval [a, b] = [0.1, 10.0], Golden Ratio φ = (1 + sqrt(5)) / 2 ≈ 1.6180339

    At each iteration:
        c = b - (b - a) / φ
        d = a + (b - a) / φ
        If f(c) < f(d): set b = d  (minimum is in [a, d])
        Else:          set a = c  (minimum is in [c, b])
    Repeat until |b - a| < 1e-4. Return T_opt = (a + b) / 2.
    """
    PHI = (np.sqrt(5.0) + 1.0) / 2.0
    a, b, tol = 0.1, 10.0, 1e-4

    def ece_at_temp(T: float) -> float:
        P = stable_softmax(logits, temperature=T)
        return compute_ece(y_indices, P, n_bins)

    c = b - (b - a) / PHI
    d = a + (b - a) / PHI
    fc = ece_at_temp(c)
    fd = ece_at_temp(d)

    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d, fd = c, fc
            c = b - (b - a) / PHI
            fc = ece_at_temp(c)
        else:
            a = c
            c, fc = d, fd
            d = a + (b - a) / PHI
            fd = ece_at_temp(d)

    best_T = (a + b) / 2.0
    if ece_at_temp(1.0) <= ece_at_temp(best_T):
        return 1.0
    return best_T


# ---------------------------------------------------------------------------
# Vector (Platt-Style) Scaling — Per-Class Affine Calibration
# ---------------------------------------------------------------------------

def vector_scale_calibration(
    logits: np.ndarray, y_indices: np.ndarray, n_classes: int,
    max_iter: int = 200, lr: float = 0.01, tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn per-class scale (a_k) and shift (b_k) parameters to calibrate probabilities.

    SIMPLE EXPLANATION:
    -------------------
    Single temperature scaling uses 1 dial for all classes. But what if the model is overconfident
    about Class 1 ('Cat') but underconfident about Class 2 ('Dog')?
    Vector scaling gives EVERY class its own scale multiplier (a_k) and offset shift (b_k).
    It uses gradient descent to adjust these parameters until validation probabilities are honest.

    MATH LOGIC:
    -----------
    1. Calibrated Logits: Z'_{ik} = a_k * Z_{ik} + b_k
    2. Calibrated Probabilities: P_{ik} = softmax(Z'_i)_k
    3. Objective: Minimize Negative Log-Likelihood (NLL)
          NLL = - (1 / N) * Σ_i Σ_k Y_{ik} * log(P_{ik})
    4. Exact Partial Derivatives:
          Residual r_{ik} = (P_{ik} - Y_{ik}) / N
          ∂NLL / ∂a_k = Σ_i (r_{ik} * Z_{ik})
          ∂NLL / ∂b_k = Σ_i r_{ik}
    5. Gradient Descent Update:
          a_k = a_k - lr * (∂NLL / ∂a_k)
          b_k = b_k - lr * (∂NLL / ∂b_k)

    Returns:
        (scale_vector_a, shift_vector_b)
    """
    K = n_classes
    a = np.ones(K, dtype=np.float64)
    b = np.zeros(K, dtype=np.float64)
    N = len(y_indices)
    _EPS = 1e-15
    Y_one_hot = np.eye(K, dtype=np.float64)[y_indices]

    prev_nll = np.inf
    for _ in range(max_iter):
        cal_logits = logits * a + b
        P = stable_softmax(cal_logits, temperature=1.0)
        residual = (P - Y_one_hot) / N
        grad_a = np.sum(residual * logits, axis=0)
        grad_b = np.sum(residual, axis=0)
        a -= lr * grad_a
        b -= lr * grad_b

        # Convergence check to avoid wasted iterations
        nll = float(-np.sum(Y_one_hot * np.log(np.maximum(P, _EPS))) / N)
        if abs(prev_nll - nll) < tol:
            break
        prev_nll = nll

    return a, b
