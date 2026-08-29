"""Calibration tools for TowerGB."""

from __future__ import annotations

import numpy as np

from towergb._engine import stable_softmax


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Calculate the calibration error (ECE)."""
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


def optimize_temperature(logits: np.ndarray, y_indices: np.ndarray, n_classes: int, n_bins: int = 15) -> float:
    """Find the best temperature value using golden section search."""
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
