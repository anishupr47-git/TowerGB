"""Calibration tools for TowerGB.

This file adjusts model confidence so predicted percentages match real-world accuracy.
If the model says 80% confident, it should be right 8 times out of 10.
"""

from __future__ import annotations

import numpy as np

from towergb._engine import stable_softmax


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Calculate the calibration error.

    This measures how close confidence percentages are to actual correct answers.
    A lower number means the model is more honest about its confidence.
    """
    N: int = len(y_true)
    if N == 0:
        return 0.0

    y_true_arr: np.ndarray = np.asarray(y_true).ravel()

    # Find highest probability and the chosen category for each row
    confidences: np.ndarray = np.max(y_prob, axis=1)
    predictions: np.ndarray = np.argmax(y_prob, axis=1)
    correct: np.ndarray = (predictions == y_true_arr).astype(np.float64)

    # Group confidence into equal buckets from 0 to 1
    bin_boundaries: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices: np.ndarray = np.digitize(confidences, bin_boundaries[1:-1])

    # Count samples, correct answers, and confidence totals for each bucket in one C pass
    counts: np.ndarray = np.bincount(bin_indices, minlength=n_bins)
    correct_sums: np.ndarray = np.bincount(bin_indices, weights=correct, minlength=n_bins)
    conf_sums: np.ndarray = np.bincount(bin_indices, weights=confidences, minlength=n_bins)

    # Only look at buckets that have at least one sample
    has_samples: np.ndarray = counts > 0
    acc_b: np.ndarray = correct_sums[has_samples] / counts[has_samples]
    conf_b: np.ndarray = conf_sums[has_samples] / counts[has_samples]
    ece: float = float(np.sum((counts[has_samples] / N) * np.abs(acc_b - conf_b)))

    return ece


def optimize_temperature(
    logits: np.ndarray,
    y_indices: np.ndarray,
    n_classes: int,
    n_bins: int = 15,
) -> float:
    """Find the best temperature value to make confidence honest.

    Uses a fast search method to find the temperature that gives the lowest error.
    """
    PHI: float = (np.sqrt(5.0) + 1.0) / 2.0  # Golden ratio number

    a: float = 0.1
    b: float = 10.0
    tol: float = 1e-4

    def ece_at_temp(T: float) -> float:
        # Helper to test one temperature value
        P: np.ndarray = stable_softmax(logits, temperature=T)
        return compute_ece(y_indices, P, n_bins)

    # Pick test points inside the range
    c: float = b - (b - a) / PHI
    d: float = a + (b - a) / PHI
    fc: float = ece_at_temp(c)
    fd: float = ece_at_temp(d)

    # Narrow down the range until we find the best number
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

    return (a + b) / 2.0
