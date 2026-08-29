"""Post-hoc temperature calibration for TowerGB.

Implements Expected Calibration Error (ECE) computation and
golden-section search optimization of the temperature scaling
parameter *T*.

Temperature scaling is a single-parameter post-hoc calibration method
that preserves the model's discriminative ranking while adjusting the
sharpness of predicted probability distributions.  The optimal *T*
minimizes ECE on a held-out validation set.
"""

from __future__ import annotations

import numpy as np

from towergb._engine import stable_softmax


# ──────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ──────────────────────────────────────────────────────────────────────
def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    r"""Compute Expected Calibration Error (ECE).

    ECE measures the discrepancy between predicted confidence and
    empirical accuracy across equal-width confidence bins:

    .. math::

        \text{ECE} = \sum_{b=1}^{B}\frac{n_b}{N}
                     \bigl|\operatorname{acc}(b) - \operatorname{conf}(b)\bigr|

    where :math:`\operatorname{acc}(b)` is the fraction of correct predictions
    in bin *b* and :math:`\operatorname{conf}(b)` is the mean predicted
    confidence.

    Implementation uses ``np.digitize`` for O(N) bin assignment followed
    by O(B) per-bin aggregation — no nested loops over samples.

    Parameters
    ----------
    y_true : np.ndarray, shape ``(N,)``
        True class indices (integer-encoded).
    y_prob : np.ndarray, shape ``(N, K)``
        Predicted probability matrix (row-stochastic).
    n_bins : int, default 15
        Number of equally-spaced confidence bins in ``[0, 1]``.

    Returns
    -------
    ece : float
        Expected Calibration Error ∈ ``[0, 1]``.
    """
    N: int = len(y_true)
    if N == 0:
        return 0.0

    y_true_arr: np.ndarray = np.asarray(y_true).ravel()

    # Max predicted probability = model confidence
    confidences: np.ndarray = np.max(y_prob, axis=1)  # (N,)
    predictions: np.ndarray = np.argmax(y_prob, axis=1)  # (N,)
    correct: np.ndarray = (predictions == y_true_arr).astype(np.float64)

    # O(N) bin assignment using digitize
    bin_boundaries: np.ndarray = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize returns indices in [0, n_bins-1] when using interior edges
    bin_indices: np.ndarray = np.digitize(confidences, bin_boundaries[1:-1])

    # O(B) per-bin aggregation — B is a small constant (15)
    ece: float = 0.0
    for b in range(n_bins):
        mask: np.ndarray = bin_indices == b
        n_b: int = int(mask.sum())
        if n_b > 0:
            acc_b: float = float(correct[mask].mean())
            conf_b: float = float(confidences[mask].mean())
            ece += (n_b / N) * abs(acc_b - conf_b)

    return ece


# ──────────────────────────────────────────────────────────────────────
# Golden-Section Temperature Optimization
# ──────────────────────────────────────────────────────────────────────
def optimize_temperature(
    logits: np.ndarray,
    y_indices: np.ndarray,
    n_classes: int,
    n_bins: int = 15,
) -> float:
    r"""Optimize temperature *T* via golden-section search to minimize ECE.

    Golden-section search exploits the unimodal structure of
    :math:`\text{ECE}(T)` on the interval :math:`[0.1,\,10.0]`,
    converging in :math:`O\!\left(\log(1/\varepsilon)\right)` iterations
    without gradient computation.

    The golden ratio :math:`\varphi = (1+\sqrt{5})/2 \approx 1.618`
    ensures each iteration shrinks the search interval by factor
    :math:`1/\varphi \approx 0.618`, achieving optimal worst-case
    convergence among bracket methods.

    Parameters
    ----------
    logits : np.ndarray, shape ``(N, K)``
        Raw (un-temperature-scaled) logit matrix.
    y_indices : np.ndarray, shape ``(N,)``
        True class indices (integer-encoded).
    n_classes : int
        Number of distinct classes *K*.
    n_bins : int, default 15
        ECE bin count.

    Returns
    -------
    T_opt : float
        Optimal temperature ∈ ``(0, ∞)`` that minimizes ECE.
    """
    PHI: float = (np.sqrt(5.0) + 1.0) / 2.0  # Golden ratio ≈ 1.618

    a: float = 0.1
    b: float = 10.0
    tol: float = 1e-4

    def ece_at_temp(T: float) -> float:
        """Evaluate ECE at a given temperature T."""
        P: np.ndarray = stable_softmax(logits, temperature=T)
        return compute_ece(y_indices, P, n_bins)

    # Initialize two interior probe points
    c: float = b - (b - a) / PHI
    d: float = a + (b - a) / PHI
    fc: float = ece_at_temp(c)
    fd: float = ece_at_temp(d)

    # Iterate: each step reuses one probe (only 1 new evaluation per iter)
    while abs(b - a) > tol:
        if fc < fd:
            # Optimal T is in [a, d] — discard right portion
            b = d
            d, fd = c, fc
            c = b - (b - a) / PHI
            fc = ece_at_temp(c)
        else:
            # Optimal T is in [c, b] — discard left portion
            a = c
            c, fc = d, fd
            d = a + (b - a) / PHI
            fd = ece_at_temp(d)

    return (a + b) / 2.0
