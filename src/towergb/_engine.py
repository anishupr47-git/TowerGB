"""Core computational engine for TowerGB.

Mathematical primitives for the dual-tower architecture:

    ▸ stable_softmax           — Numerically stable temperature-scaled softmax
    ▸ cross_entropy_loss_and_grad — Forward CE loss + analytical ∂L/∂W
    ▸ compute_risk_metrics     — Tower 2 risk signals (log-loss, Brier, Var)
    ▸ pareto_arbiter           — Cross-tower fitness scoring → ensemble weights

Design invariants:
    • All operations are fully vectorized (no Python loops over N or D).
    • Arrays are enforced C-contiguous for CPU cache-line optimization.
    • BLAS dgemm acceleration via numpy's ``@`` operator for O(N·D·K) matmuls.
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# Machine-epsilon floor to prevent log(0) → -inf.
_EPS: float = 1e-15


# ──────────────────────────────────────────────────────────────────────
# Softmax
# ──────────────────────────────────────────────────────────────────────
def stable_softmax(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    r"""Numerically stable row-wise softmax with temperature scaling.

    .. math::

        P_{ik} = \frac{\exp\!\bigl(Z_{ik}/T - \max_j Z_{ij}/T\bigr)}
                      {\sum_j \exp\!\bigl(Z_{ij}/T - \max_j Z_{ij}/T\bigr)}

    The ``max``-subtraction trick eliminates overflow for arbitrarily large
    logits (tested up to :math:`|Z| > 10^4`) while preserving exact
    mathematical equivalence — the shift cancels in the ratio.

    Parameters
    ----------
    Z : np.ndarray, shape ``(N, K)``
        Raw logit matrix.  Forced C-contiguous for cache efficiency.
    temperature : float, default 1.0
        Temperature scaling :math:`T > 0`.
        *Higher* T → softer (more uniform) distribution.
        *Lower*  T → sharper (toward hard argmax).

    Returns
    -------
    P : np.ndarray, shape ``(N, K)``, dtype ``float64``
        Row-stochastic probability matrix (:math:`\sum_k P_{ik} = 1`).
    """
    # Enforce C-contiguous float64 layout for optimal cache utilization
    Z_c: np.ndarray = np.ascontiguousarray(Z, dtype=np.float64)

    # Temperature-scaled logits with per-row max subtraction
    Z_scaled: np.ndarray = Z_c / temperature
    Z_shifted: np.ndarray = Z_scaled - Z_scaled.max(axis=1, keepdims=True)

    # Vectorized exponentiation and row-normalization
    exp_Z: np.ndarray = np.exp(Z_shifted)
    P: np.ndarray = exp_Z / exp_Z.sum(axis=1, keepdims=True)

    return P


# ──────────────────────────────────────────────────────────────────────
# Cross-Entropy Loss + Gradient
# ──────────────────────────────────────────────────────────────────────
def cross_entropy_loss_and_grad(
    X: np.ndarray,
    Y_one_hot: np.ndarray,
    W: np.ndarray,
    temperature: float = 1.0,
) -> tuple[float, np.ndarray]:
    r"""Compute mean cross-entropy loss and its analytical gradient w.r.t. W.

    **Forward pass:**

    .. math::

        \text{logits} = X\,W,\qquad P = \operatorname{softmax}\!\left(
        \frac{\text{logits}}{T}\right)

    **Loss (mean cross-entropy):**

    .. math::

        \mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}
                       Y_{ik}\,\ln(P_{ik} + \varepsilon)

    **Gradient (softmax–CE Jacobian identity):**

    .. math::

        \frac{\partial\mathcal{L}}{\partial W}
        = \frac{1}{N \cdot T}\,X^\top\!(P - Y)

    The elegant residual form :math:`(P - Y)` arises because the Jacobian
    :math:`\partial P / \partial\,\text{logits}` contracted with the CE
    derivative collapses to a single subtraction.

    Parameters
    ----------
    X : np.ndarray, shape ``(N, D)``
        Feature matrix (C-contiguous preferred).
    Y_one_hot : np.ndarray, shape ``(N, K)``
        One-hot encoded target matrix.
    W : np.ndarray, shape ``(D, K)``
        Weight matrix mapping features → class logits.
    temperature : float, default 1.0
        Softmax temperature scaling.

    Returns
    -------
    loss : float
        Scalar mean cross-entropy loss.
    grad : np.ndarray, shape ``(D, K)``
        Analytical gradient :math:`\partial\mathcal{L}/\partial W`.
    """
    N: int = X.shape[0]

    # Forward: logits → probabilities via temperature-scaled softmax
    # Single BLAS dgemm call for the N×D @ D×K multiply
    logits: np.ndarray = X @ W  # (N, K)
    P: np.ndarray = stable_softmax(logits, temperature)

    # Cross-entropy with ε-floor to prevent log(0)
    loss_val: float = float(
        -np.sum(Y_one_hot * np.log(np.maximum(P, _EPS))) / N
    )

    # Gradient via softmax–CE residual identity: dL/dW = X^T(P−Y) / (N·T)
    residual: np.ndarray = P - Y_one_hot  # (N, K)
    grad: np.ndarray = (X.T @ residual) / (N * temperature)  # (D, K) — single dgemm

    return loss_val, grad


# ──────────────────────────────────────────────────────────────────────
# Tower 2: Risk Metrics
# ──────────────────────────────────────────────────────────────────────
def compute_risk_metrics(
    P: np.ndarray,
    Y_one_hot: np.ndarray,
) -> dict[str, float]:
    r"""Compute Tower 2 risk and calibration metrics.

    Three complementary signals quantify prediction quality:

    1. **Log-loss** (information-theoretic divergence):

       .. math::

           \bar{\mathcal{L}} = -\frac{1}{N}\sum_{i}\sum_{k}
                                Y_{ik}\,\ln P_{ik}

    2. **Brier score** (proper scoring rule, L2 in probability space):

       .. math::

           \mathcal{B} = \frac{1}{N}\sum_{i}\sum_{k}
                          (P_{ik} - Y_{ik})^2

    3. **Sample-wise loss variance** (calibration stability):

       .. math::

           \operatorname{Var}(\mathcal{L}) =
           \operatorname{Var}\!\left(\left\{
           -\sum_k Y_{ik}\ln P_{ik}\right\}_{i=1}^N\right)

       High variance signals inconsistent confidence — the model is
       certain on some samples but uncertain on others, a hallmark
       of poor calibration.

    Parameters
    ----------
    P : np.ndarray, shape ``(N, K)``
        Predicted probability matrix (row-stochastic).
    Y_one_hot : np.ndarray, shape ``(N, K)``
        One-hot encoded targets.

    Returns
    -------
    metrics : dict
        Keys: ``'log_loss'``, ``'brier'``, ``'loss_var'``.
    """
    # ε-clamped probabilities — fully vectorized
    P_safe: np.ndarray = np.maximum(P, _EPS)

    # Per-sample cross-entropy losses: shape (N,)
    sample_losses: np.ndarray = -np.sum(Y_one_hot * np.log(P_safe), axis=1)

    return {
        "log_loss": float(np.mean(sample_losses)),
        "brier": float(np.mean(np.sum((P - Y_one_hot) ** 2, axis=1))),
        "loss_var": float(np.var(sample_losses, ddof=0)),
    }


# ──────────────────────────────────────────────────────────────────────
# Cross-Tower Pareto Arbiter
# ──────────────────────────────────────────────────────────────────────
def pareto_arbiter(
    metrics_list: list[dict[str, float]],
    weights: dict[str, float],
) -> np.ndarray:
    r"""Cross-Tower Pareto Arbiter for multi-pass ensemble weighting.

    Computes a composite fitness score per training pass:

    .. math::

        F_m = w_{\text{acc}}\!\cdot\!\text{Acc}_m
            - w_{\text{brier}}\!\cdot\!\mathcal{B}_m
            - w_{\text{loss}}\!\cdot\!\bar{\mathcal{L}}_m
            - w_{\text{var}}\!\cdot\!\operatorname{Var}(\mathcal{L}_m)

    Then applies softmax to yield ensemble weights:

    .. math::

        \alpha_m = \frac{\exp(F_m)}{\sum_j \exp(F_j)}

    Passes with higher fitness (better accuracy, lower loss/risk) receive
    proportionally greater weight in the final soft-probability ensemble.

    Parameters
    ----------
    metrics_list : list of dict
        Per-pass metrics. Each dict: ``'accuracy'``, ``'log_loss'``,
        ``'brier'``, ``'loss_var'``.
    weights : dict
        Arbiter coefficients: ``'accuracy'``, ``'log_loss'``, ``'brier'``,
        ``'loss_var'``.

    Returns
    -------
    ensemble_weights : np.ndarray, shape ``(M,)``
        Softmax-normalized ensemble weights (:math:`\sum_m \alpha_m = 1`).
    """
    # Extract arbiter coefficients with sensible defaults
    w_acc: float = weights.get("accuracy", 1.0)
    w_brier: float = weights.get("brier", 0.5)
    w_loss: float = weights.get("log_loss", 0.5)
    w_var: float = weights.get("loss_var", 0.3)

    # Vectorize metric extraction → O(M) numpy ops instead of scalar loop
    accs: np.ndarray = np.array(
        [m.get("accuracy", 0.0) for m in metrics_list], dtype=np.float64
    )
    briers: np.ndarray = np.array(
        [m.get("brier", 0.0) for m in metrics_list], dtype=np.float64
    )
    log_losses: np.ndarray = np.array(
        [m.get("log_loss", 0.0) for m in metrics_list], dtype=np.float64
    )
    loss_vars: np.ndarray = np.array(
        [m.get("loss_var", 0.0) for m in metrics_list], dtype=np.float64
    )

    # Composite fitness: vectorized linear combination
    fitness: np.ndarray = (
        w_acc * accs - w_brier * briers - w_loss * log_losses - w_var * loss_vars
    )

    # Softmax with max-subtraction for numerical stability
    fitness_shifted: np.ndarray = fitness - fitness.max()
    exp_fitness: np.ndarray = np.exp(fitness_shifted)
    ensemble_weights: np.ndarray = exp_fitness / exp_fitness.sum()

    return ensemble_weights
