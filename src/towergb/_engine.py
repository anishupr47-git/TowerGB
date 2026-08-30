"""Math tools and core optimization algorithms for TowerGB.

This module provides pure mathematical functions for probability calculation,
loss evaluation, gradient computation, Adam optimization, and ensemble voting.
Every function is documented with both a simple plain-English explanation
and the exact mathematical formulas.
"""

from __future__ import annotations

import numpy as np

# Extremely small constant to prevent log(0) or division by zero
_EPS = 1e-15


def _safe_weight_sum(sw: np.ndarray) -> float:
    """Calculate total sample weights cleanly without division-by-zero risk.

    SIMPLE EXPLANATION:
    -------------------
    When rows have custom importance weights (like high-priority fraud cases),
    we sum up all weights. If all weights happen to be zero, we fall back to
    counting the total number of rows so our math never breaks (avoids dividing by 0).

    MATH LOGIC:
    -----------
    W_sum = Σ w_i  for i in [1, N]
    If W_sum > 0 return W_sum else return N.
    """
    total = float(np.sum(sw))
    return total if total > 0.0 else float(len(sw))


def stable_softmax(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert raw neural scores (logits) into clean probabilities that sum to 100% (1.0).

    SIMPLE EXPLANATION:
    -------------------
    Imagine an exam where raw scores are [10, 5, 2]. Softmax turns these raw scores
    into percentage probabilities (e.g. 95%, 4%, 1%).
    To prevent large numbers from exploding and crashing the computer (floating-point overflow),
    we subtract the highest score from all scores before exponentiating.
    The temperature parameter acts like a confidence dial: higher temperature makes
    probabilities more spread out (uncertain), lower temperature makes them sharper.

    MATH LOGIC:
    -----------
    1. Scale by temperature: Z_scaled = Z / T
    2. Subtract max for stability: Z_shifted = Z_scaled - max(Z_scaled)
    3. Exponentiate: exp_Z = exp(Z_shifted)
    4. Normalize: P_i = exp_Z_i / Σ_j exp_Z_j

    Formula:
        P_i = exp((Z_i - max(Z)) / T) / Σ_j exp((Z_j - max(Z)) / T)
    """
    Z_c = np.ascontiguousarray(Z, dtype=np.float64)
    Z_scaled = Z_c / temperature
    Z_shifted = Z_scaled - Z_scaled.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Optimizer Primitives (Adam & Gradient Clipping)
# ---------------------------------------------------------------------------

def adam_step(
    grad: np.ndarray, m: np.ndarray, v: np.ndarray, t: int,
    lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute one update step using the Adaptive Moment Estimation (Adam) optimizer.

    SIMPLE EXPLANATION:
    -------------------
    Plain gradient descent is like walking downhill step-by-step. Adam is like a smart heavy
    ball rolling down a hill:
    1. 'm' (first moment) acts like momentum — keeping speed in the general downhill direction.
    2. 'v' (second moment) acts like automatic shock absorbers — taking smaller steps on steep,
       bumpy terrain and larger steps on smooth, gentle slopes.
    This helps the model converge much faster and get stuck far less often.

    MATH LOGIC:
    -----------
    At time step t with gradient g_t:
    1. Update 1st moment (mean gradient):         m_t = β_1 * m_{t-1} + (1 - β_1) * g_t
    2. Update 2nd moment (uncentered variance):   v_t = β_2 * v_{t-1} + (1 - β_2) * (g_t)^2
    3. Correct 1st moment bias (for early t):     m̂_t = m_t / (1 - (β_1)^t)
    4. Correct 2nd moment bias (for early t):     v̂_t = v_t / (1 - (β_2)^t)
    5. Compute parameter step:                    step = lr * m̂_t / (sqrt(v̂_t) + ε)

    Returns:
        (step, updated_m, updated_v)
    """
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    step = lr * m_hat / (np.sqrt(v_hat) + eps)
    return step, m, v


def clip_gradient(grad: np.ndarray, max_norm: float) -> np.ndarray:
    """Cap gradient vector length to prevent exploding steps while keeping direction intact.

    SIMPLE EXPLANATION:
    -------------------
    If the model receives wild, unexpected data, gradients can become massive and cause
    violent weight jumps. Gradient clipping measures the overall length (L2 norm) of the gradient
    vector. If it's longer than `max_norm`, we shrink the vector down to `max_norm` length
    without altering its direction.

    MATH LOGIC:
    -----------
    L2 norm: ||g||_2 = sqrt(Σ g_i^2)
    If ||g||_2 > max_norm:
        g_clipped = g * (max_norm / ||g||_2)
    Else:
        g_clipped = g
    """
    grad_norm = float(np.linalg.norm(grad))
    if grad_norm > max_norm:
        return grad * (max_norm / grad_norm)
    return grad


# ---------------------------------------------------------------------------
# Loss & Gradient Calculation
# ---------------------------------------------------------------------------

def cross_entropy_loss_and_grad(
    X: np.ndarray, Y_one_hot: np.ndarray, W: np.ndarray,
    temperature: float = 1.0, l2_reg: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute Categorical Cross-Entropy loss and exact analytical gradient.

    SIMPLE EXPLANATION:
    -------------------
    - Loss (Mistake Score): Measures how far the AI's predicted probabilities are from
      the actual true labels. Perfect guess = 0 loss; confident wrong guess = high penalty.
    - Gradient (Correction Direction): Tells us exactly how much to tweak each weight matrix entry
      to make fewer mistakes on the next iteration.
    - L2 Regularization (Weight Penalty): Adds a small cost for large weight values so the model
      remains simple and doesn't overfit to noise.

    MATH LOGIC:
    -----------
    1. Compute raw logits: Z = X @ W
    2. Compute probabilities: P = softmax(Z / T)
    3. Categorical Cross-Entropy:
          Loss_unweighted = - Σ_k (Y_k * log(max(P_k, ε)))
          Loss = Σ (w_i * Loss_i) / Σ w_i  [or mean across N if no sample_weight]
    4. Gradient w.r.t weights W:
          Residual R = P - Y_one_hot  (difference between predicted & actual)
          ∇_W Loss = (X^T @ (R * w)) / (W_sum * T)
    5. L2 Weight Penalty Addition:
          Loss_total = Loss + 0.5 * λ * ||W||_F^2
          ∇_W Loss_total = ∇_W Loss + λ * W
    """
    logits = X @ W
    P = stable_softmax(logits, temperature)

    if sample_weight is not None:
        sw = sample_weight.ravel()
        w_sum = _safe_weight_sum(sw)
        sample_losses = -np.sum(Y_one_hot * np.log(np.maximum(P, _EPS)), axis=1)
        loss_val = float(np.sum(sw * sample_losses) / w_sum)
        residual = (P - Y_one_hot) * sw[:, None]
        grad = (X.T @ residual) / (w_sum * temperature)
    else:
        N = X.shape[0]
        loss_val = float(-np.sum(Y_one_hot * np.log(np.maximum(P, _EPS))) / N)
        residual = P - Y_one_hot
        grad = (X.T @ residual) / (N * temperature)

    if l2_reg > 0.0:
        loss_val += 0.5 * l2_reg * float(np.sum(W ** 2))
        grad += l2_reg * W

    return loss_val, grad


# ---------------------------------------------------------------------------
# Risk Metrics & Pareto Ensemble Arbiter
# ---------------------------------------------------------------------------

def compute_risk_metrics(
    P: np.ndarray, Y_one_hot: np.ndarray, sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate statistical risk metrics: Log-Loss, Brier Score, and Loss Variance.

    SIMPLE EXPLANATION:
    -------------------
    This checks model quality across 3 dimensions:
    1. Log-Loss: Penalizes confident wrong answers.
    2. Brier Score: Measures mean squared distance between prediction probability and 1.0 or 0.0.
    3. Loss Variance: Checks if error is steady across all rows or wildly volatile.

    MATH LOGIC:
    -----------
    - Sample Log Loss:  l_i = -Σ_k Y_{ik} log(P_{ik})
    - Sample Brier:     b_i = Σ_k (P_{ik} - Y_{ik})^2
    - Weighted Means:   LogLoss = Σ(w_i l_i)/W_sum,  Brier = Σ(w_i b_i)/W_sum
    - Loss Variance:    Var(l) = Σ(w_i (l_i - MeanLogLoss)^2) / W_sum
    """
    P_safe = np.maximum(P, _EPS)
    sample_losses = -np.sum(Y_one_hot * np.log(P_safe), axis=1)
    sample_brier = np.sum((P - Y_one_hot) ** 2, axis=1)

    if sample_weight is not None:
        sw = sample_weight.ravel()
        w_sum = _safe_weight_sum(sw)
        mean_loss = float(np.sum(sw * sample_losses) / w_sum)
        mean_brier = float(np.sum(sw * sample_brier) / w_sum)
        var_loss = float(np.sum(sw * (sample_losses - mean_loss) ** 2) / w_sum)
    else:
        mean_loss = float(np.mean(sample_losses))
        mean_brier = float(np.mean(sample_brier))
        var_loss = float(np.var(sample_losses, ddof=0))

    return {
        "log_loss": mean_loss,
        "brier": mean_brier,
        "loss_var": var_loss,
    }


def pareto_arbiter(
    metrics_list: list[dict[str, float]], weights: dict[str, float],
) -> np.ndarray:
    """Compute optimal ensemble voting weights based on multi-objective fitness.

    SIMPLE EXPLANATION:
    -------------------
    Acts as an impartial referee for ensemble passes. Passes that are accurate,
    have low loss, low Brier distance, and low loss variance earn a higher 'fitness' score.
    Fitness scores are converted into voting percentages that add up to 1.0 (100%).

    MATH LOGIC:
    -----------
    1. Fitness Score per pass m:
          Fitness_m = w_acc * Accuracy_m - w_brier * Brier_m - w_loss * LogLoss_m - w_var * LossVar_m
    2. Convert fitness to ensemble weights via Softmax:
          α_m = exp(Fitness_m - max(Fitness)) / Σ_k exp(Fitness_k - max(Fitness))
    """
    w_acc = weights.get("accuracy", 1.0)
    w_brier = weights.get("brier", 0.5)
    w_loss = weights.get("log_loss", 0.5)
    w_var = weights.get("loss_var", 0.3)

    accs = np.array([m.get("accuracy", 0.0) for m in metrics_list], dtype=np.float64)
    briers = np.array([m.get("brier", 0.0) for m in metrics_list], dtype=np.float64)
    log_losses = np.array([m.get("log_loss", 0.0) for m in metrics_list], dtype=np.float64)
    loss_vars = np.array([m.get("loss_var", 0.0) for m in metrics_list], dtype=np.float64)

    fitness = w_acc * accs - w_brier * briers - w_loss * log_losses - w_var * loss_vars
    fitness_shifted = fitness - fitness.max()
    exp_fitness = np.exp(fitness_shifted)
    return np.asarray(exp_fitness / exp_fitness.sum(), dtype=np.float64)
