"""Math tools for TowerGB."""

from __future__ import annotations

import numpy as np

# Small number to avoid division or log of zero
_EPS = 1e-15


def stable_softmax(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn scores into probabilities that add up to 1."""
    Z_c = np.ascontiguousarray(Z, dtype=np.float64)
    Z_scaled = Z_c / temperature
    Z_shifted = Z_scaled - Z_scaled.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy_loss_and_grad(X: np.ndarray, Y_one_hot: np.ndarray, W: np.ndarray, temperature: float = 1.0) -> tuple[float, np.ndarray]:
    """Calculate the mistake score (loss) and gradient."""
    N = X.shape[0]
    logits = X @ W
    P = stable_softmax(logits, temperature)
    loss_val = float(-np.sum(Y_one_hot * np.log(np.maximum(P, _EPS))) / N)
    residual = P - Y_one_hot
    grad = (X.T @ residual) / (N * temperature)
    return loss_val, grad


def compute_risk_metrics(P: np.ndarray, Y_one_hot: np.ndarray) -> dict[str, float]:
    """Check mistake size and prediction stability."""
    P_safe = np.maximum(P, _EPS)
    sample_losses = -np.sum(Y_one_hot * np.log(P_safe), axis=1)
    return {
        "log_loss": float(np.mean(sample_losses)),
        "brier": float(np.mean(np.sum((P - Y_one_hot) ** 2, axis=1))),
        "loss_var": float(np.var(sample_losses, ddof=0)),
    }


def pareto_arbiter(metrics_list: list[dict[str, float]], weights: dict[str, float]) -> np.ndarray:
    """Give a voting percentage to each training round."""
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
