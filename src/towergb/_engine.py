"""Math tools for TowerGB.

This file contains functions to:
1. Turn scores into chances that add up to 100% (softmax).
2. Measure mistakes and calculate how to fix them (loss and gradient).
3. Check how risky or confident the model is (risk metrics).
4. Pick the best models to trust the most (arbiter).
"""

from __future__ import annotations

import numpy as np

# A very small number so we never divide by zero or take log of zero
_EPS: float = 1e-15


def stable_softmax(Z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn numbers into probabilities that add up to 1 for each row.

    High numbers get higher chances.
    Temperature changes how confident the chances look.
    """
    # Make sure numbers are stored cleanly in memory
    Z_c: np.ndarray = np.ascontiguousarray(Z, dtype=np.float64)

    # Scale numbers by temperature and subtract the max so numbers do not blow up
    Z_scaled: np.ndarray = Z_c / temperature
    Z_shifted: np.ndarray = Z_scaled - Z_scaled.max(axis=1, keepdims=True)

    # Exponentiate and divide by total so every row adds up to 1
    exp_Z: np.ndarray = np.exp(Z_shifted)
    P: np.ndarray = exp_Z / exp_Z.sum(axis=1, keepdims=True)

    return P


def cross_entropy_loss_and_grad(
    X: np.ndarray,
    Y_one_hot: np.ndarray,
    W: np.ndarray,
    temperature: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Calculate the mistake score (loss) and how to change weights to fix it (gradient).

    X: input data table
    Y_one_hot: correct answers (1 for true class, 0 for others)
    W: weights used for guessing
    temperature: confidence scaler
    """
    N: int = X.shape[0]

    # Calculate scores and turn them into chances
    logits: np.ndarray = X @ W
    P: np.ndarray = stable_softmax(logits, temperature)

    # Calculate how wrong the guesses are on average
    loss_val: float = float(
        -np.sum(Y_one_hot * np.log(np.maximum(P, _EPS))) / N
    )

    # Calculate the direction to adjust weights to reduce mistakes
    residual: np.ndarray = P - Y_one_hot
    grad: np.ndarray = (X.T @ residual) / (N * temperature)

    return loss_val, grad


def compute_risk_metrics(
    P: np.ndarray,
    Y_one_hot: np.ndarray,
) -> dict[str, float]:
    """Check how good, careful, and steady the predictions are.

    Returns:
    - log_loss: average mistake size
    - brier: difference between guess chance and true answer squared
    - loss_var: how steady the model is across different rows
    """
    # Keep chances safe from zero
    P_safe: np.ndarray = np.maximum(P, _EPS)

    # Check mistakes for each row
    sample_losses: np.ndarray = -np.sum(Y_one_hot * np.log(P_safe), axis=1)

    return {
        "log_loss": float(np.mean(sample_losses)),
        "brier": float(np.mean(np.sum((P - Y_one_hot) ** 2, axis=1))),
        "loss_var": float(np.var(sample_losses, ddof=0)),
    }


def pareto_arbiter(
    metrics_list: list[dict[str, float]],
    weights: dict[str, float],
) -> np.ndarray:
    """Give a score to each round of training and pick who gets more vote.

    Rounds with higher accuracy and lower mistakes get higher votes.
    """
    # Get importance weights for each metric
    w_acc: float = weights.get("accuracy", 1.0)
    w_brier: float = weights.get("brier", 0.5)
    w_loss: float = weights.get("log_loss", 0.5)
    w_var: float = weights.get("loss_var", 0.3)

    # Collect numbers from all rounds
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

    # Calculate overall grade for each round
    fitness: np.ndarray = (
        w_acc * accs - w_brier * briers - w_loss * log_losses - w_var * loss_vars
    )

    # Turn grades into vote percentages that add up to 100%
    fitness_shifted: np.ndarray = fitness - fitness.max()
    exp_fitness: np.ndarray = np.exp(fitness_shifted)
    ensemble_weights: np.ndarray = exp_fitness / exp_fitness.sum()

    return ensemble_weights
