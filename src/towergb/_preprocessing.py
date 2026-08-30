"""Preprocessing utilities for TowerGB.

This module provides data cleaning and feature engineering tools:
1. Missing value (NaN) imputation with column means.
2. Z-score feature standardization (mean = 0, std = 1).
3. Polynomial interaction feature expansion (e.g., x_i^2 and x_i * x_j).

Every function is documented with both a simple plain-English explanation
and the exact mathematical formulas.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Missing Value (NaN) Handling
# ---------------------------------------------------------------------------

def compute_nan_stats(
    X: np.ndarray, sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the average value of non-missing entries in each column.

    SIMPLE EXPLANATION:
    -------------------
    If a table has missing entries (blank cells or NaNs), we shouldn't just guess randomly.
    This function calculates the average (mean) of all known non-missing values in each column
    so we can fill in the blanks intelligently. If sample weights are provided, important
    rows contribute more to the column average.

    MATH LOGIC:
    -----------
    For column j over valid (non-NaN) indices:
        μ_j = Σ_{i ∈ valid} (w_i * X_{ij}) / Σ_{i ∈ valid} w_i
    If all values in column j are NaN, set μ_j = 0.0.

    Returns:
        (col_means, col_has_nan_boolean_mask)
    """
    col_has_nan = np.any(np.isnan(X), axis=0)
    if sample_weight is not None:
        sw = sample_weight.ravel()
        col_means = np.zeros(X.shape[1], dtype=np.float64)
        for j in range(X.shape[1]):
            valid = ~np.isnan(X[:, j])
            w_valid = sw[valid]
            w_sum = float(np.sum(w_valid))
            if w_sum > 0:
                col_means[j] = float(np.sum(X[valid, j] * w_valid) / w_sum)
            else:
                col_means[j] = 0.0
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
    return col_means, col_has_nan


def impute_nans(X: np.ndarray, col_means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace all missing (NaN) values in a matrix with pre-computed column means.

    SIMPLE EXPLANATION:
    -------------------
    Takes empty/blank cells in the table and fills them with the column's average value.
    This allows downstream math operations (like matrix multiplication) to work smoothly
    without breaking or producing undefined results.

    MATH LOGIC:
    -----------
    For every cell (i, j):
        If isnan(X_{ij}): X_{ij} = μ_j

    Returns:
        (X_imputed, nan_mask_boolean_matrix)
    """
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = X.copy()
        np.putmask(X, nan_mask, np.broadcast_to(col_means, X.shape))
    return X, nan_mask


# ---------------------------------------------------------------------------
# Feature Standardization (Z-Score Normalization)
# ---------------------------------------------------------------------------

def fit_standardizer(
    X: np.ndarray, sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column mean (μ) and standard deviation (σ) for z-score scaling.

    SIMPLE EXPLANATION:
    -------------------
    Imagine a table with 'Salary' ($50,000 to $150,000) and 'Years of Experience' (1 to 10).
    Without standardization, the huge Salary numbers would dominate the model's math, ignoring
    Experience.
    This function measures the average (mean) and spread (std) of every column so we can put
    all features on an equal playing field (centered around 0 with spread of 1).

    MATH LOGIC:
    -----------
    Weighted Mean:             μ_j = Σ (w_i * X_{ij}) / Σ w_i
    Weighted Variance:         σ_j^2 = Σ (w_i * (X_{ij} - μ_j)^2) / Σ w_i
    Weighted Standard Dev:     σ_j = sqrt(σ_j^2)
    Safety check:              If σ_j < 1e-12, set σ_j = 1.0 (prevents division by zero).

    Returns:
        (mean_vector, std_vector)
    """
    if sample_weight is not None:
        sw = sample_weight.ravel()
        sw_sum = float(np.sum(sw))
        if sw_sum > 0:
            mean = np.average(X, axis=0, weights=sw)
            var = np.average((X - mean) ** 2, axis=0, weights=sw)
            std = np.sqrt(var)
        else:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
    else:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Transform features into Z-scores using pre-computed mean and standard deviation.

    SIMPLE EXPLANATION:
    -------------------
    Converts raw feature numbers into Z-scores. A Z-score tells us:
    'How many standard deviations above or below average is this value?'
    0 means exactly average, +2 means far above average, -2 means far below average.

    MATH LOGIC:
    -----------
    Z_{ij} = (X_{ij} - μ_j) / σ_j
    """
    return np.asarray((X - mean) / std, dtype=np.float64)


# ---------------------------------------------------------------------------
# Polynomial Feature Expansion
# ---------------------------------------------------------------------------

def expand_polynomial(
    X: np.ndarray, degree: int = 2,
) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Expand features by generating non-linear squared terms and pair interactions.

    SIMPLE EXPLANATION:
    -------------------
    Standard linear models can only draw straight lines or flat planes. By adding
    squared terms (x_i^2) and cross-product terms (x_i * x_j), we create new synthetic
    features that let the model capture curved boundaries and interaction effects
    (for example: 'High Salary AND Low Experience → High Promotion Chance').

    MATH LOGIC:
    -----------
    Given original feature vector x = [x_1, x_2, ..., x_D]:
    For degree = 2, the expanded vector becomes:
        x_exp = [ x_1, x_2, ..., x_D,            (original linear terms)
                  x_1^2, x_2^2, ..., x_D^2,      (quadratic squared terms)
                  x_1*x_2, x_1*x_3, ..., x_{D-1}*x_D ]  (cross-interaction terms)

    Number of expanded features = D + D + D*(D-1)/2 = D*(D + 3)/2.

    Returns:
        (X_expanded, feature_map)
        where feature_map tracks which original feature index produced each expanded column.
    """
    _N, D = X.shape
    feature_map: list[tuple[int, ...]] = [(i,) for i in range(D)]

    if degree < 2:
        return X.copy(), feature_map

    parts = [X]

    # Squared terms: x_i²
    parts.append(X ** 2)
    feature_map.extend((i,) for i in range(D))

    # Cross terms: x_i · x_j  for all i < j
    if D > 1:
        idx_i, idx_j = np.triu_indices(D, k=1)
        parts.append(X[:, idx_i] * X[:, idx_j])
        feature_map.extend(
            (int(i), int(j)) for i, j in zip(idx_i, idx_j)
        )

    return np.hstack(parts), feature_map
