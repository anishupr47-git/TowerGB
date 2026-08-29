# TowerGB

TowerGB is a fast, enterprise-ready machine learning model for tabular classification. It features a dual-tower ensemble architecture with built-in probability calibration, feature importance attribution, class balancing, and sub-millisecond inference.

[![CI](https://github.com/anishupr47-git/TableGB/actions/workflows/ci.yml/badge.svg)](https://github.com/anishupr47-git/TableGB/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## How It Works

TowerGB works in four steps:

1. **Tower 1 (Accuracy)**: Trains bootstrap passes and evaluates accuracy.
2. **Tower 2 (Risk & Calibration)**: Evaluates log-loss, Brier score, and sample-wise loss variance.
3. **Pareto Arbiter**: Computes optimal ensemble voting weights based on accuracy and risk.
4. **Temperature Calibration**: Minimizes Expected Calibration Error (ECE) via Golden Section Search.

---

## Key Features

- **Zero Heavy Dependencies**: Pure NumPy and Scikit-Learn.
- **Sub-Millisecond Latency**: 0.09 ms per batch inference via single collapsed matrix multiplication.
- **Feature Importances & Coefficients**: Full interpretability via `.feature_importances_`, `.coef_`, and `.intercept_`.
- **Imbalanced Data Ready**: Built-in `class_weight='balanced'` and `sample_weight` support.
- **L2 Regularization**: Built-in weight decay (`l2_reg`) to prevent overfitting.
- **Parallel Training**: Native multi-core CPU scaling (`n_jobs=-1`).
- **Scikit-Learn Standard**: 100% compliant with `Pipeline`, `GridSearchCV`, `cross_val_score`, and serialization.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/anishupr47-git/TableGB.git
cd TableGB

# Create a virtual environment
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Or activate on macOS/Linux:
# source .venv/bin/activate

# Install the package
pip install -e ".[test]"
```

---

## Quick Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from towergb import TowerGBClassifier

# 1. Create example table data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                           n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create and train the model with L2 regularization and balanced weights
clf = TowerGBClassifier(n_passes=5, learning_rate=0.1, l2_reg=1e-4, random_state=42)
clf.fit(X_train, y_train)

# 3. Check accuracy
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")

# 4. View feature importance ranking
print(f"Top feature importance: {clf.feature_importances_[:5]}")

# 5. Tune confidence probabilities
clf.calibrate(X_test, y_test)
proba = clf.predict_proba(X_test)
print(f"Calibrated probabilities shape: {proba.shape}")
```

---

## Settings and Options

### `TowerGBClassifier` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_passes` | `int` | `5` | Number of ensemble passes |
| `learning_rate` | `float` | `0.1` | Gradient descent learning rate |
| `max_iter` | `int` | `300` | Maximum iterations per pass |
| `temperature` | `float` | `1.0` | Initial softmax temperature |
| `l2_reg` | `float` | `1e-4` | L2 weight regularization penalty |
| `subsample_ratio` | `float` | `1.0` | Subsampling ratio per pass |
| `class_weight` | `str \| dict \| None` | `None` | Class balancing (e.g. `'balanced'`) |
| `tol` | `float` | `1e-6` | Convergence tolerance |
| `random_state` | `int \| None` | `None` | Random seed |
| `arbiter_weights` | `dict \| None` | `None` | Multi-objective Pareto arbiter weights |
| `n_jobs` | `int \| None` | `None` | CPU cores for parallel pass training |

### Public Attributes

| Attribute | Type | Description |
|---|---|---|
| `classes_` | `ndarray` | Unique class labels |
| `n_features_in_` | `int` | Number of features seen during fit |
| `feature_importances_` | `ndarray` | Normalized importance score per feature (sums to 1.0) |
| `coef_` | `ndarray` | Learned feature weight coefficients |
| `intercept_` | `ndarray` | Learned bias intercepts |
| `temperature_` | `float` | Calibrated softmax temperature |
| `n_iter_` | `ndarray` | Iterations executed per pass |

### Main Methods

| Method | Description |
|---|---|
| `.fit(X, y, sample_weight=None)` | Train the ensemble |
| `.predict(X)` | Predict class label |
| `.predict_proba(X)` | Calibrated probability estimates |
| `.calibrate(X_val, y_val)` | Post-hoc ECE temperature optimization |
| `.score(X, y, sample_weight=None)` | Accuracy score |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with test coverage
pytest tests/ -v --cov=towergb --cov-report=term-missing
```

---

## Running Benchmarks

```bash
# Install benchmark tools
pip install -e ".[benchmark]"

# Run speed and accuracy comparison
python benchmarks/run_benchmarks.py
```

---

## License

[MIT](LICENSE)
