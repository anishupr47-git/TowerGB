# TowerGB

TowerGB is a fast and simple machine learning model for tabular data (data in rows and columns). It predicts which category a row belongs to.

[![CI](https://github.com/anishupr47-git/TableGB/actions/workflows/ci.yml/badge.svg)](https://github.com/anishupr47-git/TableGB/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## How It Works

TowerGB works in four simple steps:

1. **Tower 1 (Accuracy)**: Trains several rounds of models and checks how many answers each round gets right.
2. **Tower 2 (Risk and Confidence)**: Checks how steady and careful each round is so it avoids making overconfident mistakes.
3. **Arbiter**: Combines the rounds by giving more voting power to the rounds that are accurate and steady.
4. **Calibration**: Tunes confidence percentages so when the model says it is 80% sure, it really is right 80% of the time.

---

## Key Features

- **No heavy dependencies**: Only uses NumPy and Scikit-Learn.
- **Fast**: Fast training and sub-millisecond predictions.
- **Scikit-Learn Compatible**: Works with standard tools like pipelines and cross-validation.

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

# Install the package and test tools
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

# 2. Create and train the model
clf = TowerGBClassifier(n_passes=5, learning_rate=0.1, random_state=42)
clf.fit(X_train, y_train)

# 3. Check accuracy
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")

# 4. Tune confidence probabilities
clf.calibrate(X_test, y_test)
proba = clf.predict_proba(X_test)
print(f"Probabilities shape: {proba.shape}")
```

---

## Settings and Options

### `TowerGBClassifier` Parameters

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `n_passes` | `int` | `5` | Number of training rounds |
| `learning_rate` | `float` | `0.1` | Step size when learning |
| `max_iter` | `int` | `300` | Maximum learning steps per round |
| `temperature` | `float` | `1.0` | Starting confidence scale |
| `subsample_ratio` | `float` | `0.8` | Percentage of data used per round |
| `tol` | `float` | `1e-6` | Stop learning when error stops changing |
| `random_state` | `int \| None` | `None` | Random seed for repeatable results |
| `arbiter_weights` | `dict \| None` | `None` | Custom voting importance weights |

### Main Methods

| Method | What it does |
|---|---|
| `.fit(X, y)` | Trains the model on table data |
| `.predict(X)` | Predicts the category for each row |
| `.predict_proba(X)` | Gives percentage chances for each category |
| `.calibrate(X_val, y_val)` | Tunes confidence scores on test data |
| `.score(X, y)` | Calculates accuracy score |

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
