# Learning TowerGB: A Simple Walkthrough

Welcome! This guide explains how **TowerGB** works in simple, friendly words. No complicated jargon—just clear, step-by-step explanations.

---

## 1. What is TowerGB? (The Big Idea)

Imagine you are trying to guess what animal is in a picture (Cat, Dog, or Bird) based on clues like size, ear shape, and tail length.

Instead of asking just **one** person to guess:
- TowerGB asks a **team of 5 helpers** to look at the clues.
- Some helpers might be super accurate.
- Some helpers might be too confident and make big mistakes.
- A smart referee (**The Arbiter**) checks all 5 helpers and gives **more voting power** to the helpers that are accurate, careful, and steady.
- Finally, the model tunes its confidence percentage (**Calibration**) so that if it says "I am 80% sure," it is actually right 8 times out of 10!

---

## 2. The 4 Steps of How It Works

Here is the exact journey your data takes from start to finish:

```
[ Your Data Table ]
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 1. Train 5 Helper Models (Bootstrap Rounds)            │
│    Each helper learns from a random sample of rows.    │
└────────────────────────────────────────────────────────┘
        │
        ├───► Tower 1: Checks Accuracy (how many right?)
        │
        └───► Tower 2: Checks Risk (how steady and safe?)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 2. The Arbiter (Vote Counter)                          │
│    Gives higher vote share to the best helpers.        │
│    Combines all helpers into one super-fast brain.     │
└────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ 3. Temperature Calibration (Honest Confidence)         │
│    Tuning the confidence dial so percentages match     │
│    real-world accuracy.                                │
└────────────────────────────────────────────────────────┘
        │
        ▼
[ Final Predictions: Category + Honest Probabilities ]
```

---

## 3. Flagship Enterprise Features

TowerGB includes 8 powerful features found in top-tier machine learning libraries:

1. **Feature Importances (`clf.feature_importances_`)**:
   - Shows a score for each column from 0% to 100% telling you which columns were most important in making predictions.
2. **Missing Value Handling (`handle_missing=True`)**:
   - Fills missing cells (NaNs) with column averages and adds flag columns showing where data was missing.
3. **Polynomial Interactions (`interaction_degree=2`)**:
   - Creates new cross-product features (like $x_1 \times x_2$) so the model can capture non-linear patterns.
4. **Epistemic Uncertainty (`clf.predict_uncertainty(X)`)**:
   - Measures how much the 5 helper models disagree on each row. Higher score = model is less certain!
5. **Dual Calibration (`method="temperature"` or `"vector"`)**:
   - Global temperature dial or per-class Platt scaling to ensure predicted chances match real-world accuracy.
6. **Class Balancing (`class_weight='balanced'`)**:
   - Automatically gives higher importance to rare categories (like finding credit card fraud).
7. **L2 Regularization (`l2_reg=1e-4`)**:
   - Keeps model weights small and tidy to avoid memorizing noise.
8. **Parallel Training (`n_jobs=-1`)**:
   - Uses all available CPU cores to train the helper rounds simultaneously.

---

## 4. Tour of the Files (How the Code is Written)

The project is organized into clean, focused files. Here is what each file does:

### 📁 `src/towergb/_engine.py` (The Math Engine)

This file contains the fast math helpers:

1. **`stable_softmax(Z, temperature)`**:
   - Takes raw scores (like `[2.0, 5.0, 1.0]`) and turns them into percentages that add up to 100% (like `[4%, 92%, 4%]`).
   - It subtracts the biggest number first so numbers never get too big and crash the computer.

2. **`cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature, l2_reg, sample_weight)`**:
   - **Loss**: Measures how big the mistake was on average. If the true answer was "Dog" but the model guessed "Cat", the loss is high.
   - **Gradient**: Tells the model which direction to turn its weights to make fewer mistakes on the next try.
   - **Regularization & Weights**: Supports L2 weight decay and per-sample weights.

3. **`compute_risk_metrics(P, Y_one_hot, sample_weight)`**:
   - Calculates **Log-Loss** (mistake penalty).
   - Calculates **Brier Score** (distance from the true answer).
   - Calculates **Loss Variance** (checks if the model is steady across all rows or wildly guessing on some).

4. **`pareto_arbiter(metrics_list, weights)`**:
   - Looks at the score card for each of the training rounds.
   - Gives each round a grade: `Grade = Accuracy - Mistakes - Risk`.
   - Converts those grades into voting percentages (weights) that add up to 1.

---

### 📁 `src/towergb/calibration.py` (The Honesty Tuner)

When an AI model says "I am 99% sure," it is often overconfident and wrong. This file fixes that!

1. **`compute_ece(y_true, y_prob, n_bins)`**:
   - **ECE** stands for *Expected Calibration Error*.
   - It groups predictions into buckets (for example, all guesses between 70% and 80% confidence).
   - It checks: *Was the model actually right 75% of the time in that bucket?*
   - If not, ECE is high. If yes, ECE is close to 0 (perfect honesty).

2. **`optimize_temperature(logits, y_indices, n_classes)`**:
   - Searches for the best **Temperature ($T$)** number between `0.1` and `10.0`.
   - Uses **Golden Section Search** (a fast method that narrows down the search window like playing the high-low guessing game).

3. **`vector_scale_calibration(logits, y_indices, n_classes)`**:
   - Tunes individual scale and shift dials ($a_k \cdot z_k + b_k$) for every class via gradient descent.

---

### 📁 `src/towergb/estimator.py` (The Main Classifier)

This is the main class you interact with: `TowerGBClassifier`.

It follows the official **scikit-learn** style so anyone familiar with Python machine learning can use it immediately.

Key methods and attributes inside:
- **`__init__(...)`**: Sets your settings (number of passes, learning rate, regularization, interaction degree, class weights, jobs).
- **`.fit(X, y, sample_weight=None)`**: Preprocesses features, trains the ensemble passes, computes Arbiter weights, collapses into one matrix, and calculates `feature_importances_`.
- **`.predict_proba(X)`**: Returns percentage chances for every class.
- **`.predict(X)`**: Picks the category with the highest percentage chance.
- **`.predict_uncertainty(X)`**: Measures how much ensemble passes disagree on predictions.
- **`.calibrate(X_val, y_val, method="temperature")`**: Tunes confidence temperature or vector scaling on validation data.
- **`.score(X, y)`**: Returns the accuracy score.

---

## 5. How to Use It (Hands-On Code)

Here is how you can train and use TowerGB with feature importances:

```python
from sklearn.datasets import make_classification
from towergb import TowerGBClassifier

# 1. Create simple example data (100 rows, 4 features, 2 classes)
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# 2. Create the model with L2 regularization, missing value handling, and balanced classes
clf = TowerGBClassifier(n_passes=5, learning_rate=0.1, l2_reg=1e-4, handle_missing=True, random_state=42)

# 3. Train on data
clf.fit(X, y)

# 4. Inspect feature importance
print("Feature importances:", clf.feature_importances_)

# 5. Predict categories for new rows
predictions = clf.predict(X[:5])
print("Guesses:", predictions)

# 6. Predict probabilities (chances)
chances = clf.predict_proba(X[:5])
print("Chances:", chances)

# 7. Check model disagreement (uncertainty)
print("Uncertainty:", clf.predict_uncertainty(X[:5]))

# 8. Check overall accuracy
print("Accuracy:", clf.score(X, y))
```

---

## 6. Commands You Can Run in the Terminal

### Run all tests:
```bash
.venv\Scripts\pytest tests/ -v
```
*(All 69 tests will pass with green checkmarks!)*

### Run the benchmark race:
```bash
.venv\Scripts\python benchmarks/run_benchmarks.py
```

### Check code style and types:
```bash
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
```

---

## Quick Summary Cheat Sheet

| Component | What it is | Why it matters |
|---|---|---|
| **Tower 1** | Accuracy Checker | Finds models that make the most correct guesses. |
| **Tower 2** | Risk & Calibration Checker | Penalizes models that make wild or unsteady guesses. |
| **Arbiter** | The Voting Referee | Combines all models by giving the best ones more voting power. |
| **Calibration** | Confidence Tuning | Makes sure percentage chances match real accuracy. |
| **Feature Importance** | Interpretability | Tells you which columns drove the model's decisions. |
| **Class Balancing** | Fairness on Rare Events | Handles imbalanced datasets like fraud detection. |
| **Single Matrix Inference** | Fast Math Trick | Blends all models into one single math step for sub-millisecond speed. |
