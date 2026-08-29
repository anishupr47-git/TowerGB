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

## 3. Tour of the Files (How the Code is Written)

The project is organized into clean, focused files. Here is what each file does:

### 📁 `src/towergb/_engine.py` (The Math Engine)

This file contains the fast math helpers:

1. **`stable_softmax(Z, temperature)`**:
   - Takes raw scores (like `[2.0, 5.0, 1.0]`) and turns them into percentages that add up to 100% (like `[4%, 92%, 4%]`).
   - It subtracts the biggest number first so numbers never get too big and crash the computer.

2. **`cross_entropy_loss_and_grad(X, Y_one_hot, W, temperature)`**:
   - **Loss**: Measures how big the mistake was on average. If the true answer was "Dog" but the model guessed "Cat", the loss is high.
   - **Gradient**: Tells the model which direction to turn its weights to make fewer mistakes on the next try.

3. **`compute_risk_metrics(P, Y_one_hot)`**:
   - Calculates **Log-Loss** (mistake penalty).
   - Calculates **Brier Score** (distance from the true answer).
   - Calculates **Loss Variance** (checks if the model is steady across all rows or wildly guessing on some).

4. **`pareto_arbiter(metrics_list, weights)`**:
   - Looks at the score card for each of the 5 training rounds.
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
   - A higher temperature softens overconfident guesses.
   - A lower temperature sharpens guesses.
   - Uses **Golden Section Search** (a super fast method that narrows down the search window like playing the high-low guessing game).

---

### 📁 `src/towergb/estimator.py` (The Main Classifier)

This is the main class you interact with: `TowerGBClassifier`.

It follows the official **scikit-learn** style so anyone familiar with Python machine learning can use it immediately.

Key methods inside:
- **`__init__(...)`**: Sets your settings (number of passes, learning rate, max iterations).
- **`.fit(X, y)`**:
  1. Validates table data (handles numbers, text labels, and pandas DataFrames).
  2. Runs 5 training passes with random data samples.
  3. Records accuracy from Tower 1 and risk from Tower 2.
  4. Runs the Arbiter to calculate the best combined weights.
- **`.predict_proba(X)`**: Multiplies your input by the combined weights and returns percentage chances for every class.
- **`.predict(X)`**: Picks the category with the highest percentage chance.
- **`.calibrate(X_val, y_val)`**: Takes a test set and finds the best temperature so probabilities are honest.
- **`.score(X, y)`**: Returns the accuracy score (for example, `0.85` means 85% correct).

---

### 📁 `tests/` (The Safety Testers)

We wrote 3 test files to make sure the code never breaks:

1. **`test_engine.py`**:
   - Tests that softmax doesn't crash even if numbers are huge ($10,000+$).
   - Tests that chances always add up to 100%.
   - Tests that training reduces mistakes over time.
   - Tests that the best model gets the biggest vote in the Arbiter.

2. **`test_gradcheck.py`**:
   - Checks our math by comparing the fast formula against manual tiny step-by-step slope calculations (central finite differences).
   - Makes sure the difference is smaller than $0.0000001$.

3. **`test_estimator.py`**:
   - Tests scikit-learn compatibility (`check_estimator`).
   - Tests 2-class and 5-class problems.
   - Tests text labels like `"cat"`, `"dog"`, `"bird"`.
   - Tests pandas DataFrames.
   - Tests saving and loading with `pickle` and `joblib`.
   - Tests that temperature calibration improves confidence.

---

### 📁 `benchmarks/run_benchmarks.py` (The Speed & Accuracy Race)

This script creates a table of 5,000 rows and 20 columns, then races TowerGB against:
- **LogisticRegression**
- **RandomForest**
- **XGBoost**

It measures 4 things:
1. **Accuracy**: How many guesses were right?
2. **Log-Loss**: How small were the mistakes?
3. **ECE**: How honest was the confidence score?
4. **Latency (ms)**: How fast did it predict 1,000 rows (in milliseconds)?

---

## 4. How to Use It (Hands-On Code)

Here is how you can train and use TowerGB in just 6 lines of code:

```python
from sklearn.datasets import make_classification
from towergb import TowerGBClassifier

# 1. Create simple example data (100 rows, 4 features, 2 classes)
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

# 2. Create the model
clf = TowerGBClassifier(n_passes=5, learning_rate=0.1, random_state=42)

# 3. Train on data
clf.fit(X, y)

# 4. Predict categories for new rows
predictions = clf.predict(X[:5])
print("Guesses:", predictions)

# 5. Predict probabilities (chances)
chances = clf.predict_proba(X[:5])
print("Chances:", chances)

# 6. Check overall accuracy
print("Accuracy:", clf.score(X, y))
```

---

## 5. Commands You Can Run in the Terminal

### Run all tests:
```bash
.venv\Scripts\pytest tests/ -v
```
*(All 39 tests will pass with green checkmarks!)*

### Run the benchmark race:
```bash
.venv\Scripts\python benchmarks/run_benchmarks.py
```

### Check code style and types:
```bash
.venv\Scripts\ruff check .
.venv\Scripts\mypy src
```
*(Both will pass with 0 errors!)*

---

## Quick Summary Cheat Sheet

| Component | What it is | Why it matters |
|---|---|---|
| **Tower 1** | Accuracy Checker | Finds models that make the most correct guesses. |
| **Tower 2** | Risk & Confidence Checker | Penalizes models that make wild or unsteady guesses. |
| **Arbiter** | The Voting Referee | Combines all models by giving the best ones more voting power. |
| **Calibration** | Confidence Tuning | Makes sure the percentage chances match reality. |
| **Single Matrix Inference** | Fast Math Trick | Blends all models into one single math step for sub-millisecond speed. |
