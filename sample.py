import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from towergb import TowerGBClassifier

# 1. Generate standard 10,000-row medical dataset
rng = np.random.RandomState(42)
N = 10000

age = rng.randint(20, 80, size=N)
bp = rng.randint(90, 180, size=N)
chol = rng.randint(150, 350, size=N)
hr = rng.randint(100, 200, size=N)
ecg = rng.choice([0, 1, 2], size=N)

# Standard risk score formula
risk_score = (
    0.04 * (age - 50)
    + 0.03 * (bp - 130)
    + 0.02 * (chol - 200)
    - 0.02 * (hr - 150)
    + rng.randn(N) * 2.0
)
y = (risk_score > 0).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Blood Pressure (Systolic)": bp,
    "Cholesterol (mg/dL)": chol,
    "Max Heart Rate": hr,
    "Resting ECG": ecg,
})
feature_cols = ["Age", "Blood Pressure (Systolic)", "Cholesterol (mg/dL)", "Max Heart Rate", "Resting ECG"]

# 2. Train / Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_cols], y, test_size=0.2, random_state=42
)

# 3. Train Model 1: TowerGB
clf_tgb = TowerGBClassifier(random_state=42)
clf_tgb.fit(X_train, y_train)

# 4. Train Model 2: Logistic Regression
clf_lr = LogisticRegression(max_iter=1000, random_state=42)
clf_lr.fit(X_train, y_train)

# 5. Train Model 3: XGBoost
clf_xgb = XGBClassifier(random_state=42)
clf_xgb.fit(X_train, y_train)

# 6. Evaluate and print Accuracy & Log Loss for all 3 models
print("=== Model Comparison on 2,000 Out-of-Sample Test Rows ===")

print("\n1. TowerGBClassifier:")
print(f"   - Test Accuracy: {clf_tgb.score(X_test, y_test) * 100:.2f}%")
print(f"   - Test Log Loss: {log_loss(y_test, clf_tgb.predict_proba(X_test)):.6f}")

print("\n2. Logistic Regression:")
print(f"   - Test Accuracy: {clf_lr.score(X_test, y_test) * 100:.2f}%")
print(f"   - Test Log Loss: {log_loss(y_test, clf_lr.predict_proba(X_test)):.6f}")

print("\n3. XGBoost Classifier:")
print(f"   - Test Accuracy: {clf_xgb.score(X_test, y_test) * 100:.2f}%")
print(f"   - Test Log Loss: {log_loss(y_test, clf_xgb.predict_proba(X_test)):.6f}")
