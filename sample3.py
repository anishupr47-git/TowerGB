import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Add src to path for TowerGB
sys.path.insert(0, str(Path(__file__).parent / "src"))
from towergb import TowerGBClassifier
from towergb.calibration import compute_ece

# 1. Generate Realistic Financial Fraud Dataset (5,000 rows)
print("=== Generating Synthetic Financial Fraud Dataset (5,000 rows) ===")
rng = np.random.RandomState(42)
N = 5000

amount = rng.exponential(scale=150.0, size=N)
device_trust = rng.uniform(0.0, 1.0, size=N)
failed_attempts = rng.poisson(lam=0.5, size=N)
account_age_days = rng.randint(1, 1000, size=N)
transaction_hour = rng.randint(0, 24, size=N)

# Introduce 5% missing NaNs into features
amount[rng.choice(N, size=int(0.05 * N), replace=False)] = np.nan
device_trust[rng.choice(N, size=int(0.05 * N), replace=False)] = np.nan

# Non-linear Fraud Risk score (High Amount + Low Device Trust + Night Hours)
risk_score = (
    0.002 * np.nan_to_num(amount, nan=150.0)
    - 3.5 * np.nan_to_num(device_trust, nan=0.5)
    + 0.8 * failed_attempts
    - 0.003 * account_age_days
    + np.where(transaction_hour < 5, 1.5, 0.0)
    + rng.normal(0, 1.0, size=N)
)

# Top ~10% highest risk transactions marked as Fraud (1), rest Legitimate (0)
threshold = np.percentile(risk_score, 90)
y = (risk_score >= threshold).astype(int)

df = pd.DataFrame({
    "Amount": amount,
    "Device_Trust_Score": device_trust,
    "Failed_Attempts_24h": failed_attempts,
    "Account_Age_Days": account_age_days,
    "Transaction_Hour": transaction_hour
})

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)} samples | Test size: {len(X_test)} samples")
print(f"Fraud distribution: {y_train.sum()} Fraud ({y_train.mean()*100:.1f}%), {len(y_train)-y_train.sum()} Legitimate\n")

# For baseline models (Logistic Regression & XGBoost), impute NaNs
X_train_imp = X_train.fillna(X_train.median())
X_test_imp = X_test.fillna(X_train.median())

# 2. Define Models
models = {
    "TowerGB (Linear, degree=1)": TowerGBClassifier(
        n_passes=5,
        learning_rate=0.1,
        handle_missing=True,
        class_weight="balanced",
        random_state=42
    ),
    "TowerGB (Quadratic, degree=2)": TowerGBClassifier(
        n_passes=5,
        learning_rate=0.1,
        interaction_degree=2,
        handle_missing=True,
        class_weight="balanced",
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        eval_metric="logloss",
        random_state=42
    )
}

# Measure Latency Helper
def measure_latency(model, X_data, n_runs=100):
    model.predict_proba(X_data)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict_proba(X_data)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0

# 3. Train & Evaluate
results = []
for name, clf in models.items():
    # Use un-imputed for TowerGB, imputed for others
    X_tr = X_train if "TowerGB" in name else X_train_imp
    X_te = X_test if "TowerGB" in name else X_test_imp
    
    t0 = time.perf_counter()
    clf.fit(X_tr, y_train)
    train_time = time.perf_counter() - t0
    
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)
    
    acc = accuracy_score(y_test, preds)
    loss = log_loss(y_test, probs)
    ece = compute_ece(y_test, probs)
    latency = measure_latency(clf, X_te)
    
    results.append({
        "Model": name,
        "Accuracy": f"{acc*100:.2f}%",
        "Log-Loss": f"{loss:.4f}",
        "ECE": f"{ece:.4f}",
        "Latency (ms)": f"{latency:.2f} ms",
        "Train Time": f"{train_time:.3f} s"
    })

# 4. Print Benchmark Comparison Table
results_df = pd.DataFrame(results)
print("=== Model Comparison Results (Financial Fraud Dataset) ===")
print(results_df.to_string(index=False))

# 5. Check TowerGB Uncertainty Estimation
tgb = models["TowerGB (Quadratic, degree=2)"]
uncertainty = tgb.predict_uncertainty(X_test)
print("\n=== TowerGB Epistemic Uncertainty Feature ===")
print(f"Mean Uncertainty across Test Set: {uncertainty.mean():.6f}")
print(f"Max Uncertainty on Confusing Cases: {uncertainty.max():.6f}")
