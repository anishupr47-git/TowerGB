import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from towergb import TowerGBClassifier

# 1. Create employee dataset
data = {
    "Name": [
        "Aarav Sharma", "Bianca Vance", "Carlos Mendez", "Divya Patel",
        "Ethan Walker", "Fatima Al-Mansoor", "George Chen", "Hannah Abbott",
        "Ian Kowalski", "Julia Santos", "Kevin O'Connor", "Layla Hassan",
    ],
    "Salary": [
        45000, 92000, 61000, 38000,
        115000, 74000, 52000, 49000,
        83000, 102000, 41000, 68000,
    ],
    "Experience (Years)": [
        2, 7, 4, 1,
        10, 5, 3, 2,
        6, 8, 1, 4,
    ],
    "Performance Score (1-10)": [
        6, 9, 7, 5,
        8, 8, 4, 7,
        6, 9, 4, 8,
    ],
    "Promoted": [
        "No", "Yes", "Yes", "No",
        "Yes", "Yes", "No", "No",
        "Yes", "Yes", "No", "Yes",
    ],
}

df = pd.DataFrame(data)

# 2. Features and target
feature_cols = ["Salary", "Experience (Years)", "Performance Score (1-10)"]
X = df[feature_cols]
y = df["Promoted"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train TowerGB model
clf = TowerGBClassifier(
    n_passes=5,
    learning_rate=0.1,
    l2_reg=1e-4,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train_scaled, y_train)

# 6. Helper function to predict promotion and project new earnings
def evaluate_employee(name: str, salary: float, experience: int, performance: int) -> dict:
    """Predict promotion chance and project expected new salary earnings."""
    emp_df = pd.DataFrame({
        "Salary": [salary],
        "Experience (Years)": [experience],
        "Performance Score (1-10)": [performance],
    })
    emp_scaled = scaler.transform(emp_df)

    pred = clf.predict(emp_scaled)[0]
    proba = clf.predict_proba(emp_scaled)[0]
    yes_idx = list(clf.classes_).index("Yes")
    promotion_chance = float(proba[yes_idx])

    # Calculate raise percentage based on promotion & performance
    if pred == "Yes":
        # Promotion raise: base 15% + up to 10% extra based on high performance
        raise_percent = 0.15 + (performance / 10.0) * 0.10
    else:
        # Standard annual adjustment: 3% to 5% based on performance
        raise_percent = 0.03 + (performance / 10.0) * 0.02

    projected_salary = salary * (1.0 + raise_percent)
    raise_amount = projected_salary - salary

    return {
        "name": name,
        "current_salary": salary,
        "experience": experience,
        "performance": performance,
        "promoted": pred,
        "promotion_chance": promotion_chance * 100.0,
        "raise_percent": raise_percent * 100.0,
        "raise_amount": raise_amount,
        "projected_salary": projected_salary,
    }


print("=" * 65)
print("   TowerGB Employee Promotion & Earnings Predictor")
print("=" * 65)

print(f"\n[+] Trained on {len(X_train)} employees | Test Accuracy: {clf.score(X_test_scaled, y_test) * 100:.1f}%")

print("\n[+] Feature Importances:")
for col, imp in zip(feature_cols, clf.feature_importances_):
    print(f"    - {col:<26}: {imp * 100:.1f}%")

# 7. Evaluate sample candidates
candidates = [
    ("Anish (You)", 85000, 5, 9),
    ("Sarah Jenkins", 55000, 3, 6),
    ("David Miller", 110000, 9, 8),
]

print("\n" + "=" * 65)
print("   PROMOTION & PROJECTED SALARY REPORT")
print("=" * 65)

for name, sal, exp, perf in candidates:
    res = evaluate_employee(name, sal, exp, perf)
    status_icon = "[PROMOTED]" if res["promoted"] == "Yes" else "[NOT PROMOTED]"

    print(f"\nCandidate: {res['name']}")
    print(f"  - Current Salary   : ${res['current_salary']:,.2f}")
    print(f"  - Experience       : {res['experience']} years | Performance: {res['performance']}/10")
    print(f"  - Promotion Status : {status_icon} ({res['promoted']})")
    print(f"  - Promotion Chance : {res['promotion_chance']:.1f}%")
    print(f"  - Projected Raise  : +{res['raise_percent']:.1f}% (+${res['raise_amount']:,.2f})")
    print(f"  - NEW EARNINGS     : ${res['projected_salary']:,.2f} / year")

print("\n" + "=" * 65)
print("Tip: Add any name, salary, and experience to candidates in sample.py!")
print("=" * 65 + "\n")
