from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is in Python path even if run with global python
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from towergb import TowerGBClassifier

# 1. Create employee training dataset
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


def evaluate_employee(name: str, salary: float, experience: float, performance: float) -> dict:
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

    if pred == "Yes":
        raise_percent = 0.15 + (performance / 10.0) * 0.10
    else:
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


def print_result(res: dict) -> None:
    status_icon = "[PROMOTED]" if res["promoted"] == "Yes" else "[NOT PROMOTED]"
    print("\n" + "-" * 55)
    print(f"  PROMOTION & SALARY REPORT FOR: {res['name'].upper()}")
    print("-" * 55)
    print(f"  - Current Salary   : ${res['current_salary']:,.2f}")
    print(f"  - Experience       : {res['experience']} years | Performance: {res['performance']}/10")
    print(f"  - Promotion Status : {status_icon} ({res['promoted']})")
    print(f"  - Promotion Chance : {res['promotion_chance']:.1f}%")
    print(f"  - Projected Raise  : +{res['raise_percent']:.1f}% (+${res['raise_amount']:,.2f})")
    print(f"  - NEW EARNINGS     : ${res['projected_salary']:,.2f} / year")
    print("-" * 55 + "\n")


def interactive_session() -> None:
    print("=" * 65)
    print("   TowerGB Interactive Promotion & Earnings Predictor")
    print("=" * 65)
    print(f"[+] Model trained on {len(X_train)} rows | Accuracy: {clf.score(X_test_scaled, y_test) * 100:.1f}%\n")

    while True:
        try:
            name_input = input("Enter Name (or press Enter for 'Anish'): ").strip()
            name = name_input if name_input else "Anish"

            sal_input = input("Enter Current Salary (e.g. 75000): ").strip().replace("$", "").replace(",", "")
            salary = float(sal_input) if sal_input else 75000.0

            exp_input = input("Enter Years of Experience (e.g. 5): ").strip()
            experience = float(exp_input) if exp_input else 5.0

            perf_input = input("Enter Performance Score 1-10 (e.g. 8): ").strip()
            performance = float(perf_input) if perf_input else 8.0

            res = evaluate_employee(name, salary, experience, performance)
            print_result(res)

            again = input("Would you like to test another profile? (y/n): ").strip().lower()
            if again not in ("y", "yes"):
                print("\nThank you for using TowerGB! Goodbye.\n")
                break
        except (ValueError, KeyboardInterrupt):
            print("\nExiting session. Goodbye!")
            break


if __name__ == "__main__":
    interactive_session()
