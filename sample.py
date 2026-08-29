import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from towergb import TowerGBClassifier

# 1. Create employee promotion dataset
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

# 2. Pick numerical features to learn from and target category
feature_cols = ["Salary", "Experience (Years)", "Performance Score (1-10)"]
X = df[feature_cols]
y = df["Promoted"]

# 3. Split into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Standardize numerical features so all columns are on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("  TowerGB Employee Promotion Prediction")
print("=" * 60)
print(f"\nTraining on {len(X_train)} employees, testing on {len(X_test)} employees.\n")

# 5. Create and train TowerGB model
clf = TowerGBClassifier(
    n_passes=5,
    learning_rate=0.1,
    l2_reg=1e-4,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train_scaled, y_train)

# 6. Check test accuracy
accuracy = clf.score(X_test_scaled, y_test)
print(f"[+] Test Accuracy: {accuracy * 100:.1f}%\n")

# 7. Check feature importances (which column mattered most)
print("[+] Feature Importances:")
for col, imp in zip(feature_cols, clf.feature_importances_):
    print(f"    - {col:<26}: {imp * 100:.1f}%")

# 8. Make predictions on test employees
print("\n[+] Test Set Predictions:")
preds = clf.predict(X_test_scaled)
probas = clf.predict_proba(X_test_scaled)

test_names = df.loc[X_test.index, "Name"].values
for name, actual, pred, proba in zip(test_names, y_test, preds, probas):
    yes_idx = list(clf.classes_).index("Yes")
    yes_chance = proba[yes_idx] * 100
    print(
        f"    - {name:<18} | Actual: {actual:<3} | Predicted: {pred:<3} "
        f"| Promotion Chance: {yes_chance:.1f}%"
    )

# 9. Predict on a new unseen employee
print("\n" + "=" * 60)
print("  Predicting on New Unseen Employee")
print("=" * 60)

new_employee = pd.DataFrame({
    "Salary": [88000],
    "Experience (Years)": [6],
    "Performance Score (1-10)": [9],
})

new_employee_scaled = scaler.transform(new_employee)
pred_new = clf.predict(new_employee_scaled)[0]
proba_new = clf.predict_proba(new_employee_scaled)[0]
yes_idx = list(clf.classes_).index("Yes")

print(f"\nProfile: Salary=$88,000, Experience=6 yrs, Performance=9/10")
print(f"Prediction: {pred_new}")
print(f"Promotion Chance: {proba_new[yes_idx] * 100:.1f}%\n")
