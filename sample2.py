import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

# Add src to path for TowerGB
sys.path.insert(0, str(Path(__file__).parent / "src"))
from towergb import TowerGBClassifier

# Raw Training Dataset
train_data = [
    {"Transaction_ID": "TX_1001", "User_ID": "USR_9921", "Amount": "14.50", "Transaction_Type": "POS", "Location_Country": "US", "Device_Trust_Score": "0.98", "Failed_Attempts_24h": "0", "Is_Fraud": "Legitimate"},
    {"Transaction_ID": "TX_1002", "User_ID": "USR_3104", "Amount": "4999.00", "Transaction_Type": "Wire_Transfer", "Location_Country": "RU", "Device_Trust_Score": "0.12", "Failed_Attempts_24h": "4", "Is_Fraud": "Fraud"},
    {"Transaction_ID": "TX_1003", "User_ID": "USR_8820", "Amount": "NaN", "Transaction_Type": "Online_Checkout", "Location_Country": "US", "Device_Trust_Score": "0.85", "Failed_Attempts_24h": "0", "Is_Fraud": "Legitimate"},
    {"Transaction_ID": "TX_1004", "User_ID": "USR_1192", "Amount": '"$12,500.00"', "Transaction_Type": "Crypto_Withdrawal", "Location_Country": "NG", "Device_Trust_Score": "null", "Failed_Attempts_24h": "6", "Is_Fraud": "Fraud"},
    {"Transaction_ID": "TX_1005", "User_ID": "USR_5541", "Amount": "-25.00", "Transaction_Type": "ATM_Withdrawal", "Location_Country": "CA", "Device_Trust_Score": "0.91", "Failed_Attempts_24h": "0", "Is_Fraud": "Legitimate"},
    {"Transaction_ID": "TX_1006", "User_ID": "missing", "Amount": "850.00", "Transaction_Type": "Online_Checkout", "Location_Country": "UK", "Device_Trust_Score": "0.44", "Failed_Attempts_24h": "1", "Is_Fraud": "Legitimate"},
    {"Transaction_ID": "TX_1007", "User_ID": "USR_4402", "Amount": "9800.00", "Transaction_Type": "WIRE_TRANSFER", "Location_Country": "PA", "Device_Trust_Score": "0.05", "Failed_Attempts_24h": "8", "Is_Fraud": "Fraud"},
    {"Transaction_ID": "TX_1008", "User_ID": "USR_7731", "Amount": "42.10", "Transaction_Type": "POS", "Location_Country": "empty", "Device_Trust_Score": "0.95", "Failed_Attempts_24h": "0", "Is_Fraud": "Legitimate"},
    {"Transaction_ID": "TX_1009", "User_ID": "USR_9910", "Amount": "0.00", "Transaction_Type": "Online_Checkout", "Location_Country": "US", "Device_Trust_Score": "0.78", "Failed_Attempts_24h": "NaN", "Is_Fraud": "Legitimate"},
    {"Transaction_ID": "TX_1010", "User_ID": "USR_2048", "Amount": "1500000.00", "Transaction_Type": "Wire_Transfer", "Location_Country": "CY", "Device_Trust_Score": "0.18", "Failed_Attempts_24h": "999", "Is_Fraud": "Fraud"},
    {"Transaction_ID": "TX_1011", "User_ID": "USR_6632", "Amount": "120.00", "Transaction_Type": "Unknown_Channel", "Location_Country": "DE", "Device_Trust_Score": "0.88", "Failed_Attempts_24h": "0", "Is_Fraud": "NaN"},
    {"Transaction_ID": "TX_1012", "User_ID": "USR_3391", "Amount": "310.45", "Transaction_Type": "POS", "Location_Country": "US", "Device_Trust_Score": "1.45", "Failed_Attempts_24h": "0", "Is_Fraud": "Legitimate"}
]

# Raw Test Dataset
test_data = [
    {"Transaction_ID": "TX_9001", "User_ID": "USR_4421", "Amount": "89.99", "Transaction_Type": "POS", "Location_Country": "US", "Device_Trust_Score": "0.94", "Failed_Attempts_24h": "0"},
    {"Transaction_ID": "TX_9002", "User_ID": "USR_8812", "Amount": "7450.00", "Transaction_Type": "Crypto_Withdrawal", "Location_Country": "null", "Device_Trust_Score": "0.08", "Failed_Attempts_24h": "5"},
    {"Transaction_ID": "TX_9003", "User_ID": "USR_1093", "Amount": "NaN", "Transaction_Type": "Online_Checkout", "Location_Country": "FR", "Device_Trust_Score": "0.81", "Failed_Attempts_24h": "1"},
    {"Transaction_ID": "TX_9004", "User_ID": "missing", "Amount": '"3,200 USD"', "Transaction_Type": "Wire_Transfer", "Location_Country": "SG", "Device_Trust_Score": "NaN", "Failed_Attempts_24h": "3"},
    {"Transaction_ID": "TX_9005", "User_ID": "USR_5509", "Amount": "18.20", "Transaction_Type": "POS", "Location_Country": "JP", "Device_Trust_Score": "0.99", "Failed_Attempts_24h": "0"},
    {"Transaction_ID": "TX_9006", "User_ID": "USR_7714", "Amount": "12400.00", "Transaction_Type": "P2P_Transfer", "Location_Country": "BZ", "Device_Trust_Score": "-0.20", "Failed_Attempts_24h": "7"}
]

df_train = pd.DataFrame(train_data)
df_test = pd.DataFrame(test_data)

# Drop missing target row from training set
df_train = df_train[df_train["Is_Fraud"].notna() & (df_train["Is_Fraud"] != "NaN")].copy()
target_map = {"Legitimate": 0, "Fraud": 1}
y = df_train["Is_Fraud"].map(target_map).values
df_train = df_train.drop(columns=["Is_Fraud"])

# Cleaning helper functions
def parse_amount(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "null", ""]:
        return np.nan
    clean_str = re.sub(r'[\$,USD"\s]', '', str(val))
    try:
        return abs(float(clean_str))
    except ValueError:
        return np.nan

def parse_score(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "null", ""]:
        return np.nan
    try:
        return float(np.clip(float(val), 0.0, 1.0))
    except ValueError:
        return np.nan

def parse_failed(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "null", ""]:
        return np.nan
    try:
        v = float(val)
        return np.nan if v >= 999 else v
    except ValueError:
        return np.nan

def parse_cat(val):
    if pd.isna(val) or str(val).strip().lower() in ["nan", "null", "empty", "missing", ""]:
        return "UNKNOWN"
    return str(val).strip().upper()

# Apply cleaning to train and test
for df in [df_train, df_test]:
    df["Amount"] = df["Amount"].apply(parse_amount)
    df["Device_Trust_Score"] = df["Device_Trust_Score"].apply(parse_score)
    df["Failed_Attempts_24h"] = df["Failed_Attempts_24h"].apply(parse_failed)
    
    # Fill numeric NaNs with train medians
    for col in ["Amount", "Device_Trust_Score", "Failed_Attempts_24h"]:
        median_val = df_train[col].median()
        df[col] = df[col].fillna(median_val)
        
    # Cap Amount outliers at 95th percentile of train dataset
    amount_cap = df_train["Amount"].quantile(0.95)
    df["Amount"] = np.clip(df["Amount"], 0, amount_cap)

    # Clean categorical columns
    for col in ["Transaction_ID", "User_ID", "Transaction_Type", "Location_Country"]:
        df[col] = df[col].apply(parse_cat)

# Ordinal encode categorical columns to numerical values
cat_cols = ["Transaction_ID", "User_ID", "Transaction_Type", "Location_Country"]
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols])
df_test[cat_cols] = encoder.transform(df_test[cat_cols])

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    df_train, y, test_size=0.3, random_state=42, stratify=y
)

# 1. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lr_preds_raw = lin_reg.predict(X_test)
lr_preds = (lr_preds_raw >= 0.5).astype(int)
lr_acc = accuracy_score(y_test, lr_preds)
lr_probs = np.clip(lr_preds_raw, 1e-15, 1 - 1e-15)
lr_loss = log_loss(y_test, lr_probs)

# 2. Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_acc = log_reg.score(X_test, y_test)
log_loss_val = log_loss(y_test, log_reg.predict_proba(X_test))

# 3. TowerGB Classifier
tgb = TowerGBClassifier(random_state=42)
tgb.fit(X_train, y_train)
tgb_acc = tgb.score(X_test, y_test)
tgb_loss = log_loss(y_test, tgb.predict_proba(X_test))

# 4. XGBoost Classifier
xgb = XGBClassifier(random_state=42, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_acc = xgb.score(X_test, y_test)
xgb_loss = log_loss(y_test, xgb.predict_proba(X_test))

# Print Accuracy and Loss of every model
print("=== Model Comparison Results ===")
print(f"Linear Regression   - Accuracy: {lr_acc * 100:.2f}%, Log Loss: {lr_loss:.4f}")
print(f"Logistic Regression - Accuracy: {log_acc * 100:.2f}%, Log Loss: {log_loss_val:.4f}")
print(f"TowerGB             - Accuracy: {tgb_acc * 100:.2f}%, Log Loss: {tgb_loss:.4f}")
print(f"XGBoost             - Accuracy: {xgb_acc * 100:.2f}%, Log Loss: {xgb_loss:.4f}")
