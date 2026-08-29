#!/usr/bin/env python
"""Compare TowerGB speed and accuracy against other models."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from towergb import TowerGBClassifier
from towergb.calibration import compute_ece

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _measure_latency(model: Any, X: np.ndarray, n_runs: int = 100) -> float:
    """Measure mean inference latency in milliseconds."""
    model.predict_proba(X)
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict_proba(X)
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0


def run_benchmarks() -> None:
    """Train each model and print comparison results."""
    print("=" * 78)
    print(" TowerGB Benchmark Suite")
    print("=" * 78)

    print("\n[1/3] Generating dataset...")
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=15, n_classes=5,
        n_clusters_per_class=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"    Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"    Test:  {X_test.shape[0]} samples, 5 classes")

    print("\n[2/3] Training models...")
    models = {
        "TowerGB": TowerGBClassifier(n_passes=5, max_iter=200, learning_rate=0.05, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    if _HAS_XGB:
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=100, eval_metric="mlogloss", random_state=42, verbosity=0,
        )
    else:
        print("    [!] xgboost not installed -- skipping XGBClassifier")

    results = {}
    for name, model in models.items():
        print(f"    Training {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        print(f"({train_time:.2f}s)")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_proba)
        ece = compute_ece(y_test, y_proba)
        latency_ms = _measure_latency(model, X_test)

        results[name] = {
            "Accuracy": acc,
            "Log-Loss": ll,
            "ECE": ece,
            "Latency (ms)": latency_ms,
        }

    print("\n[3/3] Results")
    print("-" * 78)
    header = f"{'Model':<25} {'Accuracy':>10} {'Log-Loss':>10} {'ECE':>10} {'Latency (ms)':>14}"
    print(header)
    print("-" * 78)

    for name, metrics in results.items():
        row = (
            f"{name:<25} "
            f"{metrics['Accuracy']:>10.4f} "
            f"{metrics['Log-Loss']:>10.4f} "
            f"{metrics['ECE']:>10.4f} "
            f"{metrics['Latency (ms)']:>14.2f}"
        )
        print(row)

    print("-" * 78)
    print("\n[OK] Benchmark complete. Higher Accuracy is better. Lower Log-Loss/ECE/Latency is better.")


if __name__ == "__main__":
    run_benchmarks()
