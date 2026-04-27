# evaluate.py — Model evaluation for Heart Disease prediction
# Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
# Output: ROC curves + comparison table

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from heart.preprocessing import preprocess

MODELS_DIR = "heart/models"
PLOTS_DIR  = "heart/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "Random Forest":       "random_forest.pkl",
    "SVM":                 "svm.pkl",
    "XGBoost":             "xgboost.pkl",
}

# ── 1. Preprocess data ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Data Preprocessing")
print("=" * 60)
X_train, X_test, y_train, y_test = preprocess(verbose=False)

# ── 2. Load models and compute metrics ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Model Evaluation on Test Set")
print("=" * 60)

results = {}

for name, filename in MODEL_FILES.items():
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        print(f"\n  [WARNING] Model file not found: {model_path}")
        print(f"  Run train.py before evaluate.py")
        continue

    model  = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
        "F1-score":  f1,
        "ROC-AUC":   auc,
        "y_prob":    y_prob,
    }

    print(f"\n  {'='*40}")
    print(f"  Model: {name}")
    print(f"  {'='*40}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-score:  {f1:.4f}")
    print(f"    ROC-AUC:   {auc:.4f}")

# ── 3. ROC curves ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: ROC Curves")
print("=" * 60)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, ax = plt.subplots(figsize=(9, 7))

for (name, metrics), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
    auc_val = metrics["ROC-AUC"]
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

ax.set_title("ROC Curves — Heart Disease Models", fontsize=13)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()

roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"  [OK] ROC curves saved: {roc_path}")

# ── 4. Summary comparison table ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Comparison Table")
print("=" * 60)

summary_data = {
    name: {k: v for k, v in metrics.items() if k != "y_prob"}
    for name, metrics in results.items()
}
summary_df = pd.DataFrame(summary_data).T.round(4)
summary_df.index.name = "Model"

print(f"\n{summary_df.to_string()}")

# ── 5. Best models ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Best Models")
print("=" * 60)

best_f1_name  = max(results, key=lambda n: results[n]["F1-score"])
best_auc_name = max(results, key=lambda n: results[n]["ROC-AUC"])

print(f"\n  Best by F1-score:  {best_f1_name}")
print(f"    F1-score = {results[best_f1_name]['F1-score']:.4f}")

print(f"\n  Best by ROC-AUC:   {best_auc_name}")
print(f"    ROC-AUC  = {results[best_auc_name]['ROC-AUC']:.4f}")

print("\n[Evaluation complete]")
