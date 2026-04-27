# evaluate.py — Model evaluation for Diabetes prediction

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAMES = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM",
    "XGBoost",
]

MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "DecisionTree":       "Decision Tree",
    "RandomForest":       "Random Forest",
    "SVM":                "SVM",
    "XGBoost":            "XGBoost",
}


def load_models() -> dict:
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run train.py first to train the models."
            )
        models[name] = joblib.load(path)
        print(f"  Loaded: {name}")
    return models


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model":     MODEL_DISPLAY.get(name, name),
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1-score":  f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*40}")
    print(f"Model: {metrics['Model']}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-score:  {metrics['F1-score']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    return metrics


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    plt.figure(figsize=(9, 7))

    colors = ["steelblue", "tomato", "green", "purple", "orange"]

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        label = MODEL_DISPLAY.get(name, name)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")

    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Curves — All Models (Diabetes)", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] {path}")


def print_summary_table(results: list) -> None:
    df = pd.DataFrame(results).set_index("Model")
    df_fmt = df.map(lambda x: f"{x:.4f}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df_fmt.to_string())


def print_best_models(results: list) -> None:
    df = pd.DataFrame(results)

    best_f1_row  = df.loc[df["F1-score"].idxmax()]
    best_auc_row = df.loc[df["ROC-AUC"].idxmax()]

    print("\n" + "=" * 60)
    print("BEST MODELS")
    print("=" * 60)
    print(f"Best by F1-score:  {best_f1_row['Model']:<22} F1 = {best_f1_row['F1-score']:.4f}")
    print(f"Best by ROC-AUC:   {best_auc_row['Model']:<22} AUC = {best_auc_row['ROC-AUC']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL EVALUATION — DIABETES")
    print("=" * 60)

    _, X_test, _, y_test = preprocess()

    print("\nLoading models:")
    models = load_models()

    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    plot_roc_curves(models, X_test, y_test)
    print_summary_table(results)
    print_best_models(results)
