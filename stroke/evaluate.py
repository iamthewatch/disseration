# evaluate.py — Model evaluation for Stroke prediction
# Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
# Output: ROC curves + comparison table

import os
import sys
import joblib
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
from preprocessing import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_DISPLAY_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'random_forest':       'Random Forest',
    'svm':                 'SVM',
    'xgboost':             'XGBoost',
}


def evaluate_all():
    print("=" * 60)
    print("DATA PREPROCESSING (reproducing test split)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_and_preprocess()

    results = {}

    print("\n" + "=" * 60)
    print("MODEL METRICS ON TEST SET")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    for model_key, display_name in MODEL_DISPLAY_NAMES.items():
        model_path = os.path.join(MODELS_DIR, f'{model_key}.pkl')

        if not os.path.exists(model_path):
            print(f"\n[WARNING] File not found: {model_path}")
            print("  Run train.py first")
            continue

        model = joblib.load(model_path)

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        roc_auc   = roc_auc_score(y_test, y_pred_prob)

        results[display_name] = {
            'Accuracy':  acc,
            'Precision': precision,
            'Recall':    recall,
            'F1-score':  f1,
            'ROC-AUC':   roc_auc,
        }

        print(f"\n[{display_name}]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        ax.plot(fpr, tpr, label=f'{display_name} (AUC={roc_auc:.3f})')

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves — Stroke Models')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"\n[Saved] {roc_path}")

    if not results:
        print("\nNo results to display. Run train.py first.")
        return

    results_df = pd.DataFrame(results).T.sort_values('F1-score', ascending=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

    best_f1     = results_df['F1-score'].idxmax()
    best_rocauc = results_df['ROC-AUC'].idxmax()

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print(f"  Best by F1-score : {best_f1}"
          f" ({results_df.loc[best_f1, 'F1-score']:.4f})")
    print(f"  Best by ROC-AUC  : {best_rocauc}"
          f" ({results_df.loc[best_rocauc, 'ROC-AUC']:.4f})")


if __name__ == '__main__':
    evaluate_all()
