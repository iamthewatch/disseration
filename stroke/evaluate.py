# ==============================================================
# Оценка моделей (Evaluation)
# - Загрузка всех 5 сохранённых моделей и скейлера
# - Метрики: Accuracy, Precision, Recall, F1, ROC-AUC
# - ROC-кривые для всех моделей на одном графике
# - Сводная таблица сравнения
# - Вывод лучшей модели по F1 и ROC-AUC
# ==============================================================

import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# Добавляем текущую папку в путь, чтобы импортировать preprocessing.py
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Читаемые названия моделей для графиков и таблиц
MODEL_DISPLAY_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'random_forest':       'Random Forest',
    'svm':                 'SVM',
    'xgboost':             'XGBoost',
}


def evaluate_all():
    """Оценивает все обученные модели и выводит сводную таблицу."""

    # ── Воспроизводим те же предобработанные данные ───────────
    print("=" * 60)
    print("ПРЕДОБРАБОТКА ДАННЫХ (воспроизведение тестовой выборки)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_and_preprocess()

    results = {}

    print("\n" + "=" * 60)
    print("МЕТРИКИ МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 60)

    # Для ROC-кривых собираем данные по всем моделям
    fig, ax = plt.subplots(figsize=(8, 6))
    # Диагональ случайного классификатора
    ax.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')

    for model_key, display_name in MODEL_DISPLAY_NAMES.items():
        model_path = os.path.join(MODELS_DIR, f'{model_key}.pkl')

        # Проверяем наличие файла модели
        if not os.path.exists(model_path):
            print(f"\n[ПРЕДУПРЕЖДЕНИЕ] Файл не найден: {model_path}")
            print("  Сначала запустите train.py")
            continue

        # ── Загрузка модели ───────────────────────────────────
        model = joblib.load(model_path)

        # ── Предсказания ──────────────────────────────────────
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # ── Вычисление метрик ─────────────────────────────────
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

        # ── Вывод метрик ──────────────────────────────────────
        print(f"\n[{display_name}]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  ROC-AUC   : {roc_auc:.4f}")

        # ── ROC-кривая ────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        ax.plot(fpr, tpr, label=f'{display_name} (AUC={roc_auc:.3f})')

    # ── Сохранение графика ROC-кривых ─────────────────────────
    ax.set_xlabel('False Positive Rate (1 - Специфичность)')
    ax.set_ylabel('True Positive Rate (Чувствительность)')
    ax.set_title('ROC-кривые всех моделей')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(PLOTS_DIR, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"\n[Сохранено] {roc_path}")

    if not results:
        print("\nНет результатов для отображения. Запустите train.py.")
        return

    # ── Сводная таблица ───────────────────────────────────────
    results_df = pd.DataFrame(results).T.sort_values('F1-score', ascending=False)

    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 60)
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Лучшие модели ─────────────────────────────────────────
    best_f1     = results_df['F1-score'].idxmax()
    best_rocauc = results_df['ROC-AUC'].idxmax()

    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)
    print(f"  Лучшая модель по F1-score : {best_f1}"
          f" ({results_df.loc[best_f1, 'F1-score']:.4f})")
    print(f"  Лучшая модель по ROC-AUC  : {best_rocauc}"
          f" ({results_df.loc[best_rocauc, 'ROC-AUC']:.4f})")


if __name__ == '__main__':
    evaluate_all()
