# ============================================================
# evaluate.py — Оценка качества обученных моделей
# Метрики: Accuracy, Precision, Recall, F1, ROC-AUC
# Дополнительно: ROC-кривые и сводная таблица сравнения
# ============================================================

import os
import sys
import joblib
import numpy as np
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

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from heart.preprocessing import preprocess

MODELS_DIR = "heart/models"
PLOTS_DIR = "heart/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Имена моделей и соответствующие имена файлов pkl
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "Random Forest":       "random_forest.pkl",
    "SVM":                 "svm.pkl",
    "XGBoost":             "xgboost.pkl",
}

# -------------------------------------------------------
# 1. Предобработка данных (получаем тестовую выборку)
# -------------------------------------------------------
print("=" * 60)
print("ЭТАП 1: Предобработка данных")
print("=" * 60)
X_train, X_test, y_train, y_test = preprocess(verbose=False)

# -------------------------------------------------------
# 2. Загрузка моделей и вычисление метрик
# -------------------------------------------------------
print("\n" + "=" * 60)
print("ЭТАП 2: Оценка моделей на тестовой выборке")
print("=" * 60)

results = {}  # имя модели -> dict с метриками

for name, filename in MODEL_FILES.items():
    model_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(model_path):
        print(f"\n  [ПРЕДУПРЕЖДЕНИЕ] Файл модели не найден: {model_path}")
        print(f"  Запустите train.py перед evaluate.py")
        continue

    # Загружаем обученную модель
    model = joblib.load(model_path)

    # Предсказания классов и вероятностей
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # вероятность класса 1

    # Вычисляем метрики
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
        "y_prob":    y_prob,   # нужно для ROC-кривой
    }

    # Выводим метрики для каждой модели
    print(f"\n  {'='*40}")
    print(f"  Модель: {name}")
    print(f"  {'='*40}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-score:  {f1:.4f}")
    print(f"    ROC-AUC:   {auc:.4f}")

# -------------------------------------------------------
# 3. ROC-кривые для всех моделей на одном графике
# -------------------------------------------------------
print("\n" + "=" * 60)
print("ЭТАП 3: Построение ROC-кривых")
print("=" * 60)

# Цвета для каждой модели
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig, ax = plt.subplots(figsize=(9, 7))

for (name, metrics), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, metrics["y_prob"])
    auc_val = metrics["ROC-AUC"]
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=color, linewidth=2)

# Диагональная линия случайного классификатора
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Случайный классификатор")

ax.set_title("ROC-кривые моделей предсказания болезней сердца", fontsize=13)
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()

roc_path = os.path.join(PLOTS_DIR, "roc_curves.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"  [OK] ROC-кривые сохранены: {roc_path}")

# -------------------------------------------------------
# 4. Итоговая сводная таблица сравнения моделей
# -------------------------------------------------------
print("\n" + "=" * 60)
print("ЭТАП 4: Сводная таблица сравнения моделей")
print("=" * 60)

# Формируем DataFrame без столбца y_prob
summary_data = {
    name: {k: v for k, v in metrics.items() if k != "y_prob"}
    for name, metrics in results.items()
}
summary_df = pd.DataFrame(summary_data).T.round(4)
summary_df.index.name = "Модель"

print(f"\n{summary_df.to_string()}")

# -------------------------------------------------------
# 5. Лучшая модель по F1-score и ROC-AUC
# -------------------------------------------------------
print("\n" + "=" * 60)
print("ЭТАП 5: Лучшие модели")
print("=" * 60)

best_f1_name  = max(results, key=lambda n: results[n]["F1-score"])
best_auc_name = max(results, key=lambda n: results[n]["ROC-AUC"])

print(f"\n  Лучшая модель по F1-score:  {best_f1_name}")
print(f"    F1-score = {results[best_f1_name]['F1-score']:.4f}")

print(f"\n  Лучшая модель по ROC-AUC:   {best_auc_name}")
print(f"    ROC-AUC  = {results[best_auc_name]['ROC-AUC']:.4f}")

print("\n[Оценка завершена]")
