# evaluate.py — Оценка обученных моделей для предсказания диабета

import os
import sys
import joblib
import numpy as np
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

# Добавляем папку diabetes в sys.path для импорта preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess

# Пути
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Имена всех сохранённых моделей
MODEL_NAMES = [
    "LogisticRegression",
    "DecisionTree",
    "RandomForest",
    "SVM",
    "XGBoost",
]


def load_models() -> dict:
    """Загружает все сохранённые модели из папки models/."""
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Модель не найдена: {path}\n"
                "Запустите сначала train.py для обучения моделей."
            )
        models[name] = joblib.load(path)
        print(f"  Загружена: {name}")
    return models


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """
    Вычисляет метрики для одной модели:
    Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{'='*40}")
    print(f"Модель: {name}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-score:  {metrics['F1-score']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")

    return metrics


def plot_roc_curves(models: dict, X_test, y_test) -> None:
    """
    Строит ROC-кривые для всех моделей на одном графике.
    Сохраняет в diabetes/plots/roc_curves.png.
    """
    plt.figure(figsize=(9, 7))

    # Цвета для кривых
    colors = ["steelblue", "tomato", "green", "purple", "orange"]

    for (name, model), color in zip(models.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc:.3f})")

    # Линия случайного классификатора
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Случайный классификатор")

    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC-кривые всех моделей", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Сохранено] {path}")


def print_summary_table(results: list) -> None:
    """Выводит итоговую таблицу сравнения всех моделей."""
    df = pd.DataFrame(results).set_index("Model")
    # Форматируем до 4 знаков
    df_fmt = df.map(lambda x: f"{x:.4f}")

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
    print("=" * 60)
    print(df_fmt.to_string())


def print_best_models(results: list) -> None:
    """Выводит лучшие модели по F1-score и ROC-AUC."""
    df = pd.DataFrame(results)

    best_f1_row = df.loc[df["F1-score"].idxmax()]
    best_auc_row = df.loc[df["ROC-AUC"].idxmax()]

    print("\n" + "=" * 60)
    print("ЛУЧШИЕ МОДЕЛИ")
    print("=" * 60)
    print(
        f"Лучшая по F1-score:  {best_f1_row['Model']:<22} "
        f"F1 = {best_f1_row['F1-score']:.4f}"
    )
    print(
        f"Лучшая по ROC-AUC:   {best_auc_row['Model']:<22} "
        f"AUC = {best_auc_row['ROC-AUC']:.4f}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("ОЦЕНКА МОДЕЛЕЙ")
    print("=" * 60)

    # Загружаем предобработанные данные (нужна только тестовая выборка)
    _, X_test, _, y_test = preprocess()

    # Загружаем все обученные модели
    print("\nЗагрузка моделей:")
    models = load_models()

    # Оцениваем каждую модель и собираем метрики
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)

    # Строим ROC-кривые
    plot_roc_curves(models, X_test, y_test)

    # Итоговая таблица
    print_summary_table(results)

    # Лучшие модели
    print_best_models(results)
