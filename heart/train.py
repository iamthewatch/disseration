# ============================================================
# train.py — Обучение моделей машинного обучения
# Модели: Logistic Regression, Decision Tree, Random Forest,
#         SVM, XGBoost
# ============================================================

import os
import sys
import joblib
import numpy as np

# Добавляем корень проекта в sys.path, чтобы импортировать preprocessing.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from heart.preprocessing import preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

MODELS_DIR = "heart/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------
# 1. Получение предобработанных данных
# -------------------------------------------------------
print("=" * 55)
print("ЭТАП 1: Предобработка данных")
print("=" * 55)
X_train, X_test, y_train, y_test = preprocess(verbose=True)

# -------------------------------------------------------
# 2. Определение моделей для обучения
# -------------------------------------------------------
# Все модели используют random_state=42 там, где это поддерживается
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "SVM": SVC(
        probability=True,   # необходимо для ROC-AUC
        random_state=42,
        kernel="rbf",
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    ),
}

# -------------------------------------------------------
# 3. Обучение моделей и сохранение результатов
# -------------------------------------------------------
print("\n" + "=" * 55)
print("ЭТАП 2: Обучение моделей")
print("=" * 55)

train_results = {}

for name, model in models.items():
    print(f"\n  Обучение: {name}...")
    model.fit(X_train, y_train)

    # Точность на обучающей выборке
    train_acc = model.score(X_train, y_train)
    train_results[name] = train_acc
    print(f"    Точность на train: {train_acc:.4f}")

    # Формируем безопасное имя файла для сохранения модели
    safe_name = name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
    joblib.dump(model, model_path)
    print(f"    [OK] Модель сохранена: {model_path}")

# -------------------------------------------------------
# 4. Итоговая таблица точности на train
# -------------------------------------------------------
print("\n" + "=" * 55)
print("ИТОГИ ОБУЧЕНИЯ — Точность на обучающей выборке:")
print("=" * 55)
print(f"  {'Модель':<30} {'Train Accuracy':>15}")
print("  " + "-" * 46)
for name, acc in train_results.items():
    print(f"  {name:<30} {acc:>14.4f}")

print("\n[Обучение завершено] Модели сохранены в папке:", MODELS_DIR)
