# ==============================================================
# Обучение моделей (Training)
# Модели:
#   1. Logistic Regression
#   2. Decision Tree
#   3. Random Forest
#   4. SVM (Support Vector Machine)
#   5. XGBoost
# Каждая модель сохраняется в stroke/models/<name>.pkl
# ==============================================================

import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Добавляем текущую папку в путь, чтобы импортировать preprocessing.py
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_preprocess

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models():
    """Возвращает словарь {имя: экземпляр модели}."""
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
        ),
        'decision_tree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
        ),
        'svm': SVC(
            probability=True,   # нужно для ROC-AUC
            random_state=42,
            class_weight='balanced',
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
        ),
    }


def train_all():
    """Загружает данные, обучает все модели и сохраняет их на диск."""

    # ── Получение предобработанных данных ─────────────────────
    print("=" * 60)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_and_preprocess()

    models = get_models()

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n[{name.upper()}]")

        # Обучение модели
        model.fit(X_train, y_train)

        # Точность на обучающей выборке
        train_acc = model.score(X_train, y_train)
        print(f"  Точность на train: {train_acc:.4f}")

        # Сохранение модели
        model_path = os.path.join(MODELS_DIR, f'{name}.pkl')
        joblib.dump(model, model_path)
        print(f"  Сохранено: {model_path}")

    print("\n" + "=" * 60)
    print("Все модели обучены и сохранены в stroke/models/")
    print("=" * 60)


if __name__ == '__main__':
    train_all()
