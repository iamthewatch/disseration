# train.py — Обучение моделей машинного обучения для предсказания диабета

import os
import sys
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Добавляем папку diabetes в sys.path для импорта preprocessing
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess

# Папка для сохранения моделей
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def get_models() -> dict:
    """
    Возвращает словарь с моделями для обучения.
    random_state=42 установлен везде, где поддерживается.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "SVM": SVC(
            probability=True,   # нужно для predict_proba при оценке ROC-AUC
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def train_and_save(models: dict, X_train, y_train, X_test, y_test) -> None:
    """
    Обучает каждую модель, выводит точность на обучающей выборке
    и сохраняет модель в файл .pkl.
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nОбучение: {name}")
        model.fit(X_train, y_train)

        # Точность на обучающей выборке
        train_acc = model.score(X_train, y_train)
        print(f"  Точность на train: {train_acc:.4f}")

        # Сохранение модели
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"  Сохранена: {model_path}")

    print("\nВсе модели обучены и сохранены.")


if __name__ == "__main__":
    # Получаем предобработанные данные
    X_train, X_test, y_train, y_test = preprocess()

    # Инициализируем и обучаем модели
    models = get_models()
    train_and_save(models, X_train, y_train, X_test, y_test)
