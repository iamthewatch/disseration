# ============================================================
# preprocessing.py — Предобработка данных
# Включает: обработку пропусков, кодирование, SMOTE,
#           масштабирование и разбивку на train/test
# ============================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Путь к данным и папке для сохранения моделей
DATA_PATH = "data/heart.csv"
MODELS_DIR = "heart/models"

os.makedirs(MODELS_DIR, exist_ok=True)


def preprocess(verbose: bool = True):
    """
    Выполняет полный цикл предобработки датасета heart.csv.

    Возвращает:
        X_train, X_test, y_train, y_test (numpy arrays)
    """

    # -------------------------------------------------------
    # 1. Загрузка данных
    # -------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    if verbose:
        print("=" * 50)
        print(f"Загружено строк: {df.shape[0]}, столбцов: {df.shape[1]}")

    # -------------------------------------------------------
    # 2. Проверка и обработка пропущенных значений
    # -------------------------------------------------------
    missing = df.isnull().sum()
    if missing.any():
        if verbose:
            print("\nПропущенные значения найдены:")
            print(missing[missing > 0])
        # Числовые столбцы — заполняем медианой
        for col in df.select_dtypes(include="number").columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        # Категориальные — заполняем модой
        for col in df.select_dtypes(include="object").columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        if verbose:
            print("  [OK] Пропуски устранены (медиана / мода)")
    else:
        if verbose:
            print("\n[OK] Пропущенных значений не обнаружено")

    # -------------------------------------------------------
    # 3. Кодирование категориальных признаков (LabelEncoder)
    # -------------------------------------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    # Удаляем целевую переменную из списка на случай, если она попала в object
    categorical_cols = [c for c in categorical_cols if c != "target"]

    if categorical_cols:
        if verbose:
            print(f"\nКатегориальные признаки для кодирования: {categorical_cols}")
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        if verbose:
            print("  [OK] LabelEncoder применён")
    else:
        if verbose:
            print("\n[OK] Категориальных признаков для кодирования нет")

    # -------------------------------------------------------
    # 4. Разделение на признаки (X) и целевую переменную (y)
    # -------------------------------------------------------
    TARGET_COL = "target"
    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    feature_names = df.drop(columns=[TARGET_COL]).columns.tolist()

    if verbose:
        print(f"\nПризнаки ({len(feature_names)}): {feature_names}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Распределение классов: {dict(zip(unique, counts))}")

    # -------------------------------------------------------
    # 5. Обработка дисбаланса классов с помощью SMOTE
    # -------------------------------------------------------
    unique, counts = np.unique(y, return_counts=True)
    imbalance_ratio = counts.max() / counts.min()

    if imbalance_ratio > 1.5:
        if verbose:
            print(f"\nДисбаланс классов обнаружен (соотношение {imbalance_ratio:.2f}). Применяем SMOTE...")
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        unique_new, counts_new = np.unique(y, return_counts=True)
        if verbose:
            print(f"  [OK] После SMOTE: {dict(zip(unique_new, counts_new))}")
    else:
        if verbose:
            print(f"\n[OK] Дисбаланс классов в норме (соотношение {imbalance_ratio:.2f}). SMOTE не нужен")

    # -------------------------------------------------------
    # 6. Разбивка на обучающую и тестовую выборки (80/20)
    # -------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if verbose:
        print(f"\nРазбивка train/test (80/20):")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # -------------------------------------------------------
    # 7. Масштабирование признаков (StandardScaler)
    # -------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # обучаем на train, применяем к train
    X_test = scaler.transform(X_test)         # применяем к test (без переобучения)

    # Сохраняем scaler для последующего использования в evaluate.py
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    if verbose:
        print(f"\n[OK] StandardScaler применён")
        print(f"[OK] Scaler сохранён: {scaler_path}")
        print("\n[Предобработка завершена]")

    return X_train, X_test, y_train, y_test


# Запуск напрямую: python heart/preprocessing.py
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess(verbose=True)
