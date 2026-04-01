# preprocessing.py — Предобработка данных для предсказания диабета

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Пути к файлам
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Целевая переменная
TARGET = "Outcome"

# Признаки, где 0 биологически невозможен (заменяем на NaN)
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(path: str) -> pd.DataFrame:
    """Загружает датасет из CSV файла."""
    df = pd.read_csv(path)
    print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")
    return df


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заменяет биологически невозможные нулевые значения на NaN
    в столбцах Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    """
    df = df.copy()
    for col in ZERO_AS_NAN_COLS:
        zero_count = (df[col] == 0).sum()
        df[col] = df[col].replace(0, np.nan)
        print(f"  {col}: заменено {zero_count} нулей на NaN")
    return df


def fill_nan_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет NaN медианным значением каждого столбца."""
    df = df.copy()
    for col in ZERO_AS_NAN_COLS:
        median_val = df[col].median()
        nan_count = df[col].isnull().sum()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: заполнено {nan_count} NaN медианой ({median_val:.2f})")
    return df


def preprocess(path: str = DATA_PATH):
    """
    Полный пайплайн предобработки данных.

    Шаги:
    1. Загрузка данных
    2. Замена 0 → NaN в биологически невозможных столбцах
    3. Заполнение NaN медианой
    4. Разделение на признаки X и целевую переменную y
    5. Стратифицированное разделение на train/test (80/20)
    6. Применение SMOTE для балансировки классов на обучающей выборке
    7. Масштабирование признаков с помощью StandardScaler
    8. Сохранение скейлера

    Возвращает: X_train, X_test, y_train, y_test
    """
    print("=" * 60)
    print("ПРЕДОБРАБОТКА ДАННЫХ")
    print("=" * 60)

    # 1. Загрузка
    df = load_data(path)

    # 2. Замена нулей на NaN
    print("\nЗамена биологически невозможных нулей на NaN:")
    df = replace_zeros_with_nan(df)

    # 3. Заполнение медианой
    print("\nЗаполнение NaN медианным значением:")
    df = fill_nan_with_median(df)

    # 4. Разделение на X и y
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"\nПризнаки: {list(X.columns)}")
    print(f"Распределение классов до SMOTE: {dict(y.value_counts())}")

    # 5. Стратифицированное разделение train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nРазмер обучающей выборки: {X_train.shape[0]} строк")
    print(f"Размер тестовой выборки:   {X_test.shape[0]} строк")

    # 6. SMOTE — балансировка классов только на train
    print("\nПрименение SMOTE для балансировки классов...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Распределение классов после SMOTE: {dict(pd.Series(y_train_resampled).value_counts())}")
    print(f"Размер обучающей выборки после SMOTE: {X_train_resampled.shape[0]} строк")

    # 7. Масштабирование признаков (fit только на train, transform на train и test)
    print("\nПрименение StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # 8. Сохранение скейлера
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Скейлер сохранён: {scaler_path}")

    print("\nПредобработка завершена.")
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()
    print(f"\nФормы массивов:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
