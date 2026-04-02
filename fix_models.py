"""
fix_models.py — Переобучение XGBoost и пересохранение scaler для всех трёх заболеваний.

Запуск из корня проекта:
    python fix_models.py

Что делает скрипт:
  - Загружает и предобрабатывает данные (SMOTE, StandardScaler, LabelEncoder, медиана)
  - Обучает XGBClassifier для каждого заболевания
  - Сохраняет scaler.pkl и xgboost.pkl через joblib (protocol=2) в папки models/
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательная функция сохранения
# ─────────────────────────────────────────────────────────────────────────────

def save(obj, path: str) -> None:
    """Сохраняет объект через joblib с protocol=2 для максимальной совместимости."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path, protocol=2)
    print(f"  Сохранено: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. ИНСУЛЬТ
# ═════════════════════════════════════════════════════════════════════════════

def run_stroke() -> None:
    """
    Предобработка датасета инсульта и обучение XGBoost.
    Признаки (порядок как в app.py):
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        gender, ever_married, work_type, Residence_type, smoking_status
    """
    print("\n" + "=" * 60)
    print("ИНСУЛЬТ")
    print("=" * 60)

    # ── Загрузка ─────────────────────────────────────────────────
    df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")
    print(f"Загружено строк: {len(df)}")

    # ── Удаление id ──────────────────────────────────────────────
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # ── Удаление строк с gender='Other' (слишком мало примеров) ──
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    # ── Заполнение пропусков bmi медианой ────────────────────────
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df = df.dropna()
    print(f"После очистки: {len(df)} строк")

    # ── LabelEncoder для категориальных признаков ─────────────────
    # Результат кодирования (алфавитный порядок sklearn):
    #   gender:         Female=0, Male=1
    #   ever_married:   No=0, Yes=1
    #   work_type:      Govt_job=0, Never_worked=1, Private=2, Self-employed=3, children=4
    #   Residence_type: Rural=0, Urban=1
    #   smoking_status: Unknown=3, formerly smoked=0, never smoked=1, smokes=2
    cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print("Категориальные признаки закодированы (LabelEncoder).")

    # ── Разделение на X / y ───────────────────────────────────────
    # Порядок столбцов соответствует порядку ввода в app.py
    feature_cols = [
        "age", "hypertension", "heart_disease",
        "avg_glucose_level", "bmi",
        "gender", "ever_married", "work_type",
        "Residence_type", "smoking_status",
    ]
    X = df[feature_cols].values
    y = df["stroke"].values

    # ── SMOTE ─────────────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print(f"После SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── StandardScaler (fit на всех признаках, как ожидает app.py) ─
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Обучение XGBoost ──────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Точность XGBoost на test: {acc:.4f}")

    # ── Сохранение ────────────────────────────────────────────────
    save(scaler, "stroke/models/scaler.pkl")
    save(model,  "stroke/models/xgboost.pkl")
    print("Done: stroke")


# ═════════════════════════════════════════════════════════════════════════════
# 2. БОЛЕЗНИ СЕРДЦА
# ═════════════════════════════════════════════════════════════════════════════

def run_heart() -> None:
    """
    Предобработка датасета болезней сердца и обучение XGBoost.
    Признаки (порядок как в app.py):
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    """
    print("\n" + "=" * 60)
    print("БОЛЕЗНИ СЕРДЦА")
    print("=" * 60)

    # ── Загрузка ─────────────────────────────────────────────────
    df = pd.read_csv("data/heart.csv")
    print(f"Загружено строк: {len(df)}, столбцов: {len(df.columns)}")

    # ── Обработка пропусков ───────────────────────────────────────
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── LabelEncoder для строковых столбцов ───────────────────────
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "target"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Разделение на X / y ───────────────────────────────────────
    TARGET = "target"
    X = df.drop(columns=[TARGET]).values
    y = df[TARGET].values
    print(f"Признаки: {list(df.drop(columns=[TARGET]).columns)}")

    # ── SMOTE (если дисбаланс > 1.5) ─────────────────────────────
    counts = np.bincount(y)
    ratio  = counts.max() / counts.min()
    if ratio > 1.5:
        smote = SMOTE(random_state=42)
        X, y  = smote.fit_resample(X, y)
        print(f"После SMOTE: {dict(zip(*np.unique(y, return_counts=True)))}")
    else:
        print(f"SMOTE не нужен (соотношение классов {ratio:.2f})")

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── StandardScaler ────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Обучение XGBoost ──────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Точность XGBoost на test: {acc:.4f}")

    # ── Сохранение ────────────────────────────────────────────────
    save(scaler, "heart/models/scaler.pkl")
    save(model,  "heart/models/xgboost.pkl")
    print("Done: heart")


# ═════════════════════════════════════════════════════════════════════════════
# 3. ДИАБЕТ
# ═════════════════════════════════════════════════════════════════════════════

def run_diabetes() -> None:
    """
    Предобработка датасета диабета и обучение XGBoost.
    Признаки (порядок как в app.py):
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    """
    print("\n" + "=" * 60)
    print("ДИАБЕТ")
    print("=" * 60)

    # ── Загрузка ─────────────────────────────────────────────────
    df = pd.read_csv("data/diabetes.csv")
    print(f"Загружено строк: {len(df)}")

    # ── Замена биологически невозможных нулей на NaN ──────────────
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_nan:
        df[col] = df[col].replace(0, np.nan)

    # ── Заполнение NaN медианой ───────────────────────────────────
    for col in zero_as_nan:
        df[col] = df[col].fillna(df[col].median())
    print("Нулевые значения заменены на медиану.")

    # ── Разделение на X / y ───────────────────────────────────────
    TARGET = "Outcome"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Признаки: {list(X.columns)}")
    print(f"Распределение классов: {dict(y.value_counts())}")

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── SMOTE (только на train) ───────────────────────────────────
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"После SMOTE: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # ── StandardScaler ────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Обучение XGBoost ──────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Точность XGBoost на test: {acc:.4f}")

    # ── Сохранение ────────────────────────────────────────────────
    save(scaler, "diabetes/models/scaler.pkl")
    save(model,  "diabetes/models/xgboost.pkl")
    print("Done: diabetes")


# ═════════════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_stroke()
    run_heart()
    run_diabetes()
    print("\n" + "=" * 60)
    print("Все модели успешно пересохранены.")
    print("=" * 60)
