# ==============================================================
# Предобработка данных (Preprocessing)
# - Удаление id
# - Заполнение пропусков (bmi → медиана)
# - LabelEncoder для категориальных признаков
# - SMOTE для балансировки классов
# - StandardScaler для числовых признаков
# - Разбивка на train/test (80/20, stratify, random_state=42)
# - Сохранение scaler в stroke/models/scaler.pkl
# ==============================================================

import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Пути
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare-dataset-stroke-data.csv')
MODELS_DIR  = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_preprocess():
    """
    Загружает датасет, выполняет всю предобработку и возвращает
    X_train, X_test, y_train, y_test (уже масштабированные).
    """

    # ── 1. Загрузка ──────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"Загружено строк: {len(df)}")

    # ── 2. Удаление технического столбца id ──────────────────
    df.drop(columns=['id'], inplace=True)

    # ── 3. Удаление строк с неизвестным полом (редкое значение)
    df = df[df['gender'] != 'Other'].reset_index(drop=True)

    # ── 4. Заполнение пропусков в bmi медианным значением ────
    bmi_median = df['bmi'].median()
    df['bmi'] = df['bmi'].fillna(bmi_median)
    print(f"Пропуски bmi заполнены медианой: {bmi_median:.2f}")

    # ── 4b. Удаление оставшихся строк с пропусками ───────────
    df = df.dropna()
    print(f"После удаления пропусков: {len(df)} строк")

    # ── 5. Кодирование категориальных признаков ───────────────
    categorical_cols = [
        'gender',
        'ever_married',
        'work_type',
        'Residence_type',
        'smoking_status',
    ]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    print("Категориальные признаки закодированы (LabelEncoder).")

    # ── 6. Разделение на признаки и целевую переменную ────────
    X = df.drop(columns=['stroke'])
    y = df['stroke']

    # ── 7. Балансировка классов с помощью SMOTE ───────────────
    # SMOTE синтетически создаёт примеры минорного класса (инсульт)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"После SMOTE: {y_resampled.value_counts().to_dict()}")

    # ── 8. Разбивка на обучение / тест (80/20) ────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled,
        y_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_resampled,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── 9. Масштабирование числовых признаков ─────────────────
    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # ── 10. Сохранение скейлера ───────────────────────────────
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Скейлер сохранён: {scaler_path}")

    return X_train, X_test, y_train, y_test


# Позволяет запустить preprocessing.py напрямую для проверки
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("\nПредобработка завершена.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
