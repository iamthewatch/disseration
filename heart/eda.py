# ============================================================
# eda.py — Разведочный анализ данных (EDA)
# Набор данных: heart.csv (предсказание болезней сердца)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к файлу данных (относительно корня проекта)
DATA_PATH = "data/heart.csv"
PLOTS_DIR = "heart/plots"

# Создаём папку для сохранения графиков, если её нет
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------------------------------------
# 1. Загрузка датасета и базовая информация
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("ФОРМА ДАТАСЕТА:")
print(f"  Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

print("\nТИПЫ ДАННЫХ:")
print(df.dtypes)

print("\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  Пропущенных значений не обнаружено")

print("\nОСНОВНАЯ СТАТИСТИКА:")
print(df.describe())

# -------------------------------------------------------
# 2. Распределение целевого класса (болезнь / нет болезни)
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
counts = df["target"].value_counts().sort_index()
bars = ax.bar(
    ["Нет болезни (0)", "Болезнь (1)"],
    counts.values,
    color=["#4C72B0", "#DD8452"],
    edgecolor="black",
)
# Подписываем количество над каждым столбцом
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontweight="bold")

ax.set_title("Распределение целевого класса", fontsize=14)
ax.set_ylabel("Количество пациентов")
ax.set_xlabel("Класс (target)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=150)
plt.close()
print(f"\n[OK] Сохранён график: {PLOTS_DIR}/class_distribution.png")

# -------------------------------------------------------
# 3. Гистограммы для всех числовых признаков
# -------------------------------------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()

n_cols = 4
n_rows = -(-len(numeric_cols) // n_cols)  # потолочное деление
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel("Значение")
    axes[i].set_ylabel("Частота")

# Скрываем лишние оси
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Гистограммы числовых признаков", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "histograms.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Сохранён график: {PLOTS_DIR}/histograms.png")

# -------------------------------------------------------
# 4. Тепловая карта корреляций
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = df.corr()
mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
# Маскируем верхний треугольник для читаемости
import numpy as np
mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix,
    mask=mask_upper,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Тепловая карта корреляций признаков", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print(f"[OK] Сохранён график: {PLOTS_DIR}/correlation_heatmap.png")

# -------------------------------------------------------
# 5. Ящичные диаграммы (boxplots) по группам target
# -------------------------------------------------------
boxplot_features = ["age", "thalach", "chol", "trestbps"]
fig, axes = plt.subplots(1, len(boxplot_features), figsize=(16, 6))

for ax, feature in zip(axes, boxplot_features):
    # Разбиваем данные по значению target
    groups = [
        df.loc[df["target"] == 0, feature].dropna(),
        df.loc[df["target"] == 1, feature].dropna(),
    ]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    # Раскрашиваем ящики
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(["Нет болезни (0)", "Болезнь (1)"])
    ax.set_title(f"{feature}", fontsize=12)
    ax.set_ylabel("Значение")

plt.suptitle("Boxplots признаков по целевому классу", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "boxplots.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"[OK] Сохранён график: {PLOTS_DIR}/boxplots.png")

print("\n[EDA завершён] Все графики сохранены в папке:", PLOTS_DIR)
