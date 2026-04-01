# ==============================================================
# EDA (Exploratory Data Analysis) — Разведочный анализ данных
# Датасет: healthcare-dataset-stroke-data.csv
# ==============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к данным и папка для сохранения графиков
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare-dataset-stroke-data.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Загрузка датасета ──────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("БАЗОВАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
print("=" * 60)
print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов\n")

print("Типы данных:")
print(df.dtypes)
print()

print("Пропущенные значения по столбцам:")
missing = df.isnull().sum()
print(missing[missing > 0])
print(f"\nВсего пропущенных значений: {df.isnull().sum().sum()}")
print()

print("Статистика по числовым признакам:")
print(df.describe())
print()

print("Распределение целевой переменной (stroke):")
print(df['stroke'].value_counts())
print(f"Доля инсультов: {df['stroke'].mean():.2%}")

# ── 2. График распределения классов ──────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['stroke'].value_counts()
ax.bar(
    ['Нет инсульта (0)', 'Инсульт (1)'],
    counts.values,
    color=['steelblue', 'tomato'],
    edgecolor='black'
)
for i, v in enumerate(counts.values):
    ax.text(i, v + 30, str(v), ha='center', fontweight='bold')
ax.set_title('Распределение классов: инсульт / нет инсульта')
ax.set_ylabel('Количество пациентов')
ax.set_xlabel('Класс')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'class_distribution.png'), dpi=150)
plt.close()
print("\n[Сохранено] class_distribution.png")

# ── 3. Гистограммы числовых признаков ─────────────────────────
numeric_cols = df.select_dtypes(include='number').columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ('id', 'stroke')]

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
for ax, col in zip(axes, numeric_cols):
    ax.hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title(f'Гистограмма: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Частота')
plt.suptitle('Гистограммы числовых признаков', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Сохранено] histograms.png")

# ── 4. Тепловая карта корреляций ──────────────────────────────
numeric_df = df[numeric_cols + ['stroke']].copy()
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    ax=ax,
    linewidths=0.5
)
ax.set_title('Матрица корреляций числовых признаков')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("[Сохранено] correlation_heatmap.png")

# ── 5. Ящики с усами (boxplot) по группам инсульт / нет ───────
box_cols = ['age', 'avg_glucose_level', 'bmi']

fig, axes = plt.subplots(1, len(box_cols), figsize=(5 * len(box_cols), 5))
for ax, col in zip(axes, box_cols):
    # Группируем данные по значению stroke
    groups = [
        df.loc[df['stroke'] == 0, col].dropna(),
        df.loc[df['stroke'] == 1, col].dropna(),
    ]
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=['Нет инсульта', 'Инсульт'])
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('tomato')
    ax.set_title(f'Boxplot: {col}')
    ax.set_ylabel(col)
plt.suptitle('Распределение признаков по группам (инсульт / нет)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("[Сохранено] boxplots.png")

print("\nEDA завершён. Все графики сохранены в папку stroke/plots/")
