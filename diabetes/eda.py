# eda.py — Разведочный анализ данных (EDA) для датасета диабета

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Папка для сохранения графиков
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Путь к датасету
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")


def load_data(path: str) -> pd.DataFrame:
    """Загружает датасет и выводит базовую информацию."""
    df = pd.read_csv(path)

    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)
    print(f"Форма датасета (строки, столбцы): {df.shape}")
    print("\nТипы данных:")
    print(df.dtypes)
    print("\nПропущенные значения:")
    print(df.isnull().sum())
    print("\nПервые 5 строк:")
    print(df.head())
    print("\nСтатистика по числовым признакам:")
    print(df.describe())

    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Строит график распределения классов (диабет / не диабет)."""
    counts = df["Outcome"].value_counts()
    labels = ["Не диабет (0)", "Диабет (1)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Столбчатая диаграмма
    axes[0].bar(labels, counts.values, color=["steelblue", "tomato"], edgecolor="black")
    axes[0].set_title("Распределение классов", fontsize=14)
    axes[0].set_ylabel("Количество")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    # Круговая диаграмма
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=["steelblue", "tomato"],
        startangle=90,
    )
    axes[1].set_title("Доля классов", fontsize=14)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Сохранено] {path}")


def plot_feature_histograms(df: pd.DataFrame) -> None:
    """Строит гистограммы для всех числовых признаков."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=12)
        axes[i].set_xlabel("Значение")
        axes[i].set_ylabel("Частота")

    # Скрываем пустые оси
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Гистограммы числовых признаков", fontsize=16, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_histograms.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Сохранено] {path}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Строит тепловую карту корреляций между признаками."""
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    mask = corr.abs() < 0.0  # показываем всё
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Корреляционная матрица признаков", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Сохранено] {path}")


def plot_boxplots_by_outcome(df: pd.DataFrame) -> None:
    """Строит ящичные диаграммы для ключевых признаков, сгруппированных по Outcome."""
    features = ["Glucose", "BMI", "Age", "Insulin"]

    fig, axes = plt.subplots(1, len(features), figsize=(16, 6))

    for ax, feature in zip(axes, features):
        data_groups = [
            df[df["Outcome"] == 0][feature].dropna(),
            df[df["Outcome"] == 1][feature].dropna(),
        ]
        bp = ax.boxplot(
            data_groups,
            patch_artist=True,
            labels=["Не диабет", "Диабет"],
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("tomato")
        ax.set_title(feature, fontsize=12)
        ax.set_ylabel("Значение")

    plt.suptitle("Boxplot по ключевым признакам (группировка по Outcome)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "boxplots_by_outcome.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Сохранено] {path}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    plot_class_distribution(df)
    plot_feature_histograms(df)
    plot_correlation_heatmap(df)
    plot_boxplots_by_outcome(df)
    print("\nEDA завершён. Все графики сохранены в папку:", PLOTS_DIR)
