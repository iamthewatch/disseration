"""
Интеллектуальная система прогнозирования заболеваний
Streamlit-приложение — минималистичный дизайн в стиле Notion / Linear
"""

import streamlit as st
import numpy as np
import pickle
import os
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация страницы
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Глобальные стили CSS — перекрытие тёмной темы Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Сброс и базовые переменные ─────────────────────────── */
:root {
    --bg:        #FFFFFF;
    --surface:   #F7F7F5;
    --border:    #E5E5E3;
    --accent:    #2563EB;
    --accent-bg: #EFF6FF;
    --text:      #1A1A1A;
    --muted:     #6B7280;
    --success:   #16A34A;
    --warning:   #D97706;
    --danger:    #DC2626;
    --success-bg:#F0FDF4;
    --warning-bg:#FFFBEB;
    --danger-bg: #FEF2F2;
    --radius:    10px;
    --font:      system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── Общий фон ───────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
.main, .block-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── Убрать padding блока ────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* ── Боковая панель ──────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── Радио-кнопки навигации ──────────────────────────────── */
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    color: var(--muted) !important;
    padding: 6px 8px !important;
    border-radius: 6px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background-color: var(--surface) !important;
    color: var(--text) !important;
}

/* ── Заголовки ───────────────────────────────────────────── */
h1, h2, h3, h4 {
    color: var(--text) !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

/* ── Текстовые элементы Streamlit ────────────────────────── */
p, span, div, label,
[data-testid="stText"],
[data-testid="stMarkdown"] {
    color: var(--text) !important;
    font-family: var(--font) !important;
}

/* ── Слайдеры ────────────────────────────────────────────── */
[data-testid="stSlider"] * { color: var(--text) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent) !important;
}

/* ── Selectbox / dropdown ────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] div {
    color: var(--text) !important;
    background-color: var(--bg) !important;
    font-family: var(--font) !important;
}
[data-baseweb="select"] > div {
    background-color: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Кнопка Predict ──────────────────────────────────────── */
div.stButton > button {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    font-family: var(--font) !important;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* ── Убрать рамки у st.info / st.success и пр. ───────────── */
[data-testid="stAlert"] {
    background-color: var(--accent-bg) !important;
    border: 1px solid #BFDBFE !important;
    color: var(--text) !important;
    border-radius: var(--radius) !important;
}

/* ── Spinner ─────────────────────────────────────────────── */
[data-testid="stSpinner"] * { color: var(--accent) !important; }

/* ── Разделитель ─────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Метрика ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted)  !important; font-size: 0.85rem !important; }
[data-testid="stMetricValue"]  { color: var(--text)   !important; font-size: 1.8rem !important; font-weight: 700 !important; }

/* ── Selectbox — фон выпадающего списка ──────────────────── */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E5E3 !important;
    color: #1A1A1A !important;
}
div[data-baseweb="select"] span {
    color: #1A1A1A !important;
}
div[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
}
li[role="option"] {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
}
li[role="option"]:hover {
    background-color: #F7F7F5 !important;
}

/* ── Шапка Streamlit — белый фон вместо тёмного ─────────── */
header[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #E5E5E3 !important;
}
.stDeployButton { display: none; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(path: str):
    """Загрузка модели с кэшированием (joblib или pickle)."""
    return joblib.load(path)


def try_load(path: str):
    """Безопасная загрузка с отображением ошибки через st.error()."""
    if not os.path.exists(path):
        st.error(f"Model file not found: `{path}`  \n"
                 f"Run `python fix_models.py` to generate the models.")
        return None
    try:
        return load_model(path)
    except Exception as exc:
        st.error(f"Error loading `{path}`:  \n`{exc}`")
        return None


def predict_risk(model, scaler, features: list) -> float:
    """Возвращает вероятность положительного класса (болезнь есть)."""
    X = np.array(features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    return float(model.predict_proba(X_scaled)[0][1])


def show_result(probability: float) -> None:
    """
    Отображает метрику, прогресс-бар и карточку интерпретации.
    Цвета: зелёный < 30 %, жёлтый 30–60 %, красный > 60 %.
    """
    pct = probability * 100

    # ── Метрика ───────────────────────────────────────────────────
    st.metric(label="Прогнозируемый риск", value=f"{pct:.1f}%")

    # ── Параметры цветовой схемы ──────────────────────────────────
    if pct < 30:
        bar_color  = "#16A34A"
        bg_color   = "#F0FDF4"
        bd_color   = "#BBF7D0"
        label      = "Низкий риск"
        icon       = "✓"
        text = (
            "<b>Низкий риск</b> — значимых показателей не выявлено. "
            "Поддерживайте здоровый образ жизни и регулярно проходите профилактические осмотры."
        )
    elif pct < 60:
        bar_color  = "#D97706"
        bg_color   = "#FFFBEB"
        bd_color   = "#FDE68A"
        label      = "Умеренный риск"
        icon       = "!"
        text = (
            "<b>Умеренный риск</b> — рекомендуется консультация врача. "
            "Обратите внимание на образ жизни и известные факторы риска."
        )
    else:
        bar_color  = "#DC2626"
        bg_color   = "#FEF2F2"
        bd_color   = "#FECACA"
        label      = "Высокий риск"
        icon       = "⚠"
        text = (
            "<b>Высокий риск</b> — необходима срочная консультация врача. "
            "Незамедлительно обратитесь к квалифицированному специалисту для детального обследования."
        )

    # ── Прогресс-бар ──────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin: 6px 0 12px 0;">
        <div style="display:flex; justify-content:space-between;
                    font-size:0.8rem; color:#6B7280; margin-bottom:4px;">
            <span>{label}</span><span>{pct:.1f}%</span>
        </div>
        <div style="background:#E5E5E3; border-radius:99px; height:8px; width:100%; overflow:hidden;">
            <div style="background:{bar_color}; width:{min(pct, 100):.1f}%;
                        height:8px; border-radius:99px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Карточка результата ───────────────────────────────────────
    st.markdown(f"""
    <div style="background:{bg_color}; border:1px solid {bd_color};
                border-radius:10px; padding:16px 20px; margin-top:4px;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
            <span style="font-size:1.1rem; font-weight:700; color:{bar_color};">{icon}</span>
            <span style="font-size:1rem; font-weight:600; color:#1A1A1A;">{label} — {pct:.1f}%</span>
        </div>
        <p style="margin:0; font-size:0.9rem; color:#374151; line-height:1.55;">{text}</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Переиспользуемые компоненты разметки
# ─────────────────────────────────────────────────────────────────────────────

def page_header(icon: str, title: str, subtitle: str) -> None:
    """Чистая шапка страницы болезни."""
    st.markdown(f"""
    <div style="margin-bottom: 28px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
            <span style="font-size:1.8rem;">{icon}</span>
            <h2 style="margin:0; font-size:1.6rem; font-weight:700;
                       color:#1A1A1A; letter-spacing:-0.03em;">{title}</h2>
        </div>
        <p style="margin:0; color:#6B7280; font-size:0.95rem; padding-left:52px;">{subtitle}</p>
        <div style="margin-top:20px; border-top:1px solid #E5E5E3;"></div>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str) -> None:
    """Подпись раздела — мелкий заглавный текст."""
    st.markdown(f"""
    <p style="font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
              text-transform:uppercase; color:#6B7280; margin:20px 0 8px 0;">{text}</p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Боковая панель — навигация
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px 0;">
        <div style="font-size:1.05rem; font-weight:700; color:#1A1A1A; letter-spacing:-0.02em;">
            🏥 MedPredict
        </div>
        <div style="font-size:0.78rem; color:#6B7280; margin-top:2px;">Система прогнозирования</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="border-top:1px solid #E5E5E3; margin-bottom:16px;"></div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Навигация",
        ["Главная", "Инсульт", "Болезни сердца", "Диабет"],
        label_visibility="collapsed",
    )

    st.markdown('<div style="border-top:1px solid #E5E5E3; margin-top:16px;"></div>',
                unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.75rem; color:#6B7280; margin-top:12px; line-height:1.6;">
        Только для исследовательских целей.<br>Версия 1.0 · 2024
    </p>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ СТРАНИЦА
# ═════════════════════════════════════════════════════════════════════════════
if page == "Главная":

    # ── Заголовок ─────────────────────────────────────────────────
    st.markdown("""
    <h1 style="font-size:2.2rem; font-weight:800; color:#1A1A1A;
               letter-spacing:-0.04em; margin-bottom:8px;">
        Интеллектуальная система прогнозирования заболеваний
    </h1>
    <p style="font-size:1.05rem; color:#6B7280; margin-bottom:36px; max-width:680px;">
        Система машинного обучения для оценки риска инсульта,
        болезней сердца и сахарного диабета на основе клинических показателей пациента.
    </p>
    """, unsafe_allow_html=True)

    # ── Три карточки болезней ─────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    card_css = (
        "background:#FFFFFF; border:1px solid #E5E5E3; border-radius:10px;"
        "padding:24px 20px; height:100%;"
    )

    with c1:
        st.markdown(f"""
        <div style="{card_css}">
            <div style="font-size:1.8rem; margin-bottom:12px;">🧠</div>
            <div style="font-size:1rem; font-weight:600; color:#1A1A1A; margin-bottom:6px;">Инсульт</div>
            <p style="font-size:0.875rem; color:#6B7280; line-height:1.55; margin-bottom:16px;">
                Оценка риска инсульта на основе демографических данных,
                показателей здоровья и образа жизни пациента.
            </p>
            <span style="display:inline-block; background:#EFF6FF; color:#2563EB;
                         font-size:0.72rem; font-weight:600; padding:3px 10px;
                         border-radius:99px; letter-spacing:0.03em;">XGBoost</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="{card_css}">
            <div style="font-size:1.8rem; margin-bottom:12px;">❤️</div>
            <div style="font-size:1rem; font-weight:600; color:#1A1A1A; margin-bottom:6px;">Болезни сердца</div>
            <p style="font-size:0.875rem; color:#6B7280; line-height:1.55; margin-bottom:16px;">
                Прогнозирование ишемической болезни сердца по кардиологическим
                и инструментальным показателям обследования.
            </p>
            <span style="display:inline-block; background:#EFF6FF; color:#2563EB;
                         font-size:0.72rem; font-weight:600; padding:3px 10px;
                         border-radius:99px; letter-spacing:0.03em;">XGBoost</span>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="{card_css}">
            <div style="font-size:1.8rem; margin-bottom:12px;">💉</div>
            <div style="font-size:1rem; font-weight:600; color:#1A1A1A; margin-bottom:6px;">Диабет</div>
            <p style="font-size:0.875rem; color:#6B7280; line-height:1.55; margin-bottom:16px;">
                Выявление риска сахарного диабета 2-го типа
                по биометрическим и лабораторным показателям.
            </p>
            <span style="display:inline-block; background:#EFF6FF; color:#2563EB;
                         font-size:0.72rem; font-weight:600; padding:3px 10px;
                         border-radius:99px; letter-spacing:0.03em;">XGBoost</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:36px;'></div>", unsafe_allow_html=True)

    # ── Как работает система ──────────────────────────────────────
    st.markdown("""
    <p style="font-size:0.72rem; font-weight:600; letter-spacing:0.08em;
              text-transform:uppercase; color:#6B7280; margin-bottom:14px;">Как это работает</p>
    """, unsafe_allow_html=True)

    h1, h2 = st.columns(2)
    info_card = (
        "background:#F7F7F5; border:1px solid #E5E5E3; border-radius:10px;"
        "padding:20px; font-size:0.875rem; color:#374151; line-height:1.6;"
    )
    with h1:
        st.markdown(f"""
        <div style="{info_card}">
            <b style="color:#1A1A1A;">Алгоритмы ML</b><br>
            Система использует ансамблевые методы (XGBoost, Random Forest, SVM),
            обученные на клинических наборах данных с балансировкой классов через SMOTE.
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div style="{info_card}">
            <b style="color:#1A1A1A;">Входные данные</b><br>
            Введите клинические показатели в соответствующем разделе.
            Система масштабирует данные и возвращает вероятность заболевания
            с цветовой индикацией уровня риска.
        </div>
        """, unsafe_allow_html=True)

    # ── Дисклеймер ────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
                padding:14px 20px; margin-top:28px;
                font-size:0.875rem; color:#1E40AF; line-height:1.55;">
        <b>ℹ Важно:</b> Данный инструмент предназначен исключительно для исследовательских целей
        и не заменяет профессиональную медицинскую консультацию. Все прогнозы носят
        информационный характер. Для постановки диагноза обратитесь к квалифицированному специалисту.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: ИНСУЛЬТ
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Инсульт":

    page_header(
        "🧠", "Прогнозирование инсульта",
        "Введите клинические и демографические данные пациента"
    )

    # Загрузка модели и скейлера
    model  = try_load("stroke/models/xgboost.pkl")
    scaler = try_load("stroke/models/scaler.pkl")

    if model and scaler:
        c1, c2 = st.columns(2)

        # ── Левая колонка ─────────────────────────────────────────
        with c1:
            section_label("Демография")
            age = st.slider("Возраст", min_value=1, max_value=100, value=45, key="s_age")

            gender_label = st.selectbox("Пол", ["Мужской", "Женский", "Другой"], key="s_gender")
            gender = {"Мужской": 1, "Женский": 0, "Другой": 2}[gender_label]

            married_label = st.selectbox("Семейное положение", ["Да", "Нет"], key="s_married")
            ever_married = 1 if married_label == "Да" else 0

            residence_label = st.selectbox("Тип проживания", ["Городской", "Сельский"], key="s_res")
            residence = 1 if residence_label == "Городской" else 0

            work_map = {
                "Частный":           2,
                "Самозанятый":       3,
                "Госслужба":         0,
                "Дети":              1,
                "Никогда не работал": 4,
            }
            work_label = st.selectbox("Тип занятости", list(work_map.keys()), key="s_work")
            work_type  = work_map[work_label]

        # ── Правая колонка ────────────────────────────────────────
        with c2:
            section_label("Показатели здоровья")

            hyp_label   = st.selectbox("Гипертония", ["Нет", "Да"], key="s_hyp")
            hypertension = 1 if hyp_label == "Да" else 0

            hd_label    = st.selectbox("Болезнь сердца", ["Нет", "Да"], key="s_hd")
            heart_disease = 1 if hd_label == "Да" else 0

            avg_glucose = st.slider(
                "Средний уровень глюкозы (мг/дл)",
                min_value=50.0, max_value=300.0, value=100.0, step=0.5, key="s_glu"
            )
            bmi = st.slider(
                "Индекс массы тела (ИМТ)",
                min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="s_bmi"
            )

            smoking_map = {
                "Никогда не курил": 1,
                "Бросил":           0,
                "Курит":            2,
                "Неизвестно":       3,
            }
            smoke_label    = st.selectbox("Статус курения", list(smoking_map.keys()), key="s_smk")
            smoking_status = smoking_map[smoke_label]

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        # ── Кнопка прогноза ───────────────────────────────────────
        if st.button("Оценить риск — Инсульт", key="stroke_btn"):
            features = [
                age, hypertension, heart_disease,
                avg_glucose, bmi,
                gender, ever_married, work_type, residence, smoking_status,
            ]
            with st.spinner("Анализ данных..."):
                try:
                    prob = predict_risk(model, scaler, features)
                    st.markdown("---")
                    st.markdown("**Результат прогноза**")
                    show_result(prob)
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: БОЛЕЗНИ СЕРДЦА
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Болезни сердца":

    page_header(
        "❤️", "Прогнозирование болезней сердца",
        "Введите кардиологические показатели пациента"
    )

    model  = try_load("heart/models/xgboost.pkl")
    scaler = try_load("heart/models/scaler.pkl")

    if model and scaler:
        c1, c2 = st.columns(2)

        # ── Левая колонка ─────────────────────────────────────────
        with c1:
            section_label("Общие данные")
            age = st.slider("Возраст (лет)", min_value=20, max_value=80, value=50, key="h_age")

            sex_label = st.selectbox("Пол", ["Мужской", "Женский"], key="h_sex")
            sex = 1 if sex_label == "Мужской" else 0

            cp_map = {
                "Типичная стенокардия":    0,
                "Атипичная стенокардия":   1,
                "Неангинальная боль":      2,
                "Бессимптомная":           3,
            }
            cp_label = st.selectbox("Тип боли в груди (cp)", list(cp_map.keys()), key="h_cp")
            cp = cp_map[cp_label]

            trestbps = st.slider(
                "АД в покое (мм рт.ст.)", min_value=80, max_value=200, value=120, key="h_bp"
            )
            chol = st.slider(
                "Холестерин (мг/дл)", min_value=100, max_value=600, value=200, key="h_chol"
            )
            fbs_label = st.selectbox("Сахар натощак > 120 мг/дл", ["Нет", "Да"], key="h_fbs")
            fbs = 1 if fbs_label == "Да" else 0

            restecg_map = {"Норма": 0, "Отклонение ST-T": 1, "Гипертрофия ЛЖ": 2}
            restecg_label = st.selectbox("ЭКГ в покое (restecg)", list(restecg_map.keys()), key="h_ecg")
            restecg = restecg_map[restecg_label]

        # ── Правая колонка ────────────────────────────────────────
        with c2:
            section_label("Нагрузочные показатели")
            thalach = st.slider(
                "Макс. ЧСС при нагрузке", min_value=60, max_value=220, value=150, key="h_thal"
            )

            exang_label = st.selectbox("Стенокардия при нагрузке (exang)", ["Нет", "Да"], key="h_ex")
            exang = 1 if exang_label == "Да" else 0

            oldpeak = st.slider(
                "Депрессия ST (oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1, key="h_old"
            )

            slope_map = {"Восходящий": 0, "Плоский": 1, "Нисходящий": 2}
            slope_label = st.selectbox("Наклон ST (slope)", list(slope_map.keys()), key="h_slope")
            slope = slope_map[slope_label]

            ca = st.selectbox(
                "Кол-во крупных сосудов (ca)", [0, 1, 2, 3], key="h_ca"
            )

            thal_map = {"Норма": 1, "Фиксированный дефект": 2, "Обратимый дефект": 3}
            thal_label = st.selectbox("Таллиевый тест (thal)", list(thal_map.keys()), key="h_thalmap")
            thal = thal_map[thal_label]

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        # ── Кнопка прогноза ───────────────────────────────────────
        if st.button("Оценить риск — Болезни сердца", key="heart_btn"):
            features = [age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]
            with st.spinner("Анализ данных..."):
                try:
                    prob = predict_risk(model, scaler, features)
                    st.markdown("---")
                    st.markdown("**Результат прогноза**")
                    show_result(prob)
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# СТРАНИЦА: ДИАБЕТ
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Диабет":

    page_header(
        "💉", "Прогнозирование диабета",
        "Введите биометрические и лабораторные показатели пациента"
    )

    model  = try_load("diabetes/models/xgboost.pkl")
    scaler = try_load("diabetes/models/scaler.pkl")

    if model and scaler:
        c1, c2 = st.columns(2)

        # ── Левая колонка ─────────────────────────────────────────
        with c1:
            section_label("Основные показатели")
            pregnancies = st.slider(
                "Количество беременностей", min_value=0, max_value=20, value=1, key="d_preg"
            )
            glucose = st.slider(
                "Уровень глюкозы (мг/дл)", min_value=0, max_value=200, value=100, key="d_glu"
            )
            blood_pressure = st.slider(
                "Артериальное давление (мм рт.ст.)", min_value=0, max_value=130, value=70, key="d_bp"
            )
            skin_thickness = st.slider(
                "Толщина кожной складки (мм)", min_value=0, max_value=100, value=20, key="d_skin"
            )

        # ── Правая колонка ────────────────────────────────────────
        with c2:
            section_label("Дополнительные показатели")
            insulin = st.slider(
                "Уровень инсулина (мкЕд/мл)", min_value=0, max_value=900, value=80, key="d_ins"
            )
            bmi = st.slider(
                "Индекс массы тела (ИМТ)", min_value=0.0, max_value=70.0, value=25.0, step=0.1, key="d_bmi"
            )
            dpf = st.slider(
                "Функция родословной диабета (DPF)",
                min_value=0.0, max_value=2.5, value=0.3, step=0.01, key="d_dpf"
            )
            age = st.slider(
                "Возраст (лет)", min_value=1, max_value=100, value=30, key="d_age"
            )

        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        # ── Кнопка прогноза ───────────────────────────────────────
        if st.button("Оценить риск — Диабет", key="diabetes_btn"):
            features = [pregnancies, glucose, blood_pressure,
                        skin_thickness, insulin, bmi, dpf, age]
            with st.spinner("Анализ данных..."):
                try:
                    prob = predict_risk(model, scaler, features)
                    st.markdown("---")
                    st.markdown("**Результат прогноза**")
                    show_result(prob)
                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {e}")
