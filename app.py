from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta
from html import escape
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "maternal_risk_xgboost.joblib"
CLINIC_DIR = PROJECT_ROOT / "data" / "clinic"
AUTH_DIR = PROJECT_ROOT / "data" / "auth"
ATTACHMENTS_DIR = CLINIC_DIR / "attachments"
PATIENTS_PATH = CLINIC_DIR / "patients.csv"
VISITS_PATH = CLINIC_DIR / "visits.csv"
ATTACHMENTS_PATH = CLINIC_DIR / "attachments.csv"
ACTIVITY_PATH = CLINIC_DIR / "activity.csv"
USERS_PATH = AUTH_DIR / "users.csv"

# Внешнее хранилище (включается, когда задан DATABASE_URL / SUPABASE_DB_URL / TURSO_DATABASE_URL).
# Пока переменных нет — приложение прозрачно работает на локальных CSV.
STORAGE_TABLES = {
    "patients.csv": "patients",
    "visits.csv": "visits",
    "attachments.csv": "attachments",
    "activity.csv": "activity",
    "users.csv": "users",
}
ATTACHMENT_BUCKET = "attachments"
LOGO_URL = "app/static/landing/branding/vascularai-logo.png"
MODEL_VIEWER_URL = "/app/static/landing/model-viewer.html"
LANDING_FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E"
    "%3Crect width='40' height='40' rx='12' fill='%230972df'/%3E"
    "%3Cpath d='M7 21h7l4-10 6 20 3-10h6' fill='none' stroke='white' "
    "stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E"
)
DEFAULT_ADMIN_PASSWORD = "admin123"
NAV_KEY = "main_nav"
PATIENT_VIEW_KEY = "patient_view"
PENDING_RISK_FILTER = "pending_risk_filter"
SESSION_QUERY_PARAM = "va_session"
SESSION_TTL_DAYS = 14

# Левая колонка окна входа: небольшие блоки о платформе, которые сменяются по кругу.
ENTRY_TIP_SECONDS = 4
ENTRY_TIPS = [
    (
        "Объяснимые решения",
        "Каждый расчёт сохраняет вероятности и SHAP-факторы, поэтому видно, что повлияло на оценку.",
    ),
    (
        "Один экран вместо пяти",
        "Карты пациенток, история визитов, вложения и риск собраны в защищённом кабинете.",
    ),
    (
        "Два рабочих места",
        "Врач ведёт пациенток, администратор управляет доступами и смотрит аналитику отдельно.",
    ),
    (
        "Данные остаются в клинике",
        "Карты, визиты и файлы хранятся локально, пароли — только в виде хэшей.",
    ),
    (
        "Модель XGBoost",
        "Оценка риска опирается на материнские клинические данные и шесть ключевых показателей.",
    ),
]

PATIENT_COLUMNS = [
    "patient_id",
    "initials",
    "age",
    "gestational_week",
    "gravida",
    "parity",
    "doctor",
    "status",
    "created_at",
    "note",
    "bmi",
    "anamnesis",
]

VISIT_COLUMNS = [
    "visit_id",
    "patient_id",
    "recorded_at",
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
    "risk_label",
    "risk_probability",
    "prob_low",
    "prob_mid",
    "prob_high",
    "top_factor",
    "visit_note",
    "bmi",
    "plgf",
    "papp_a",
    "sflt1",
    "map_value",
    "anamnesis",
]

ATTACHMENT_COLUMNS = [
    "attachment_id",
    "patient_id",
    "filename",
    "stored_path",
    "uploaded_at",
    "note",
]

ACTIVITY_COLUMNS = [
    "event_id",
    "event_at",
    "actor",
    "actor_role",
    "action",
    "target",
    "details",
]

ACTION_LABELS = {
    "login": "Вход в систему",
    "logout": "Выход из системы",
    "patient_created": "Создана карта пациентки",
    "visit_calculated": "Расчёт риска",
    "attachment_uploaded": "Загружено вложение",
    "account_created": "Создан аккаунт",
    "account_updated": "Изменён доступ",
    "password_changed": "Смена пароля",
    "profile_updated": "Обновлён профиль",
}

USER_COLUMNS = [
    "user_id",
    "username",
    "password_hash",
    "full_name",
    "role",
    "status",
    "created_at",
    "last_login_at",
    "session_hash",
    "session_expires_at",
    "specialty",
    "clinic",
    "email",
    "phone",
    "notify_email",
    "notify_sms",
]

ROLE_LABELS = {
    "doctor": "Врач",
    "admin": "Администратор",
}

ROLE_VALUES = {label: value for value, label in ROLE_LABELS.items()}

STATUS_LABELS = {
    "active": "Активен",
    "blocked": "Отключен",
}

STATUS_VALUES = {label: value for value, label in STATUS_LABELS.items()}

RISK_META = {
    "low risk": {
        "title": "Низкий риск",
        "status": "Стабильно",
        "summary": "Показатели ближе к профилю низкого риска в обучающем наборе.",
        "color": "#328d84",
        "text": "#FFFFFF",
    },
    "mid risk": {
        "title": "Средний риск",
        "status": "Наблюдение",
        "summary": "Есть факторы, которые требуют более внимательного наблюдения.",
        "color": "#b28a23",
        "text": "#FFFFFF",
    },
    "high risk": {
        "title": "Высокий риск",
        "status": "Внимание",
        "summary": "Профиль похож на записи высокого риска в обучающем наборе.",
        "color": "#bc617c",
        "text": "#FFFFFF",
    },
}

FEATURE_META = {
    "Age": {
        "label": "Возраст пациентки",
        "hint": "Полных лет на момент беременности",
        "unit": "лет",
        "min": 10,
        "max": 70,
        "default": 29,
        "step": 1,
        "format": "%d",
    },
    "SystolicBP": {
        "label": "Систолическое давление",
        "hint": "Верхнее число артериального давления",
        "unit": "мм рт. ст.",
        "min": 70,
        "max": 180,
        "default": 120,
        "step": 1,
        "format": "%d",
    },
    "DiastolicBP": {
        "label": "Диастолическое давление",
        "hint": "Нижнее число артериального давления",
        "unit": "мм рт. ст.",
        "min": 40,
        "max": 120,
        "default": 80,
        "step": 1,
        "format": "%d",
    },
    "BS": {
        "label": "Сахар крови",
        "hint": "Концентрация глюкозы в крови",
        "unit": "ммоль/л",
        "min": 3.0,
        "max": 25.0,
        "default": 7.5,
        "step": 0.1,
        "format": "%.1f",
    },
    "BodyTemp": {
        "label": "Температура тела",
        "hint": "Температура в градусах Фаренгейта",
        "unit": "F",
        "min": 95.0,
        "max": 105.0,
        "default": 98.0,
        "step": 0.1,
        "format": "%.1f",
    },
    "HeartRate": {
        "label": "Пульс в покое",
        "hint": "Частота сердечных сокращений",
        "unit": "уд/мин",
        "min": 40,
        "max": 140,
        "default": 76,
        "step": 1,
        "format": "%d",
    },
}


st.set_page_config(
    page_title="VascularAI Кабинет",
    page_icon=LANDING_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def add_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            /* Палитра лендинга VascularAI. Значения взяты 1:1 из лендинга
               (static/landing/assets/index-CBHG-z2o.css, :root). Никаких других цветов не используем. */
            --ink: #4a3238;
            --nav: #4a3238;
            --nav-soft: #5f3b41;
            --blue: #bd6875;
            --accent: #bd6875;
            --accent-dark: #a85763;
            --cyan: #e9aeb1;
            --muted: #806b6c;
            --line: #ead6cf;
            --pale: #fdf8f5;
            --panel: #fffaf5;
            --white: #fffdfa;
            --tint: #fff0ed;
            --track: #f0dfda;
            --landing-blue: #bd6875;
            --green: #328d84;
            --amber: #b28a23;
            --high: #bc617c;
            --shadow: 0 18px 42px rgba(74, 50, 56, 0.11);
            /* Иконки интерфейса (маски SVG) */
            --icon-grid: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9'%3E%3Crect x='3' y='3' width='7.5' height='7.5' rx='2'/%3E%3Crect x='13.5' y='3' width='7.5' height='7.5' rx='2'/%3E%3Crect x='3' y='13.5' width='7.5' height='7.5' rx='2'/%3E%3Crect x='13.5' y='13.5' width='7.5' height='7.5' rx='2'/%3E%3C/svg%3E");
            --icon-plus: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M12 5v14M5 12h14'/%3E%3C/svg%3E");
            --icon-users: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Ccircle cx='9' cy='8' r='3.4'/%3E%3Cpath d='M2.6 20c0-3.4 2.9-5.6 6.4-5.6s6.4 2.2 6.4 5.6'/%3E%3Cpath d='M17 5.4A3.2 3.2 0 0 1 17 11.6M18.4 19.9c0-2.4-.7-4-2-5'/%3E%3C/svg%3E");
            --icon-chart: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M4 19V5M4 19h16'/%3E%3Cpath d='M8 16v-4M12 16V8M16 16v-6'/%3E%3C/svg%3E");
            --icon-gear: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='3.2'/%3E%3Cpath d='M12 3.2v2.2M12 18.6v2.2M4.8 7.8l1.9 1.1M17.3 15.1l1.9 1.1M4.8 16.2l1.9-1.1M17.3 8.9l1.9-1.1'/%3E%3C/svg%3E");
            --icon-shield: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M12 3l7 3v6c0 4.2-2.9 7.6-7 9-4.1-1.4-7-4.8-7-9V6z'/%3E%3Cpath d='M9.3 12.2l1.9 1.9 3.6-3.7'/%3E%3C/svg%3E");
        }

        * { box-sizing: border-box; letter-spacing: 0; }

        .stApp {
            background:
                radial-gradient(ellipse at 76% 12%, rgba(233, 174, 177, 0.34), transparent 34rem),
                linear-gradient(120deg, rgba(255, 250, 245, 0.76), transparent 66%),
                var(--pale);
            color: var(--ink);
            overflow-x: hidden;
        }

        .block-container {
            max-width: none;
            padding: 1rem 1.35rem 2rem;
        }

        #MainMenu, footer, [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stHeader"] {
            display: none;
        }

        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
            pointer-events: none !important;
        }

        h1, h2, h3, h4, p, label, span, div { color: inherit; }

        section[data-testid="stSidebar"] {
            background: var(--nav);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            display: block !important;
            margin-left: 0 !important;
            max-width: 17.5rem !important;
            min-width: 17.5rem !important;
            transform: none !important;
            visibility: visible !important;
            width: 17.5rem !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] {
            display: block !important;
            margin-left: 0 !important;
            max-width: 17.5rem !important;
            min-width: 17.5rem !important;
            transform: none !important;
            visibility: visible !important;
            width: 17.5rem !important;
        }

        section[data-testid="stSidebar"] > div {
            background:
                linear-gradient(180deg, rgba(74, 50, 56, 0.98), rgba(74, 50, 56, 0.98)),
                var(--nav);
            padding: 1.5rem 0.75rem 1rem;
        }

        section[data-testid="stSidebar"] * {
            color: rgba(255, 255, 255, 0.88);
        }

        .sidebar-brand {
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.09);
            display: flex;
            gap: 0.78rem;
            margin-bottom: 0.9rem;
            padding: 0.2rem 0.35rem 1.05rem;
        }

        .sidebar-mark {
            align-items: center;
            background: rgba(233, 174, 177, 0.18);
            border: 1px solid rgba(233, 174, 177, 0.36);
            border-radius: 8px;
            color: var(--cyan) !important;
            display: inline-flex;
            flex: 0 0 2.55rem;
            font-size: 0.82rem;
            font-weight: 900;
            height: 2.55rem;
            justify-content: center;
            line-height: 1;
            width: 2.55rem;
        }

        .sidebar-brand strong {
            color: #ffffff;
            display: block;
            font-size: 1.04rem;
            font-weight: 850;
            line-height: 1.05;
        }

        .sidebar-brand span, .sidebar-user span {
            color: rgba(255, 255, 255, 0.56);
            display: block;
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.12rem;
        }

        .sidebar-role {
            align-items: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 8px;
            color: #ffffff;
            display: flex;
            font-size: 0.82rem;
            font-weight: 850;
            justify-content: space-between;
            margin: 0 0 1.35rem;
            padding: 0.68rem 0.84rem;
        }

        .sidebar-section {
            color: rgba(255, 255, 255, 0.42);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.05em;
            margin: 1rem 0 0.35rem;
            padding: 0 0.48rem;
            text-transform: uppercase;
        }

        .sidebar-user {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 8px;
            margin-top: 1.1rem;
            padding: 0.7rem 0.75rem;
        }

        .sidebar-user strong {
            color: #ffffff;
            display: block;
            font-size: 0.86rem;
            font-weight: 850;
            line-height: 1.15;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
            background: transparent;
            border: 0;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            padding: 0;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            border-radius: 8px;
            color: rgba(255, 255, 255, 0.72) !important;
            font-weight: 800;
            min-height: 2.55rem;
            padding: 0.62rem 0.72rem;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            background: rgba(233, 174, 177, 0.18);
            box-shadow: inset 4px 0 0 var(--cyan);
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255, 255, 255, 0.08);
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] div.stButton > button {
            background: transparent !important;
            border-color: rgba(255, 255, 255, 0.16) !important;
            color: #ffffff !important;
        }

        .app-header {
            background: rgba(255, 250, 245, 0.84);
            border: 1px solid rgba(234, 214, 207, 0.9);
            border-radius: 8px;
            box-shadow: 0 10px 28px rgba(95, 59, 65, 0.07);
            margin-bottom: 0.9rem;
            padding: 0.72rem 0.86rem;
        }

        .header-grid {
            align-items: center;
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1fr) auto;
        }

        .brand-lockup {
            align-items: center;
            display: flex;
            gap: 0.9rem;
            min-width: 0;
        }

        .brand-lockup img {
            display: block;
            height: 1.95rem;
            max-width: 9.5rem;
            object-fit: contain;
        }

        .brand-copy h1 {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.12;
            margin: 0;
        }

        .brand-copy p {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.42;
            margin: 0.1rem 0 0;
        }

        .header-actions {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            justify-content: flex-end;
        }

        .header-pill {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 700;
            padding: 0.38rem 0.68rem;
            white-space: nowrap;
        }

        .header-pill.user-pill {
            background: var(--ink);
            border-color: var(--ink);
            color: #ffffff;
        }

        /* ------ Окно входа: левая колонка с брендом и 3D, правая с формой ------ */
        .entry-brand img {
            display: block;
            height: 2.9rem;
            margin-bottom: 1.5rem;
            max-width: 13rem;
            object-fit: contain;
        }

        .entry-kicker {
            color: var(--blue);
            font-size: 0.7rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .entry-brand h1 {
            color: var(--ink);
            font-size: clamp(2rem, 3.4vw, 3.05rem);
            font-weight: 850;
            line-height: 1.03;
            margin: 0.6rem 0 0.8rem;
            max-width: 30rem;
        }

        .entry-brand p, .auth-form-card p, .auth-form-heading p {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.5;
            margin: 0;
            max-width: 33rem;
        }

        .entry-stats {
            display: grid;
            gap: 0.6rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 1.3rem;
            max-width: 33rem;
        }

        .entry-stat {
            background: rgba(255, 253, 250, 0.86);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 0.68rem 0.78rem;
        }

        .entry-stat strong {
            color: var(--ink);
            display: block;
            font-size: 1.35rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .entry-stat span {
            color: var(--muted);
            display: block;
            font-size: 0.7rem;
            font-weight: 750;
            line-height: 1.3;
            margin-top: 0.22rem;
        }

        .entry-tips {
            background:
                radial-gradient(circle at 88% 6%, rgba(233, 174, 177, 0.34), transparent 13rem),
                var(--nav);
            border-radius: 12px;
            margin-top: 0.6rem;
            min-height: 8.9rem;
            overflow: hidden;
            padding: 1.05rem 1.15rem;
            position: relative;
        }

        .entry-tips small {
            color: rgba(255, 255, 255, 0.52);
            display: block;
            font-size: 0.66rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .entry-tip {
            animation: tipFade 20s infinite;
            left: 1.15rem;
            opacity: 0;
            padding-right: 1.15rem;
            position: absolute;
            right: 1.15rem;
            top: 2.55rem;
        }

        .entry-tip strong {
            color: #ffffff;
            display: block;
            font-size: 1.06rem;
            font-weight: 850;
            line-height: 1.2;
        }

        .entry-tip span {
            color: rgba(255, 255, 255, 0.74);
            display: block;
            font-size: 0.88rem;
            line-height: 1.45;
            margin-top: 0.32rem;
        }

        @keyframes tipFade {
            0% { opacity: 0; transform: translateY(0.7rem); }
            3% { opacity: 1; transform: translateY(0); }
            17% { opacity: 1; transform: translateY(0); }
            20% { opacity: 0; transform: translateY(-0.7rem); }
            100% { opacity: 0; transform: translateY(-0.7rem); }
        }

        .entry-tip-dots {
            bottom: 0.95rem;
            display: flex;
            gap: 0.32rem;
            left: 1.15rem;
            position: absolute;
        }

        .entry-tip-dots i {
            animation: tipDot 20s infinite;
            background: rgba(255, 255, 255, 0.22);
            border-radius: 999px;
            display: block;
            height: 0.22rem;
            width: 1.5rem;
        }

        @keyframes tipDot {
            0% { background: var(--cyan); }
            20% { background: rgba(255, 255, 255, 0.22); }
            100% { background: rgba(255, 255, 255, 0.22); }
        }

        iframe[src*="model-viewer.html"] {
            border: 0 !important;
            display: block;
            margin: 0.5rem 0 0;
            overflow: hidden;
        }

        /* Карточка входа в правой колонке */
        [data-testid="stForm"]:has(.auth-form-heading) {
            background: rgba(255, 253, 250, 0.97);
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: 0 26px 64px rgba(74, 50, 56, 0.16);
            padding: 1.75rem 1.6rem 1.35rem;
        }

        .auth-form-heading {
            margin-bottom: 1.05rem;
        }

        .auth-form-heading .entry-kicker {
            margin-bottom: 0.5rem;
        }

        .auth-form-card h2, .auth-form-heading h2 {
            color: var(--ink);
            font-size: 1.85rem;
            font-weight: 850;
            line-height: 1.08;
            margin: 0 0 0.42rem;
        }

        .auth-safe-note {
            background: var(--tint);
            border: 1px solid var(--line);
            border-radius: 10px;
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 750;
            line-height: 1.4;
            margin-top: 0.75rem;
            padding: 0.7rem 0.78rem;
        }

        .login-copy, .guide-card, .action-panel {
            background: rgba(255, 250, 245, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1rem;
        }

        .login-copy img {
            display: block;
            height: 3rem;
            margin-bottom: 1rem;
            max-width: 12rem;
            object-fit: contain;
        }

        .login-copy h1 {
            color: var(--ink);
            font-size: clamp(1.8rem, 4vw, 3.3rem);
            font-weight: 850;
            line-height: 1;
            margin: 0 0 0.75rem;
        }

        .login-copy p, .page-note, .action-panel p {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0;
        }

        .bootstrap-note {
            background: var(--tint);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 700;
            margin-top: 1rem;
            padding: 0.7rem 0.78rem;
        }

        [data-testid="stRadio"] > div {
            background: rgba(255, 250, 245, 0.74);
            border: 1px solid var(--line);
            border-radius: 8px;
            gap: 0.25rem;
            padding: 0.35rem;
        }

        [data-testid="stRadio"] label {
            border-radius: 6px;
            color: var(--muted) !important;
            font-weight: 700;
            padding: 0.28rem 0.38rem;
        }

        [data-testid="stRadio"] label:has(input:checked) {
            background: var(--accent);
            color: #ffffff !important;
        }

        .panel, [data-testid="stForm"] {
            background: rgba(255, 250, 245, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1rem;
        }

        [data-testid="stForm"] {
            margin-bottom: 0.9rem;
        }

        .panel-title, [data-testid="stForm"] h2, [data-testid="stForm"] h3 {
            color: var(--ink);
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.18;
            margin: 0 0 0.75rem;
        }

        .page-heading {
            margin: 0.25rem 0 1rem;
        }

        .page-heading h1 {
            color: var(--ink);
            font-size: clamp(1.65rem, 3vw, 2.45rem);
            font-weight: 850;
            line-height: 1.05;
            margin: 0;
        }

        .page-heading p {
            color: var(--muted);
            font-size: 0.94rem;
            line-height: 1.42;
            margin: 0.35rem 0 0;
        }

        .section-label {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 0.75rem;
            font-weight: 800;
            justify-content: space-between;
            margin: 0.95rem 0 0.48rem;
            text-transform: uppercase;
        }

        .section-label span:last-child {
            background: var(--tint);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink);
            padding: 0.18rem 0.48rem;
            text-transform: none;
        }

        .metric-grid {
            display: grid;
            gap: 0.7rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin-bottom: 0.85rem;
        }

        .metric-card {
            background: rgba(255, 253, 250, 0.64);
            border: 1px solid var(--line);
            border-radius: 8px;
            min-height: 5.8rem;
            padding: 0.8rem;
        }

        .metric-card span {
            color: var(--muted);
            display: block;
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .metric-card strong {
            color: var(--ink);
            display: block;
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.1;
            margin-top: 0.4rem;
        }

        .table-panel {
            margin-top: 0.95rem;
            padding: 0;
            overflow: hidden;
        }

        .table-panel .panel-title {
            margin: 0;
            padding: 1rem 1rem 0.35rem;
        }

        .table-wrap {
            overflow-x: auto;
            width: 100%;
        }

        .va-table {
            border-collapse: collapse;
            min-width: 920px;
            width: 100%;
        }

        .va-table th {
            border-bottom: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 850;
            padding: 0.75rem 0.85rem;
            text-align: left;
            text-transform: uppercase;
        }

        .va-table td {
            border-bottom: 1px solid rgba(234, 214, 207, 0.72);
            color: var(--ink);
            font-size: 0.84rem;
            font-weight: 700;
            padding: 0.85rem;
            vertical-align: middle;
            white-space: nowrap;
        }

        .va-table td strong {
            color: var(--ink);
            display: block;
            font-size: 0.9rem;
            font-weight: 850;
        }

        .va-table td span:not(.risk-badge) {
            color: var(--muted);
            display: block;
            font-size: 0.74rem;
            font-weight: 700;
            margin-top: 0.08rem;
        }

        .risk-badge {
            border: 1px solid;
            border-radius: 999px;
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 850;
            padding: 0.22rem 0.48rem;
            white-space: nowrap;
        }

        .muted-badge {
            background: var(--tint);
            border-color: var(--line);
            color: var(--muted);
        }

        .empty-state {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            padding: 0.2rem 1rem 1rem;
        }

        .guide-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.85rem 0;
        }

        .guide-card {
            min-height: 9.4rem;
        }

        .guide-card small {
            color: var(--accent);
            display: block;
            font-size: 0.75rem;
            font-weight: 850;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }

        .guide-card strong {
            color: var(--ink);
            display: block;
            font-size: 1.1rem;
            font-weight: 850;
            line-height: 1.15;
            margin-bottom: 0.45rem;
        }

        .guide-card span {
            color: var(--muted);
            display: block;
            font-size: 0.86rem;
            line-height: 1.4;
        }

        .action-panel {
            margin-top: 0.85rem;
        }

        .account-status {
            align-items: center;
            display: flex;
            gap: 0.5rem;
            justify-content: space-between;
        }

        .status-dot {
            background: var(--green);
            border-radius: 999px;
            display: inline-block;
            height: 0.55rem;
            width: 0.55rem;
        }

        .field-label {
            align-items: flex-start;
            color: var(--ink);
            display: flex;
            gap: 0.55rem;
            justify-content: space-between;
            margin-bottom: 0.16rem;
        }

        .field-label strong {
            color: var(--ink);
            font-size: 0.92rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .unit-badge {
            background: var(--tint);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink);
            flex: 0 0 auto;
            font-size: 0.7rem;
            font-weight: 800;
            padding: 0.15rem 0.42rem;
            white-space: nowrap;
        }

        .field-hint {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.32;
            margin-bottom: 0.45rem;
        }

        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stPasswordInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            box-shadow: 0 1px 0 rgba(74, 50, 56, 0.04) !important;
            color: var(--ink) !important;
            border-radius: 6px !important;
            min-height: 2.65rem !important;
        }

        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stPasswordInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(190, 109, 119, 0.16) !important;
            outline: none !important;
        }

        [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
            border-color: var(--accent) !important;
        }

        [data-testid="stNumberInput"] button {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #fff !important;
            min-height: 2.65rem !important;
            min-width: 2.65rem !important;
        }

        div[data-testid="stFormSubmitButton"] > button,
        div.stButton > button,
        [data-testid="stDownloadButton"] > button {
            background: var(--white) !important;
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            color: var(--ink) !important;
            font-weight: 800 !important;
            min-height: 2.5rem !important;
            padding: 0.5rem 0.9rem !important;
        }

        button[kind="primary"],
        button[kind="primaryFormSubmit"],
        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
        div[data-testid="stDownloadButton"] > button[kind="primary"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            box-shadow: 0 9px 22px rgba(190, 109, 119, 0.24) !important;
            color: #ffffff !important;
        }

        button[kind="primary"] p, button[kind="primaryFormSubmit"] p {
            color: #ffffff !important;
        }

        div[data-testid="stFormSubmitButton"] > button:hover,
        div.stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover {
            background: var(--tint) !important;
            border-color: var(--accent) !important;
            color: var(--ink) !important;
        }

        button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
            background: var(--accent-dark) !important;
            border-color: var(--accent-dark) !important;
            color: #ffffff !important;
        }

        div[data-testid="stFormSubmitButton"] > button:focus,
        div.stButton > button:focus,
        [data-testid="stDownloadButton"] > button:focus {
            box-shadow: 0 0 0 3px rgba(190, 109, 119, 0.22) !important;
            outline: none !important;
        }

        .risk-panel {
            background: rgba(255, 250, 245, 0.92);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 1rem;
        }

        .risk-top {
            align-items: center;
            display: grid;
            gap: 1rem;
            grid-template-columns: minmax(0, 1fr) auto;
        }

        .risk-status {
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--muted);
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
            padding: 0.22rem 0.5rem;
            text-transform: uppercase;
        }

        .risk-title {
            color: var(--ink);
            font-size: clamp(1.9rem, 4vw, 3rem);
            font-weight: 850;
            line-height: 1;
            margin: 0;
        }

        .risk-summary {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0.5rem 0 0;
        }

        .score-ring {
            align-items: center;
            background: conic-gradient(var(--risk-color) var(--score), var(--track) 0);
            border-radius: 999px;
            display: flex;
            height: 8.5rem;
            justify-content: center;
            width: 8.5rem;
        }

        .score-core {
            align-items: center;
            background: var(--panel);
            border-radius: 999px;
            display: flex;
            flex-direction: column;
            height: 6.2rem;
            justify-content: center;
            width: 6.2rem;
        }

        .score-core strong {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1;
        }

        .score-core span {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 800;
            margin-top: 0.2rem;
            text-transform: uppercase;
        }

        .prob-row, .factor-row, .queue-row {
            align-items: center;
            background: rgba(255, 253, 250, 0.62);
            border: 1px solid var(--line);
            border-radius: 8px;
            display: grid;
            gap: 0.65rem;
            margin: 0.48rem 0;
            padding: 0.68rem 0.74rem;
        }

        .prob-row {
            grid-template-columns: 8.2rem minmax(0, 1fr) 3.5rem;
        }

        .prob-label, .factor-main strong, .queue-row strong {
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 800;
        }

        .prob-track {
            background: var(--track);
            border-radius: 999px;
            height: 0.65rem;
            overflow: hidden;
        }

        .prob-fill {
            border-radius: 999px;
            height: 100%;
        }

        .prob-value, .factor-score {
            color: var(--ink);
            font-variant-numeric: tabular-nums;
            font-weight: 850;
            text-align: right;
            white-space: nowrap;
        }

        .factor-row {
            border-left: 4px solid var(--factor-color);
            grid-template-columns: minmax(0, 1fr) auto;
        }

        .factor-main span, .queue-row span {
            color: var(--muted);
            display: block;
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.12rem;
        }

        .bar-row {
            align-items: center;
            display: grid;
            gap: 0.6rem;
            grid-template-columns: 8rem minmax(0, 1fr) 3rem;
            margin: 0.58rem 0;
        }

        .bar-row span {
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 800;
        }

        .bar-track {
            background: var(--track);
            border-radius: 999px;
            height: 0.78rem;
            overflow: hidden;
        }

        .bar-fill {
            border-radius: 999px;
            height: 100%;
        }

        .clinical-note {
            background: rgba(234, 214, 207, 0.34);
            border: 1px solid var(--line);
            border-radius: 8px;
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.42;
            margin-top: 0.8rem;
            padding: 0.68rem 0.78rem;
        }

        /* ------ Админ-панель ------ */
        .admin-hero {
            background:
                radial-gradient(circle at 90% 4%, rgba(233, 174, 177, 0.34), transparent 18rem),
                linear-gradient(135deg, var(--nav), var(--nav-soft));
            border-radius: 12px;
            box-shadow: var(--shadow);
            margin: 0.2rem 0 0.95rem;
            overflow: hidden;
            padding: 1.15rem 1.25rem;
        }

        .admin-hero .entry-kicker {
            color: var(--cyan);
        }

        .admin-hero h1 {
            color: #ffffff;
            font-size: clamp(1.5rem, 2.6vw, 2.15rem);
            font-weight: 850;
            line-height: 1.06;
            margin: 0.42rem 0 0.4rem;
        }

        .admin-hero p {
            color: rgba(255, 255, 255, 0.74);
            font-size: 0.92rem;
            line-height: 1.45;
            margin: 0;
            max-width: 46rem;
        }

        .admin-hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.95rem;
        }

        .admin-hero-tags span {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            color: rgba(255, 255, 255, 0.88);
            font-size: 0.74rem;
            font-weight: 800;
            padding: 0.3rem 0.62rem;
        }

        [data-testid="stTabs"] [role="tablist"] {
            background: rgba(255, 250, 245, 0.84);
            border: 1px solid var(--line);
            border-radius: 10px;
            gap: 0.25rem;
            padding: 0.32rem;
        }

        [data-testid="stTabs"] [data-testid="stTab"],
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 8px;
            color: var(--muted);
            font-weight: 800;
            padding: 0.42rem 0.85rem;
        }
        [data-testid="stTabs"] [data-testid="stTab"]:not([aria-selected="true"]):hover,
        [data-testid="stTabs"] [data-baseweb="tab"]:not([aria-selected="true"]):hover {
            background: rgba(233, 174, 177, 0.18);
        }

        [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background: var(--accent);
            color: #ffffff !important;
        }

        [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p,
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p {
            color: #ffffff !important;
            font-weight: 800;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"],
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none;
        }

        .tab-note {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.42;
            margin: 0.7rem 0 0.1rem;
        }

        /* ================== Оболочка кабинета: сайдбар и топбар ================== */
        .va-icon {
            background: currentColor;
            display: inline-block;
            height: var(--icon-size, 1.05rem);
            width: var(--icon-size, 1.05rem);
            -webkit-mask-position: center;
            -webkit-mask-repeat: no-repeat;
            -webkit-mask-size: contain;
            mask-position: center;
            mask-repeat: no-repeat;
            mask-size: contain;
        }

        .va-icon-logo { --icon-size: 1.3rem; -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 12h4l3-7 4 14 3-7h6'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M2 12h4l3-7 4 14 3-7h6'/%3E%3C/svg%3E"); }
        .va-icon-grid { -webkit-mask-image: var(--icon-grid); mask-image: var(--icon-grid); }
        .va-icon-activity { -webkit-mask-image: var(--icon-plus); mask-image: var(--icon-plus); }
        .va-icon-users { -webkit-mask-image: var(--icon-users); mask-image: var(--icon-users); }
        .va-icon-chart { -webkit-mask-image: var(--icon-chart); mask-image: var(--icon-chart); }
        .va-icon-gear { -webkit-mask-image: var(--icon-gear); mask-image: var(--icon-gear); }
        .va-icon-shield { -webkit-mask-image: var(--icon-shield); mask-image: var(--icon-shield); }
        .va-icon-logout { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 5h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-3'/%3E%3Cpath d='M10 15l-3-3 3-3M7 12h8'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 5h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-3'/%3E%3Cpath d='M10 15l-3-3 3-3M7 12h8'/%3E%3C/svg%3E"); }
        .va-icon-search { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='6.5'/%3E%3Cpath d='M16 16l4.5 4.5'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round'%3E%3Ccircle cx='11' cy='11' r='6.5'/%3E%3Cpath d='M16 16l4.5 4.5'/%3E%3C/svg%3E"); }
        .va-icon-bell { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M6.5 10a5.5 5.5 0 0 1 11 0c0 4 1.5 5.5 1.5 5.5H5S6.5 14 6.5 10z'/%3E%3Cpath d='M10 18.5a2.2 2.2 0 0 0 4 0'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M6.5 10a5.5 5.5 0 0 1 11 0c0 4 1.5 5.5 1.5 5.5H5S6.5 14 6.5 10z'/%3E%3Cpath d='M10 18.5a2.2 2.2 0 0 0 4 0'/%3E%3C/svg%3E"); }
        .va-icon-clock { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='8.4'/%3E%3Cpath d='M12 7.6V12l3 1.9'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Ccircle cx='12' cy='12' r='8.4'/%3E%3Cpath d='M12 7.6V12l3 1.9'/%3E%3C/svg%3E"); }
        .va-icon-alert { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M12 4.5l8.5 15h-17z'/%3E%3Cpath d='M12 10v4.2M12 17.2h.01'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M12 4.5l8.5 15h-17z'/%3E%3Cpath d='M12 10v4.2M12 17.2h.01'/%3E%3C/svg%3E"); }
        .va-icon-file { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 3.5V8h4.5M8.8 12.5h6.4M8.8 16h4.2'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M14 3.5H7.5A2 2 0 0 0 5.5 5.5v13a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V8z'/%3E%3Cpath d='M14 3.5V8h4.5M8.8 12.5h6.4M8.8 16h4.2'/%3E%3C/svg%3E"); }
        .va-icon-percent { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M6 18L18 6'/%3E%3Ccircle cx='7.5' cy='7.5' r='2.2'/%3E%3Ccircle cx='16.5' cy='16.5' r='2.2'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round'%3E%3Cpath d='M6 18L18 6'/%3E%3Ccircle cx='7.5' cy='7.5' r='2.2'/%3E%3Ccircle cx='16.5' cy='16.5' r='2.2'/%3E%3C/svg%3E"); }
        .va-icon-back { -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M19 12H5M11 6l-6 6 6 6'/%3E%3C/svg%3E"); mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M19 12H5M11 6l-6 6 6 6'/%3E%3C/svg%3E"); }

        .side-brand {
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.10);
            display: flex;
            gap: 0.68rem;
            margin-bottom: 0.55rem;
            padding: 0.15rem 0.35rem 1rem;
        }

        .side-brand-mark {
            align-items: center;
            background: var(--accent);
            border-radius: 10px;
            color: #ffffff !important;
            display: flex;
            flex: 0 0 2.45rem;
            height: 2.45rem;
            justify-content: center;
            width: 2.45rem;
        }

        .side-brand-name {
            color: #ffffff;
            display: block;
            font-size: 1.02rem;
            font-weight: 850;
            line-height: 1.08;
        }

        .side-brand-sub {
            color: rgba(255, 255, 255, 0.55);
            display: block;
            font-size: 0.72rem;
            margin-top: 0.08rem;
        }

        .side-role {
            align-items: center;
            background: rgba(255, 255, 255, 0.09);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            color: #ffffff;
            display: flex;
            font-size: 0.86rem;
            font-weight: 850;
            gap: 0.45rem;
            justify-content: space-between;
            margin-bottom: 0.35rem;
            padding: 0.58rem 0.72rem;
        }

        .side-role span:last-child {
            color: rgba(255, 255, 255, 0.55);
            font-size: 0.72rem;
            font-weight: 700;
        }

        .side-nav-label {
            color: rgba(255, 255, 255, 0.42);
            font-size: 0.65rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            margin: 0.85rem 0.3rem 0.35rem;
            text-transform: uppercase;
        }

        section[data-testid="stSidebar"] div.stButton > button {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            box-shadow: none !important;
            color: rgba(255, 255, 255, 0.74) !important;
            font-weight: 800 !important;
            justify-content: flex-start !important;
            min-height: 2.45rem !important;
            padding: 0.5rem 0.62rem !important;
            text-align: left !important;
        }

        section[data-testid="stSidebar"] div.stButton > button > div {
            display: flex !important;
            flex: 1 1 auto;
            justify-content: flex-start !important;
            text-align: left !important;
        }

        section[data-testid="stSidebar"] div.stButton > button p {
            font-size: 0.92rem !important;
            font-weight: 800 !important;
            text-align: left !important;
            width: 100% !important;
        }

        [data-testid="stSidebarHeader"] { display: none !important; }

        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            display: flex !important;
            flex-direction: column !important;
            min-height: calc(100vh - 3rem) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div {
            display: flex !important;
            flex: 1 1 auto;
            flex-direction: column !important;
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]
        [data-testid="stElementContainer"]:has(.side-user) {
            margin-top: auto;
        }

        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(255, 255, 255, 0.09) !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] .st-key-nav_dashboard button::before { content: ""; }
        section[data-testid="stSidebar"] [class*="st-key-nav_"] button {
            display: flex !important;
            gap: 0.6rem !important;
        }

        section[data-testid="stSidebar"] [class*="st-key-nav_"] button::before {
            background: currentColor;
            content: "";
            display: inline-block;
            flex: 0 0 1.05rem;
            height: 1.05rem;
            -webkit-mask-position: center;
            -webkit-mask-repeat: no-repeat;
            -webkit-mask-size: contain;
            mask-position: center;
            mask-repeat: no-repeat;
            mask-size: contain;
        }

        section[data-testid="stSidebar"] .st-key-nav_dashboard button::before { -webkit-mask-image: var(--icon-grid); mask-image: var(--icon-grid); }
        section[data-testid="stSidebar"] .st-key-nav_intake button::before { -webkit-mask-image: var(--icon-plus); mask-image: var(--icon-plus); }
        section[data-testid="stSidebar"] .st-key-nav_patients button::before { -webkit-mask-image: var(--icon-users); mask-image: var(--icon-users); }
        section[data-testid="stSidebar"] .st-key-nav_reports button::before { -webkit-mask-image: var(--icon-chart); mask-image: var(--icon-chart); }
        section[data-testid="stSidebar"] .st-key-nav_settings button::before { -webkit-mask-image: var(--icon-gear); mask-image: var(--icon-gear); }
        section[data-testid="stSidebar"] .st-key-nav_admin button::before { -webkit-mask-image: var(--icon-shield); mask-image: var(--icon-shield); }

        .side-user {
            align-items: center;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.11);
            border-radius: 10px;
            display: flex;
            gap: 0.6rem;
            margin-top: auto;
            padding: 0.6rem 0.65rem;
        }

        .side-user strong {
            color: #ffffff;
            display: block;
            font-size: 0.86rem;
            font-weight: 850;
            line-height: 1.15;
        }

        .side-user span {
            color: rgba(255, 255, 255, 0.55);
            display: block;
            font-size: 0.72rem;
            margin-top: 0.08rem;
        }

        .avatar {
            align-items: center;
            background: rgba(233, 174, 177, 0.24);
            border-radius: 999px;
            color: #ffffff;
            display: flex;
            flex: 0 0 2.25rem;
            font-size: 0.78rem;
            font-weight: 850;
            height: 2.25rem;
            justify-content: center;
            width: 2.25rem;
        }

        .avatar-sm {
            flex: 0 0 1.85rem;
            font-size: 0.68rem;
            height: 1.85rem;
            width: 1.85rem;
        }

        .avatar-light {
            background: var(--tint);
            color: var(--ink);
        }

        /* Топбар: хлебные крошки, поиск, уведомления, аватар */
        .topbar {
            align-items: center;
            background: rgba(255, 253, 250, 0.82);
            border: 1px solid var(--line);
            border-radius: 12px;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1.05rem;
            padding: 0.48rem 0.7rem;
        }

        .breadcrumbs {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 0.83rem;
            font-weight: 700;
            gap: 0.4rem;
            white-space: nowrap;
        }

        .breadcrumbs b { color: var(--ink); font-weight: 850; }
        .breadcrumbs i { font-style: normal; opacity: 0.45; }

        .topbar-right {
            align-items: center;
            display: flex;
            gap: 0.5rem;
        }

        .bell {
            align-items: center;
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--muted);
            display: flex;
            height: 2.15rem;
            justify-content: center;
            position: relative;
            width: 2.15rem;
        }

        .bell::after {
            background: var(--high);
            border: 2px solid var(--white);
            border-radius: 999px;
            content: "";
            height: 0.5rem;
            position: absolute;
            right: 0.32rem;
            top: 0.32rem;
            width: 0.5rem;
        }

        /* Заголовок страницы */
        .page-head { margin: 0.1rem 0 1rem; }

        .page-head h1 {
            color: var(--ink);
            font-size: clamp(1.55rem, 2.6vw, 2.1rem);
            font-weight: 850;
            line-height: 1.06;
            margin: 0;
        }

        .page-head p {
            color: var(--muted);
            font-size: 0.92rem;
            margin: 0.32rem 0 0;
        }

        /* KPI-карточки */
        .kpi-grid {
            display: grid;
            gap: 0.72rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin-bottom: 0.95rem;
        }

        .kpi-card {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 10px 26px rgba(95, 59, 65, 0.06);
            padding: 0.85rem 0.9rem;
        }

        .kpi-top { align-items: center; display: flex; justify-content: space-between; }

        .kpi-icon {
            align-items: center;
            background: var(--tint);
            border-radius: 10px;
            color: var(--accent);
            display: flex;
            height: 1.95rem;
            justify-content: center;
            width: 1.95rem;
        }

        .kpi-delta {
            background: var(--tint);
            border-radius: 999px;
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 850;
            padding: 0.16rem 0.45rem;
        }

        .kpi-delta.attention { background: rgba(188, 97, 124, 0.16); color: var(--high); }

        .kpi-value {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1.08;
            margin-top: 0.62rem;
        }

        .kpi-label { color: var(--muted); font-size: 0.76rem; font-weight: 700; margin-top: 0.14rem; }

        /* Карточки-панели */
        .card-head {
            align-items: flex-start;
            display: flex;
            gap: 0.8rem;
            justify-content: space-between;
            margin-bottom: 0.7rem;
        }

        .card-head h3 { color: var(--ink); font-size: 1.02rem; font-weight: 850; line-height: 1.2; margin: 0; }
        .card-head .card-sub { color: var(--muted); display: block; font-size: 0.78rem; font-weight: 700; margin-top: 0.16rem; }
        .card-link { color: var(--accent); font-size: 0.8rem; font-weight: 850; white-space: nowrap; }

        /* Чипы-фильтры */
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }

        .chip {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--ink);
            font-size: 0.78rem;
            font-weight: 800;
            padding: 0.34rem 0.72rem;
            white-space: nowrap;
        }

        .chip.active { background: var(--accent); border-color: var(--accent); color: #ffffff; }

        /* Строка статистики (карта пациента) */
        .stat-row {
            border-top: 1px solid var(--line);
            display: grid;
            gap: 0.6rem;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            margin-top: 0.9rem;
            padding-top: 0.85rem;
        }

        .stat-row span {
            color: var(--muted);
            display: block;
            font-size: 0.66rem;
            font-weight: 850;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .stat-row strong { color: var(--ink); display: block; font-size: 0.98rem; font-weight: 850; margin-top: 0.22rem; }

        /* Секции форм */
        .form-section {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.1em;
            margin: 1.05rem 0 0.65rem;
            padding-top: 0.85rem;
            text-transform: uppercase;
        }

        .form-section:first-child { border-top: 0; margin-top: 0; padding-top: 0; }

        .hint-line { color: var(--muted); font-size: 0.82rem; font-weight: 700; margin: 0.35rem 0 0.9rem; }

        .progress-line {
            background: var(--line);
            border-radius: 999px;
            height: 0.3rem;
            overflow: hidden;
        }

        .progress-line i { background: var(--nav); display: block; height: 100%; }

        .progress-caption {
            color: var(--muted);
            font-size: 0.74rem;
            font-weight: 800;
            margin-top: 0.35rem;
            text-align: right;
        }

        /* Пагинация */
        .pager {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 0.8rem;
            font-weight: 700;
            justify-content: space-between;
            margin-top: 0.5rem;
        }

        .pager-pages { align-items: center; display: flex; gap: 0.3rem; }

        .pager-pages span {
            align-items: center;
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 9px;
            color: var(--ink);
            display: flex;
            font-weight: 850;
            height: 1.9rem;
            justify-content: center;
            width: 1.9rem;
        }

        .pager-pages span.active { background: var(--accent); border-color: var(--accent); color: #ffffff; }

        /* Вкладки в админ-панели */
        .tab-note { color: var(--muted); font-size: 0.82rem; line-height: 1.42; margin: 0.7rem 0 0.1rem; }

        /* Карточка топбара и таблицы-строки */
        [data-testid="stHorizontalBlock"]:has(.topbar-left) {
            align-items: center;
            background: rgba(255, 253, 250, 0.82);
            border: 1px solid var(--line);
            border-radius: 12px;
            margin-bottom: 1.05rem;
            padding: 0.45rem 0.7rem;
        }

        .cell-head {
            color: var(--muted);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .cell-strong { color: var(--ink); font-size: 0.9rem; font-weight: 850; }
        .cell-text { color: var(--ink); font-size: 0.86rem; font-weight: 700; }
        .cell-muted { color: var(--muted); font-size: 0.8rem; font-weight: 700; }

        [class*="st-key-row_"] { border-bottom: 1px solid var(--line); padding: 0.4rem 0; }
        [class*="st-key-row_"]:hover { background: rgba(255, 240, 237, 0.55); }
        [class*="st-key-head_"] { border-bottom: 1px solid var(--line); padding-bottom: 0.4rem; }

        [class*="st-key-row_"] div.stButton > button,
        [class*="st-key-head_"] div.stButton > button {
            background: transparent !important;
            border-color: transparent !important;
            color: var(--accent) !important;
            font-size: 0.8rem !important;
            min-height: 2rem !important;
            padding: 0.25rem 0.5rem !important;
        }

        [class*="st-key-row_"] div.stButton > button:hover {
            background: var(--tint) !important;
            border-color: var(--line) !important;
        }

        [class*="st-key-row_"] div.stButton > button p { font-size: 0.8rem !important; font-weight: 850 !important; }

        .st-key-nav_logout button { color: var(--high) !important; }

        .st-key-nav_logout button::before {
            background: currentColor;
            content: "";
            display: inline-block;
            flex: 0 0 1.05rem;
            height: 1.05rem;
            -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 5h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-3'/%3E%3Cpath d='M10 15l-3-3 3-3M7 12h8'/%3E%3C/svg%3E");
            -webkit-mask-position: center;
            -webkit-mask-repeat: no-repeat;
            -webkit-mask-size: contain;
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M15 5h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-3'/%3E%3Cpath d='M10 15l-3-3 3-3M7 12h8'/%3E%3C/svg%3E");
            mask-position: center;
            mask-repeat: no-repeat;
            mask-size: contain;
        }

        /* Активный пункт меню */
        section[data-testid="stSidebar"]:has(.nav-active-dashboard) .st-key-nav_dashboard button,
        section[data-testid="stSidebar"]:has(.nav-active-intake) .st-key-nav_intake button,
        section[data-testid="stSidebar"]:has(.nav-active-patients) .st-key-nav_patients button,
        section[data-testid="stSidebar"]:has(.nav-active-reports) .st-key-nav_reports button,
        section[data-testid="stSidebar"]:has(.nav-active-settings) .st-key-nav_settings button,
        section[data-testid="stSidebar"]:has(.nav-active-admin) .st-key-nav_admin button {
            background: rgba(255, 255, 255, 0.13) !important;
            border-color: rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
        }

        /* Чипы-фильтры (segmented control) */
        [data-testid="stSegmentedControl"] div[role="radiogroup"],
        [data-testid="stButtonGroup"] div[role="radiogroup"] {
            background: transparent;
            border: 0;
            gap: 0.4rem;
        }

        [data-testid="stSegmentedControl"] button,
        [data-testid="stSegmentedControl"] [role="radio"],
        [data-testid="stButtonGroup"] button,
        [data-testid="stButtonGroup"] [role="radio"] {
            background: var(--white) !important;
            border: 1px solid var(--line) !important;
            border-radius: 999px !important;
            color: var(--ink) !important;
            font-size: 0.78rem !important;
            font-weight: 800 !important;
            min-height: 2.1rem !important;
            padding: 0.3rem 0.75rem !important;
        }

        [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"],
        [data-testid="stButtonGroup"] button[aria-checked="true"],
        [data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #ffffff !important;
        }

        [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] p,
        [data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] p {
            color: #ffffff !important;
        }

        [data-testid="stSegmentedControl"] p,
        [data-testid="stButtonGroup"] [role="radio"] p {
            font-size: 0.78rem !important;
            font-weight: 800 !important;
        }

        /* Языковые пилюли */
        .lang-row { display: flex; flex-wrap: wrap; gap: 0.45rem; }

        .lang-row span {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 10px;
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 800;
            padding: 0.42rem 0.85rem;
        }

        .lang-row span.active { background: var(--accent); border-color: var(--accent); color: #ffffff; }
        .lang-row span.off { color: var(--muted); opacity: 0.62; }

        /* Карточки-контейнеры (st.container(border=True)) */
        [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
            background: var(--white);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: 0 12px 30px rgba(95, 59, 65, 0.06);
            margin-bottom: 0.85rem;
            padding: 0.95rem 1.05rem;
        }

        /* Карточка с графиком обрезает содержимое по своему скруглению */
        [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(.vega-embed) {
            overflow: hidden;
        }

        /* Графики не рисуют собственную белую подложку и не выходят за скруглённую маску карточки */
        .vega-embed,
        .vega-embed > div,
        .vega-embed svg {
            background: transparent !important;
        }

        .vega-embed svg rect.background {
            fill: transparent !important;
        }

        [data-testid="stMain"] [data-testid="stElementContainer"]:has(.vega-embed) {
            border-radius: 12px;
            max-width: 100%;
            overflow: hidden;
        }

        /* Вложенные карточки без двойной тени */
        [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]
        [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
            box-shadow: none;
        }

        /* Старые пустые обёртки панелей не должны рисоваться */
        .panel:empty, .panel:blank { display: none !important; }

        /* Кнопки-ссылки */
        [class*="st-key-link_"] div.stButton > button,
        [class*="st-key-link_"] button {
            background: transparent !important;
            border-color: transparent !important;
            box-shadow: none !important;
            color: var(--accent) !important;
            font-size: 0.82rem !important;
            font-weight: 850 !important;
            min-height: 1.9rem !important;
            padding: 0.2rem 0.35rem !important;
        }

        [class*="st-key-link_"] button:hover {
            background: var(--tint) !important;
            border-color: var(--line) !important;
        }

        [class*="st-key-link_"] button p { font-size: 0.82rem !important; font-weight: 850 !important; }

        /* Кликабельные KPI-карточки */
        [data-testid="stColumn"]:has(.kpi-card) { position: relative; }

        [class*="st-key-kpihit_"] {
            inset: 0;
            position: absolute;
            z-index: 3;
        }

        [class*="st-key-kpihit_"] button {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            height: 100% !important;
            min-height: 100% !important;
            width: 100% !important;
        }

        [class*="st-key-kpihit_"] button p { color: transparent !important; font-size: 0.01rem !important; }

        .kpi-card-link { cursor: pointer; }

        [data-testid="stColumn"]:has(.kpi-card):hover .kpi-card {
            border-color: var(--accent);
            box-shadow: 0 12px 28px rgba(190, 109, 119, 0.18);
        }

        /* Поповеры в топбаре */
        [class*="st-key-bell"] button,
        [class*="st-key-avatar"] button {
            background: var(--white) !important;
            border: 1px solid var(--line) !important;
            border-radius: 999px !important;
            box-shadow: none !important;
            color: var(--muted) !important;
            height: 2.15rem !important;
            min-height: 2.15rem !important;
            padding: 0 !important;
            width: 2.15rem !important;
        }

        [class*="st-key-bell"] button p,
        [class*="st-key-avatar"] button p { font-size: 0.78rem !important; font-weight: 850 !important; }

        [class*="st-key-avatar"] button { color: var(--ink) !important; }

        .popover-title { color: var(--ink); font-size: 0.95rem; font-weight: 850; margin: 0 0 0.35rem; }

        .popover-note { color: var(--muted); font-size: 0.8rem; line-height: 1.4; margin: 0.3rem 0 0; }

        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-bottom: 0.4rem !important; }

        /* Вложенные элементы внутри карточек без двойных отступов */
        [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]
        [data-testid="stVerticalBlockBorderWrapper"] {
            box-shadow: none;
            margin-bottom: 0.6rem;
        }

        .crumb-current {
            color: var(--ink);
            font-size: 0.83rem;
            font-weight: 850;
            padding: 0.25rem 0.35rem;
            white-space: nowrap;
        }

        .crumb-sep {
            color: var(--muted);
            font-size: 0.83rem;
            opacity: 0.5;
            padding: 0.25rem 0.1rem;
        }

        .popover-row { border-bottom: 1px solid var(--line); padding: 0.4rem 0; }
        .popover-row strong { color: var(--ink); display: block; font-size: 0.82rem; font-weight: 800; }
        .popover-row span { color: var(--muted); display: block; font-size: 0.72rem; margin-top: 0.1rem; }

        [data-testid="stHorizontalBlock"]:has(.kpi-card) { margin-bottom: 0.95rem; }

        /* Окно входа — один статичный экран без прокрутки */
        .login-page-marker { display: none; }

        [data-testid="stApp"]:has(.login-page-marker),
        [data-testid="stAppViewContainer"]:has(.login-page-marker),
        [data-testid="stMain"]:has(.login-page-marker) {
            height: 100vh;
            /* auto, а не hidden: если на низком экране контент не влезет, он не обрежется */
            overflow: auto;
        }

        [data-testid="stMain"]:has(.login-page-marker) .block-container {
            display: flex;
            flex-direction: column;
            height: calc(100vh - 1.2rem);
            justify-content: center;
            overflow-x: hidden;
            padding-bottom: 0.6rem;
            padding-top: 0.6rem;
        }

        [data-testid="stMain"]:has(.login-page-marker) .block-container > [data-testid="stVerticalBlock"] {
            display: flex;
            flex: 1 1 auto;
            flex-direction: column;
        }

        [data-testid="stMain"]:has(.login-page-marker) [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        [data-testid="stMain"]:has(.login-page-marker) [data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
        }

        [data-testid="stMain"]:has(.login-page-marker) [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        /* Подсказки — в самый низ колонки */
        [data-testid="stMain"]:has(.login-page-marker) [data-testid="stElementContainer"]:has(.entry-tips) {
            margin-top: auto;
        }

        /* Карточка входа — по центру своей колонки */
        [data-testid="stMain"]:has(.login-page-marker) [data-testid="stColumn"]:has(.auth-form-heading) > [data-testid="stVerticalBlock"] {
            justify-content: center;
        }

        .entry-brand img {
            height: 2.4rem;
            margin-bottom: 0.45rem;
            max-width: 11rem;
        }

        .entry-tips { min-height: 7.4rem; }
        .entry-tip { top: 2.25rem; }

        /* Больше воздуха между полями форм */
        .field-label { margin-top: 1.15rem; }
        .field-hint { margin-bottom: 0.7rem; }

        [data-testid="stMain"] [data-testid="stNumberInput"],
        [data-testid="stMain"] [data-testid="stTextInput"],
        [data-testid="stMain"] [data-testid="stSelectbox"],
        [data-testid="stMain"] [data-testid="stTextArea"],
        [data-testid="stMain"] [data-testid="stFileUploader"] {
            margin-bottom: 0.6rem;
        }

        /* Окно входа складывается в колонку раньше, чтобы форма не уходила за экран */
        @media (max-width: 1100px) {
            [data-testid="stMain"]:has(.login-page-marker),
            [data-testid="stAppViewContainer"]:has(.login-page-marker) { height: auto; overflow: auto; }

            [data-testid="stMain"]:has(.login-page-marker) .block-container { height: auto; justify-content: flex-start; }

            [data-testid="stMain"]:has(.login-page-marker) [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }

            [data-testid="stMain"]:has(.login-page-marker) [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
            [data-testid="stMain"]:has(.login-page-marker) [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                min-width: 0 !important;
                width: 100% !important;
            }
        }

        @media (max-height: 660px) {
            [data-testid="stMain"]:has(.login-page-marker),
            [data-testid="stAppViewContainer"]:has(.login-page-marker) { height: auto; overflow: auto; }

            [data-testid="stMain"]:has(.login-page-marker) .block-container { height: auto; }
        }

        @media (max-width: 980px) {
            .kpi-grid, .stat-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .header-grid, .risk-top { grid-template-columns: 1fr; }
            .guide-grid { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }

        @media (max-width: 720px) {
            .block-container { padding: 0.85rem 0.85rem 1.8rem; }
            .kpi-grid { grid-template-columns: 1fr; }
            .metric-grid { grid-template-columns: 1fr; }
            .topbar { flex-wrap: wrap; }
            .prob-row { grid-template-columns: 6.8rem minmax(0, 1fr) 3.1rem; }
            .entry-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
            [data-testid="stHorizontalBlock"] > [data-testid="column"],
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                min-width: 0 !important;
                width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def secret_value(name: str, default: str = "") -> str:
    """Значение настройки: сначала st.secrets, потом переменные окружения."""
    value: object = ""
    try:
        value = st.secrets.get(name, "")  # type: ignore[union-attr]
    except Exception:
        value = ""
    if value in (None, ""):
        value = os.environ.get(name, "")
    text = str(value or "").strip()
    return text or default


def database_url() -> str:
    """DSN внешней БД; пустая строка — работаем на локальных CSV."""
    url = secret_value("DATABASE_URL") or secret_value("SUPABASE_DB_URL") or secret_value("TURSO_DATABASE_URL")
    if not url:
        return ""
    if url.startswith(("libsql://", "http://", "https://")):
        token = secret_value("TURSO_AUTH_TOKEN")
        url = "sqlite+libsql://" + url.split("://", 1)[1]
        if token:
            url += ("&" if "?" in url else "?") + f"authToken={token}"
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


@st.cache_resource(show_spinner=False)
def db_engine(url: str):
    from sqlalchemy import create_engine

    return create_engine(url, pool_pre_ping=True, future=True)


def engine_or_none():
    url = database_url()
    if not url:
        return None
    try:
        return db_engine(url)
    except Exception as error:  # нет драйвера или плохой DSN — остаёмся на CSV
        st.warning(f"Не удалось подключиться к внешней БД ({error}). Работаем на локальных CSV.")
        return None


_DB_TABLES_READY: set[str] = set()


def table_name(path: Path) -> str:
    name = Path(path).name
    return STORAGE_TABLES.get(name, Path(name).stem.replace("-", "_"))


def ensure_db_table(conn, table: str, columns: list[str]) -> None:
    if table in _DB_TABLES_READY:
        return
    from sqlalchemy import text

    definition = ", ".join(f'"{column}" TEXT' for column in columns)
    conn.execute(text(f'CREATE TABLE IF NOT EXISTS "{table}" ({definition})'))
    _DB_TABLES_READY.add(table)


def attachments_in_cloud() -> bool:
    return bool(secret_value("SUPABASE_URL") and secret_value("SUPABASE_SERVICE_KEY"))


def upload_attachment_to_cloud(relative_path: str, data: bytes) -> None:
    """Кладёт файл вложения в Supabase Storage (bucket из SUPABASE_BUCKET)."""
    import requests

    base = secret_value("SUPABASE_URL").rstrip("/")
    bucket = secret_value("SUPABASE_BUCKET", ATTACHMENT_BUCKET) or ATTACHMENT_BUCKET
    response = requests.post(
        f"{base}/storage/v1/object/{bucket}/{relative_path}",
        data=data,
        timeout=45,
        headers={
            "Authorization": f"Bearer {secret_value('SUPABASE_SERVICE_KEY')}",
            "Content-Type": "application/octet-stream",
            "x-upsert": "true",
        },
    )
    response.raise_for_status()


def ensure_store() -> None:
    engine = engine_or_none()
    if engine is not None:
        with engine.begin() as conn:
            ensure_db_table(conn, table_name(PATIENTS_PATH), PATIENT_COLUMNS)
            ensure_db_table(conn, table_name(VISITS_PATH), VISIT_COLUMNS)
            ensure_db_table(conn, table_name(ATTACHMENTS_PATH), ATTACHMENT_COLUMNS)
            ensure_db_table(conn, table_name(ACTIVITY_PATH), ACTIVITY_COLUMNS)
            ensure_db_table(conn, table_name(USERS_PATH), USER_COLUMNS)
        return

    CLINIC_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    if not PATIENTS_PATH.exists():
        pd.DataFrame(columns=PATIENT_COLUMNS).to_csv(PATIENTS_PATH, index=False)
    if not VISITS_PATH.exists():
        pd.DataFrame(columns=VISIT_COLUMNS).to_csv(VISITS_PATH, index=False)
    if not ATTACHMENTS_PATH.exists():
        pd.DataFrame(columns=ATTACHMENT_COLUMNS).to_csv(ATTACHMENTS_PATH, index=False)
    if not ACTIVITY_PATH.exists():
        pd.DataFrame(columns=ACTIVITY_COLUMNS).to_csv(ACTIVITY_PATH, index=False)
    if not USERS_PATH.exists():
        pd.DataFrame(columns=USER_COLUMNS).to_csv(USERS_PATH, index=False)


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    engine = engine_or_none()
    if engine is not None:
        table = table_name(path)
        with engine.begin() as conn:
            ensure_db_table(conn, table, columns)
        with engine.connect() as conn:
            frame = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        frame = frame.fillna("")
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        return frame[columns].astype(str)

    ensure_store()
    df = pd.read_csv(path, dtype=str).fillna("")
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns]


def write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    engine = engine_or_none()
    if engine is not None:
        from sqlalchemy import text

        table = table_name(path)
        payload = df[columns].fillna("").astype(str)
        with engine.begin() as conn:
            ensure_db_table(conn, table, columns)
            conn.execute(text(f'DELETE FROM "{table}"'))
            payload.to_sql(table, conn, if_exists="append", index=False)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(path, index=False, encoding="utf-8")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def session_expires_at() -> str:
    return (datetime.now() + timedelta(days=SESSION_TTL_DAYS)).strftime("%Y-%m-%d %H:%M:%S")


def parse_iso_datetime(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt, expected = password_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        120_000,
    ).hex()
    return secrets.compare_digest(digest, expected)


def seed_admin_row() -> dict[str, str]:
    return {
        "user_id": uuid.uuid4().hex[:10],
        "username": "admin",
        "password_hash": hash_password(DEFAULT_ADMIN_PASSWORD),
        "full_name": "Главный администратор",
        "role": "admin",
        "status": "active",
        "created_at": now_iso(),
        "last_login_at": "",
        "session_hash": "",
        "session_expires_at": "",
    }


def ensure_auth_store() -> None:
    """Гарантирует, что таблица пользователей существует и в ней есть админ."""
    if engine_or_none() is None:
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
    df = read_csv(USERS_PATH, USER_COLUMNS)
    changed = False
    for column in USER_COLUMNS:
        if column not in df.columns:
            df[column] = ""
            changed = True
    if df.empty:
        df = pd.DataFrame([seed_admin_row()], columns=USER_COLUMNS)
        changed = True
    if changed:
        write_users(df)


def users_df() -> pd.DataFrame:
    ensure_auth_store()
    return read_csv(USERS_PATH, USER_COLUMNS)


def write_users(df: pd.DataFrame) -> None:
    if engine_or_none() is None:
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(df, USERS_PATH, USER_COLUMNS)


def normalize_username(username: str) -> str:
    return username.strip().lower()


def is_valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9._-]{3,32}", username))


def bootstrap_credentials_visible() -> bool:
    users = users_df()
    if len(users) != 1:
        return False
    row = users.iloc[0]
    return row["username"] == "admin" and row["role"] == "admin" and row["last_login_at"] == ""


def authenticate(username: str, password: str) -> dict[str, str] | None:
    username = normalize_username(username)
    users = users_df()
    mask = users["username"].str.lower() == username
    if not mask.any():
        return None
    row_index = users.index[mask][0]
    row = users.loc[row_index]
    if row["status"] != "active":
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    users.loc[row_index, "last_login_at"] = now_iso()
    write_users(users)
    user = users.loc[row_index].to_dict()
    user.pop("password_hash", None)
    user.pop("session_hash", None)
    return {str(key): str(value) for key, value in user.items()}


def public_user(row: pd.Series) -> dict[str, str]:
    user = row.to_dict()
    user.pop("password_hash", None)
    user.pop("session_hash", None)
    return {str(key): str(value) for key, value in user.items()}


def issue_session_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    users = users_df()
    mask = users["user_id"] == user_id
    if mask.any():
        users.loc[mask, "session_hash"] = hash_session_token(token)
        users.loc[mask, "session_expires_at"] = session_expires_at()
        write_users(users)
    return token


def user_from_session_token(token: str) -> dict[str, str] | None:
    if not token:
        return None
    users = users_df()
    token_hash = hash_session_token(token)
    mask = users["session_hash"] == token_hash
    if not mask.any():
        return None
    row_index = users.index[mask][0]
    row = users.loc[row_index]
    if row["status"] != "active":
        return None
    expires_at = parse_iso_datetime(row.get("session_expires_at", ""))
    if expires_at is None or expires_at < datetime.now():
        users.loc[row_index, ["session_hash", "session_expires_at"]] = ""
        write_users(users)
        return None
    return public_user(row)


def query_session_token() -> str:
    value = st.query_params.get(SESSION_QUERY_PARAM, "")
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def set_browser_session(token: str) -> None:
    st.query_params[SESSION_QUERY_PARAM] = token


def clear_browser_session(user: dict[str, str] | None = None) -> None:
    if user and user.get("user_id"):
        users = users_df()
        mask = users["user_id"] == user["user_id"]
        if mask.any():
            users.loc[mask, ["session_hash", "session_expires_at"]] = ""
            write_users(users)
    if SESSION_QUERY_PARAM in st.query_params:
        del st.query_params[SESSION_QUERY_PARAM]


def current_user() -> dict[str, str] | None:
    user = st.session_state.get("auth_user")
    if isinstance(user, dict):
        return user
    token = query_session_token()
    user_from_token = user_from_session_token(token)
    if user_from_token:
        st.session_state["auth_user"] = user_from_token
        return user_from_token
    if token and SESSION_QUERY_PARAM in st.query_params:
        del st.query_params[SESSION_QUERY_PARAM]
    return None


def refresh_current_user(user: dict[str, str]) -> dict[str, str] | None:
    users = users_df()
    mask = users["username"].str.lower() == normalize_username(user.get("username", ""))
    if not mask.any():
        st.session_state.pop("auth_user", None)
        return None
    row = users.loc[users.index[mask][0]]
    if row["status"] != "active":
        st.session_state.pop("auth_user", None)
        return None
    refreshed = row.to_dict()
    refreshed.pop("password_hash", None)
    refreshed.pop("session_hash", None)
    refreshed = {str(key): str(value) for key, value in refreshed.items()}
    st.session_state["auth_user"] = refreshed
    return refreshed


def is_admin(user: dict[str, str] | None) -> bool:
    return bool(user and user.get("role") == "admin")


def create_user_account(username: str, full_name: str, role: str, status: str, password: str) -> tuple[bool, str]:
    username = normalize_username(username)
    if not is_valid_username(username):
        return False, "Логин: 3-32 символа, латиница, цифры, точка, дефис или подчёркивание."
    if len(password) < 6:
        return False, "Пароль должен быть не короче 6 символов."
    users = users_df()
    if username in users["username"].str.lower().tolist():
        return False, "Такой логин уже существует."
    row = {
        "user_id": uuid.uuid4().hex[:10],
        "username": username,
        "password_hash": hash_password(password),
        "full_name": full_name.strip() or username,
        "role": role,
        "status": status,
        "created_at": now_iso(),
        "last_login_at": "",
        "session_hash": "",
        "session_expires_at": "",
    }
    users = pd.concat([users, pd.DataFrame([row])], ignore_index=True)
    write_users(users)
    log_event(
        "account_created",
        target=username,
        details=f"Роль: {ROLE_LABELS.get(role, role)}, статус: {STATUS_LABELS.get(status, status)}",
    )
    return True, f"Аккаунт {username} создан."


def patients_df() -> pd.DataFrame:
    return read_csv(PATIENTS_PATH, PATIENT_COLUMNS)


def visits_df() -> pd.DataFrame:
    return read_csv(VISITS_PATH, VISIT_COLUMNS)


def attachments_df() -> pd.DataFrame:
    return read_csv(ATTACHMENTS_PATH, ATTACHMENT_COLUMNS)


def activity_df() -> pd.DataFrame:
    return read_csv(ACTIVITY_PATH, ACTIVITY_COLUMNS)


def log_event(action: str, target: str = "", details: str = "", actor: dict[str, str] | None = None) -> None:
    """Единый журнал действий: кто, что и когда сделал."""
    if actor is None:
        session_user = st.session_state.get("auth_user")
        actor = session_user if isinstance(session_user, dict) else {}
    row = {
        "event_id": uuid.uuid4().hex[:10],
        "event_at": now_iso(),
        "actor": str(actor.get("username") or "system"),
        "actor_role": ROLE_LABELS.get(str(actor.get("role", "")), str(actor.get("role") or "—")),
        "action": action,
        "target": target,
        "details": details,
    }
    frame = activity_df()
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    write_csv(frame, ACTIVITY_PATH, ACTIVITY_COLUMNS)


def next_patient_id(patients: pd.DataFrame) -> str:
    if patients.empty:
        return "PT-0001"
    ids = patients["patient_id"].astype(str).tolist()
    numbers = [int(match.group(1)) for value in ids if (match := re.match(r"PT-(\d+)$", value))]
    return f"PT-{(max(numbers) + 1 if numbers else len(ids) + 1):04d}"


def patient_label(row: pd.Series) -> str:
    initials = row.get("initials", "") or "без имени"
    age = row.get("age", "")
    week = row.get("gestational_week", "")
    detail = []
    if age:
        detail.append(f"{age} лет")
    if week:
        detail.append(f"{week} нед.")
    suffix = f" · {', '.join(detail)}" if detail else ""
    return f"{row['patient_id']} · {initials}{suffix}"


def append_patient(row: dict[str, object]) -> None:
    df = patients_df()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    write_csv(df, PATIENTS_PATH, PATIENT_COLUMNS)


def append_visit(row: dict[str, object]) -> None:
    df = visits_df()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    write_csv(df, VISITS_PATH, VISIT_COLUMNS)


def append_attachment(row: dict[str, object]) -> None:
    df = attachments_df()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    write_csv(df, ATTACHMENTS_PATH, ATTACHMENT_COLUMNS)


@st.cache_resource
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_model, _background: pd.DataFrame):
    return shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent")


def prediction_shap_values(explainer, patient: pd.DataFrame, class_index: int) -> np.ndarray:
    values = explainer.shap_values(patient)
    if isinstance(values, list):
        return np.asarray(values[class_index][0], dtype=float)
    values_array = np.asarray(values, dtype=float)
    if values_array.ndim == 3:
        return values_array[0, :, class_index]
    if values_array.ndim == 2:
        return values_array[0]
    raise ValueError(f"Unexpected SHAP values shape: {values_array.shape}")


def format_value(feature: str, value: float) -> str:
    meta = FEATURE_META[feature]
    rendered = str(int(round(value))) if meta["format"] == "%d" else f"{value:.1f}"
    return f"{rendered} {meta['unit']}"


def predict_risk(values: dict[str, float], bundle: dict) -> dict[str, object]:
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    class_names = bundle["class_names"]
    patient = pd.DataFrame([{feature: values[feature] for feature in feature_names}], columns=feature_names)
    probabilities = model.predict_proba(patient)[0]
    class_index = int(np.argmax(probabilities))
    explainer = load_explainer(model, bundle["shap_background"])
    shap_values = prediction_shap_values(explainer, patient, class_index)
    rows = sorted(
        zip(feature_names, patient.iloc[0].to_numpy(dtype=float), shap_values),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    top_feature = rows[0][0]
    return {
        "patient": patient,
        "probabilities": probabilities,
        "class_index": class_index,
        "class_name": class_names[class_index],
        "shap_values": shap_values,
        "top_factor": FEATURE_META[top_feature]["label"],
    }


def read_metrics_snapshot() -> dict:
    """Метрики модели для окна входа (файл может отсутствовать до обучения)."""
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def render_login_page() -> None:
    st.markdown('<div class="login-page-marker"></div>', unsafe_allow_html=True)
    tips_html = "".join(
        f'<div class="entry-tip" style="animation-delay:{index * ENTRY_TIP_SECONDS}s">'
        f"<strong>{escape(title)}</strong><span>{escape(text)}</span></div>"
        for index, (title, text) in enumerate(ENTRY_TIPS)
    )
    dots_html = "".join(
        f'<i style="animation-delay:{index * ENTRY_TIP_SECONDS}s"></i>' for index in range(len(ENTRY_TIPS))
    )

    left, right = st.columns([1.15, 0.85], gap="large", vertical_alignment="top")

    with left:
        st.markdown(
            f'<div class="entry-brand"><img src="{LOGO_URL}" alt="VascularAI" /></div>',
            unsafe_allow_html=True,
        )
        st.iframe(MODEL_VIEWER_URL, height=440, width="stretch", tab_index=-1)
        st.html(
            f'<div class="entry-tips"><small>Коротко о платформе</small>{tips_html}'
            f'<div class="entry-tip-dots">{dots_html}</div></div>'
        )

    with right:
        with st.form("login_form"):
            st.html(
                """
                <div class="auth-form-heading">
                    <div class="entry-kicker">Кабинет VascularAI</div>
                    <h2>Вход в систему</h2>
                    <p>Используйте аккаунт врача или администратора.</p>
                </div>
                """
            )
            username = st.text_input("Логин", placeholder="doctor01")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", width="stretch")
        st.html(
            f"""
            <div class="auth-safe-note">
                Вход сохраняется на {SESSION_TTL_DAYS} дней на этом устройстве. Выход из кабинета вручную завершит сессию.
            </div>
            """
        )
        if bootstrap_credentials_visible():
            st.info(f"Первый вход: логин `admin`, пароль `{DEFAULT_ADMIN_PASSWORD}`. После входа создайте реальные аккаунты врачей.")
        if submitted:
            user = authenticate(username, password)
            if user is None:
                st.error("Логин или пароль неверный, либо аккаунт отключён.")
                return
            token = issue_session_token(user["user_id"])
            set_browser_session(token)
            st.session_state["auth_user"] = user
            log_event(
                "login",
                target=user.get("username", ""),
                details=ROLE_LABELS.get(str(user.get("role", "")), "—"),
                actor=user,
            )
            st.rerun()


def logout() -> None:
    user = st.session_state.get("auth_user")
    if isinstance(user, dict):
        log_event("logout", target=user.get("username", ""), actor=user)
    clear_browser_session(user if isinstance(user, dict) else None)
    st.session_state.pop("auth_user", None)
    st.rerun()


NAV_SLUGS = {
    "Дашборд": "dashboard",
    "Новый расчёт": "intake",
    "Пациенты": "patients",
    "Отчёты": "reports",
    "Настройки": "settings",
    "Админ-панель": "admin",
}


def initials_of(name: str) -> str:
    parts = [part for part in re.split(r"\s+", (name or "").strip()) if part]
    if not parts:
        return "—"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()


def hex_to_rgba(color: str, alpha: float) -> str:
    value = (color or "").lstrip("#")
    if len(value) != 6:
        return f"rgba(189, 104, 117, {alpha})"
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def icon_html(name: str, size: str = "1.05rem") -> str:
    return f'<span class="va-icon {name}" style="--icon-size:{size};"></span>'


def safe_int(value: object, fallback: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return fallback


def field_label_html(label: str, unit: str = "", hint: str = "") -> str:
    unit_html = f'<span class="unit-badge">{escape(unit)}</span>' if unit else ""
    hint_html = f'<div class="field-hint">{escape(hint)}</div>' if hint else ""
    return f'<div class="field-label"><strong>{escape(label)}</strong>{unit_html}</div>{hint_html}'


def labeled_text_input(
    label: str,
    unit: str = "",
    hint: str = "",
    value: str = "",
    placeholder: str = "",
    key: str | None = None,
) -> str:
    st.markdown(field_label_html(label, unit, hint), unsafe_allow_html=True)
    return st.text_input(label, value=value, placeholder=placeholder, label_visibility="collapsed", key=key)


def labeled_number_input(
    label: str,
    unit: str = "",
    hint: str = "",
    value: int = 28,
    minimum: int = 1,
    maximum: int = 60,
    key: str | None = None,
) -> int:
    st.markdown(field_label_html(label, unit, hint), unsafe_allow_html=True)
    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=value,
        step=1,
        label_visibility="collapsed",
        key=key,
    )


def page_head(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-head"><h1>{escape(title)}</h1>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def card_head(title: str, subtitle: str = "", link: str = "") -> None:
    parts = [f'<div><h3>{escape(title)}</h3>']
    if subtitle:
        parts.append(f'<span class="card-sub">{escape(subtitle)}</span>')
    parts.append("</div>")
    if link:
        parts.append(f'<span class="card-link">{escape(link)}</span>')
    st.markdown(f'<div class="card-head">{"".join(parts)}</div>', unsafe_allow_html=True)


def kpi_cards(cards: list[tuple]) -> None:
    """cards: (иконка, подпись, значение, дельта, внимание[, действие]).

    Действие — словарь с ключами page / risk / patient: карточка становится кликабельной.
    """
    columns = st.columns(len(cards), vertical_alignment="top")
    for index, card in enumerate(cards):
        icon, label, value, delta, attention = card[:5]
        action = card[5] if len(card) > 5 else None
        delta_html = ""
        if delta:
            delta_class = "kpi-delta attention" if attention else "kpi-delta"
            delta_html = f'<span class="{delta_class}">{escape(delta)}</span>'
        with columns[index]:
            st.markdown(
                f'<div class="kpi-card{" kpi-card-link" if action else ""}">'
                f'<div class="kpi-top"><span class="kpi-icon">{icon_html(icon, "1.1rem")}</span>{delta_html}</div>'
                f'<div class="kpi-value">{escape(value)}</div>'
                f'<div class="kpi-label">{escape(label)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            if action and st.button("‎", key=f"kpihit_{index}", width="stretch", help=str(action.get("hint") or "Открыть")):
                if action.get("risk"):
                    st.session_state[PENDING_RISK_FILTER] = action["risk"]
                if action.get("patient"):
                    st.session_state[PATIENT_VIEW_KEY] = action["patient"]
                st.session_state[NAV_KEY] = action.get("page", "Дашборд")
                st.rerun()


def link_button(label: str, key: str, page: str, risk: str | None = None, patient_id: str | None = None) -> None:
    """Кнопка-ссылка: переход на раздел, при необходимости с фильтром или картой пациентки."""
    if st.button(label, key=f"link_{key}", width="stretch"):
        if risk:
            st.session_state[PENDING_RISK_FILTER] = risk
        if patient_id:
            st.session_state[PATIENT_VIEW_KEY] = patient_id
        st.session_state[NAV_KEY] = page
        st.rerun()


def render_topbar(user: dict[str, str], page: str, crumbs: list[tuple[str, str | None]] | None = None) -> str:
    """Верхняя панель: кликабельные крошки, быстрый поиск, уведомления, аватар."""
    role_label = ROLE_LABELS.get(str(user.get("role", "")), "Врач")
    trail = crumbs if crumbs else [(role_label, "Дашборд"), (page, None)]
    full_name = user.get("full_name") or user.get("username", "")
    username = str(user.get("username", ""))

    left, right = st.columns([1.45, 1], vertical_alignment="center")
    with left:
        widths: list[float] = []
        for index in range(len(trail)):
            widths.append(1.0)
            if index < len(trail) - 1:
                widths.append(0.12)
        crumb_columns = st.columns(widths, vertical_alignment="center")
        for index, (label, target) in enumerate(trail):
            with crumb_columns[index * 2]:
                if target:
                    if st.button(label, key=f"link_crumb_{index}_{target}", width="stretch"):
                        st.session_state[NAV_KEY] = target
                        if target != "Пациенты":
                            st.session_state.pop(PATIENT_VIEW_KEY, None)
                        st.rerun()
                else:
                    st.markdown(f'<div class="crumb-current">{escape(label)}</div>', unsafe_allow_html=True)
            if index < len(trail) - 1:
                with crumb_columns[index * 2 + 1]:
                    st.markdown('<div class="crumb-sep">›</div>', unsafe_allow_html=True)

    with right:
        search_col, bell_col, avatar_col = st.columns([1.8, 0.42, 0.42], vertical_alignment="center")
        with search_col:
            query = st.text_input(
                "Поиск",
                placeholder="Поиск по ID или ФИО...",
                label_visibility="collapsed",
                key="global_search",
            )
        with bell_col:
            with st.popover("‎", icon=":material/notifications:", key="bell_popover"):
                st.markdown('<div class="popover-title">Уведомления</div>', unsafe_allow_html=True)
                events = activity_df()
                own_events = events.loc[events["actor"] == username].tail(6).iloc[::-1]
                if own_events.empty:
                    st.markdown(
                        '<p class="popover-note">Событий пока нет. Здесь появятся ваши расчёты, карты пациенток и входы.</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    for _, event in own_events.iterrows():
                        action_label = ACTION_LABELS.get(str(event["action"]), str(event["action"]))
                        st.markdown(
                            f'<div class="popover-row"><strong>{escape(action_label)}</strong>'
                            f'<span>{escape(str(event["event_at"]))} · {escape(str(event["target"]))}</span></div>',
                            unsafe_allow_html=True,
                        )
        with avatar_col:
            with st.popover(initials_of(full_name), key="avatar_popover"):
                st.markdown(
                    f'<div class="popover-title">{escape(full_name)}</div>'
                    f'<p class="popover-note">{escape(role_label)} · {escape(username)}</p>',
                    unsafe_allow_html=True,
                )
                if st.button("Настройки", key="link_avatar_settings", width="stretch"):
                    st.session_state[NAV_KEY] = "Настройки"
                    st.rerun()
                if st.button("Выход", key="avatar_logout", width="stretch"):
                    logout()
    return query.strip()


def risk_pill(risk_label: str, probability: float | None = None) -> str:
    meta = RISK_META.get(risk_label, {})
    color = meta.get("color", "#806b6c")
    title = meta.get("title", risk_label)
    text = title if probability is None or pd.isna(probability) else f"{title} {probability:.0%}"
    style = (
        f"color:{color};background:{hex_to_rgba(color, 0.15)};"
        f"border-color:{hex_to_rgba(color, 0.3)};"
    )
    return f'<span class="risk-badge" style="{style}">{escape(text)}</span>'


def find_patient_by_query(query: str) -> str | None:
    """Быстрый поиск пациентки по ID или имени (для строки поиска в топбаре)."""
    patients = patients_df()
    needle = (query or "").strip().lower()
    if patients.empty or not needle:
        return None
    ids = patients["patient_id"].astype(str)
    names = patients["initials"].astype(str).str.strip()
    exact = patients.loc[(ids.str.lower() == needle) | (names.str.lower() == needle)]
    if not exact.empty:
        return str(exact.iloc[0]["patient_id"])
    partial = patients.loc[ids.str.lower().str.contains(needle, regex=False) | names.str.lower().str.contains(needle, regex=False)]
    if not partial.empty:
        return str(partial.iloc[0]["patient_id"])
    return None


def render_sidebar(user: dict[str, str], pages: list[str]) -> None:
    role_label = ROLE_LABELS.get(str(user.get("role", "")), "Врач")
    workspace = "Админ-панель" if is_admin(user) else "Кабинет врача"
    section = "Админ" if is_admin(user) else "Врач"
    full_name = user.get("full_name") or user.get("username", "")
    specialty = str(user.get("specialty") or "").strip() or role_label
    active = st.session_state.get(NAV_KEY) or pages[0]
    active_slug = NAV_SLUGS.get(str(active), "dashboard")

    st.sidebar.markdown(
        f"""
        <div class="side-brand">
            <div class="side-brand-mark">{icon_html("va-icon-logo", "1.35rem")}</div>
            <div>
                <span class="side-brand-name">VascularAI</span>
                <span class="side-brand-sub">Clinical Platform</span>
            </div>
        </div>
        <div class="side-role">
            <span>{escape(role_label)}</span>
            <span>{escape(workspace)}</span>
        </div>
        <div class="side-nav-label">{escape(section)}</div>
        <div class="nav-active-{active_slug}"></div>
        """,
        unsafe_allow_html=True,
    )

    for page in pages:
        slug = NAV_SLUGS.get(page, "page")
        if st.sidebar.button(page, key=f"nav_{slug}", width="stretch"):
            st.session_state[NAV_KEY] = page
            st.session_state.pop(PATIENT_VIEW_KEY, None)
            st.rerun()

    st.sidebar.markdown(
        f"""
        <div class="side-user">
            <span class="avatar">{escape(initials_of(full_name))}</span>
            <div>
                <strong>{escape(full_name)}</strong>
                <span>{escape(specialty)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Выход", key="nav_logout", width="stretch"):
        logout()


def section_label(left: str, right: str = "") -> None:
    st.markdown(
        f'<div class="section-label"><span>{escape(left)}</span><span>{escape(right)}</span></div>',
        unsafe_allow_html=True,
    )


def metric_cards(cards: list[tuple[str, str]]) -> None:
    html = ['<div class="metric-grid">']
    for label, value in cards:
        html.append(f'<div class="metric-card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def go_to_page(page: str) -> None:
    st.session_state[NAV_KEY] = page
    st.rerun()


def risk_badge_html(risk_label: str) -> str:
    if not risk_label:
        return '<span class="risk-badge muted-badge">Нет визита</span>'
    meta = RISK_META.get(risk_label)
    if meta is None:
        return f'<span class="risk-badge muted-badge">{escape(risk_label)}</span>'
    color = meta["color"]
    return (
        f'<span class="risk-badge" style="color:{color};background:{color}18;'
        f'border-color:{color}36;">{escape(meta["title"])}</span>'
    )


WEEKDAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
MONTHS_RU = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]


def human_date(moment: datetime) -> str:
    return f"{WEEKDAYS_RU[moment.weekday()]}, {moment.day} {MONTHS_RU[moment.month - 1]} {moment.year}"


def visits_with_context(limit: int | None = None, patient_id: str | None = None) -> pd.DataFrame:
    """Расчёты с контекстом карты: последние сверху, опционально по пациентке."""
    visits = visits_with_patient_context()
    if visits.empty:
        return visits
    frame = visits.sort_values("recorded_at", ascending=False)
    if patient_id:
        frame = frame.loc[frame["patient_id"].astype(str) == str(patient_id)]
    if limit:
        frame = frame.head(limit)
    return frame


def render_visit_rows(frame: pd.DataFrame, key_prefix: str, action: str = "Открыть") -> None:
    """Строки таблицы расчётов с действием «Открыть карту»."""
    if frame.empty:
        st.markdown('<div class="empty-state">Расчётов пока нет.</div>', unsafe_allow_html=True)
        return

    with st.container(key=f"head_{key_prefix}"):
        columns = st.columns([1.1, 0.8, 0.7, 1.3, 1.2, 0.9], vertical_alignment="center")
        for column, label in zip(columns, ["ID пациентки", "Возраст", "Срок", "Дата расчёта", "Риск", ""]):
            with column:
                st.markdown(f'<span class="cell-head">{label}</span>', unsafe_allow_html=True)

    for _, row in frame.iterrows():
        patient_id = str(row["patient_id"])
        probability = pd.to_numeric(pd.Series([row.get("risk_probability", "")]), errors="coerce").iloc[0]
        age = str(row.get("age", "") or "—")
        week = str(row.get("gestational_week", "") or "—")
        recorded = str(row.get("recorded_at", "") or "—")[:16]
        with st.container(key=f"row_{key_prefix}_{patient_id}_{str(row.get('visit_id', ''))}"):
            columns = st.columns([1.1, 0.8, 0.7, 1.3, 1.2, 0.9], vertical_alignment="center")
            with columns[0]:
                st.markdown(f'<span class="cell-strong">{escape(patient_id)}</span>', unsafe_allow_html=True)
            with columns[1]:
                st.markdown(f'<span class="cell-text">{escape(age)} лет</span>', unsafe_allow_html=True)
            with columns[2]:
                st.markdown(f'<span class="cell-text">{escape(week)} нед</span>', unsafe_allow_html=True)
            with columns[3]:
                st.markdown(f'<span class="cell-muted">{escape(recorded)}</span>', unsafe_allow_html=True)
            with columns[4]:
                st.markdown(risk_pill(str(row.get("risk_label", "")), probability), unsafe_allow_html=True)
            with columns[5]:
                if st.button(action, key=f"{key_prefix}_{patient_id}_{str(row.get('visit_id', ''))}"):
                    st.session_state[PATIENT_VIEW_KEY] = patient_id
                    st.session_state[NAV_KEY] = "Пациенты"
                    st.rerun()


def render_home_page(bundle: dict, user: dict[str, str]) -> None:
    visits = visits_df()
    patients = patients_df()
    now = datetime.now()
    month_start = pd.Timestamp(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    previous_start = month_start - pd.DateOffset(months=1)
    parsed = pd.to_datetime(visits["recorded_at"], errors="coerce")
    valid = visits.loc[parsed.notna()].copy()
    dated = parsed.loc[parsed.notna()]
    month_frame = valid.loc[dated >= month_start]
    previous_frame = valid.loc[(dated >= previous_start) & (dated < month_start)]

    def count_risk(frame: pd.DataFrame, label: str) -> int:
        return int((frame["risk_label"] == label).sum()) if not frame.empty else 0

    month_total = len(month_frame)
    previous_total = len(previous_frame)
    if previous_total:
        delta_visits = f"{(month_total - previous_total) / previous_total:+.0%}"
    else:
        delta_visits = f"{month_total} за месяц" if month_total else ""
    high_month = count_risk(month_frame, "high risk")
    high_delta = f"{high_month - count_risk(previous_frame, 'high risk'):+d}" if previous_total else ""
    mid_month = count_risk(month_frame, "mid risk")
    mid_share = mid_month / month_total if month_total else 0.0

    last_ago = "—"
    last_patient = ""
    if not dated.empty:
        last_time = dated.max()
        hours = int((pd.Timestamp(now) - last_time).total_seconds() // 3600)
        last_ago = f"{hours} ч назад" if hours < 24 else f"{hours // 24} дн назад"
        last_patient = str(valid.loc[dated.idxmax(), "patient_id"])

    full_name = user.get("full_name") or user.get("username", "")
    department = str(user.get("clinic") or "").strip() or "Акушерско-гинекологическое отделение"
    page_head(f"Добрый день, {full_name}", f"{human_date(now)} — {department}")

    kpi_cards(
        [
            (
                "va-icon-file",
                "Расчётов за месяц",
                str(month_total),
                delta_visits,
                False,
                {"page": "Отчёты", "hint": "Открыть отчёты"},
            ),
            (
                "va-icon-alert",
                "Высокий риск",
                str(high_month),
                high_delta,
                True,
                {"page": "Пациенты", "risk": "Высокий", "hint": "Показать пациенток с высоким риском"},
            ),
            (
                "va-icon-percent",
                "Средний риск",
                f"{mid_share:.0%}",
                f"{mid_month} расчётов",
                False,
                {"page": "Пациенты", "risk": "Средний", "hint": "Показать пациенток со средним риском"},
            ),
            (
                "va-icon-clock",
                "Последний расчёт",
                last_ago,
                last_patient,
                False,
                {"page": "Пациенты", "patient": last_patient or None, "hint": "Открыть карту пациентки"},
            ),
        ]
    )

    action_col, _ = st.columns([0.26, 0.74])
    with action_col:
        if st.button("Новый расчёт риска", type="primary", width="stretch"):
            go_to_page("Новый расчёт")

    with st.container(border=True):
        head_col, link_col = st.columns([1.6, 0.4], vertical_alignment="center")
        with head_col:
            card_head("Последние пациенты", "10 последних расчётов")
        with link_col:
            link_button("Все пациенты →", "dash_patients", "Пациенты")
        render_visit_rows(visits_with_context(limit=10), "dash")

    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        with st.container(border=True):
            card_head("Очередь внимания", "средний и высокий риск")
            render_queue(visits, patients)

    with right:
        with st.container(border=True):
            metrics = bundle.get("metrics", {})
            card_head("Статус модели", "XGBoost · maternal health risk")
            st.markdown(
                "<p class=\"cell-muted\">Accuracy "
                f"{metrics.get('accuracy', 0):.1%}, balanced accuracy {metrics.get('balanced_accuracy', 0):.1%}. "
                "SHAP-факторы показываются после расчёта конкретного визита.</p>",
                unsafe_allow_html=True,
            )


def render_feature_input(feature: str, value: float | None = None) -> float:
    meta = FEATURE_META[feature]
    default_value = meta["default"] if value is None else value
    st.markdown(
        f"""
        <div>
            <div class="field-label">
                <strong>{escape(meta['label'])}</strong>
                <span class="unit-badge">{escape(meta['unit'])}</span>
            </div>
            <div class="field-hint">{escape(meta['hint'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.number_input(
        label=f"{meta['label']}, {meta['unit']}",
        min_value=meta["min"],
        max_value=meta["max"],
        value=default_value,
        step=meta["step"],
        format=meta["format"],
        label_visibility="collapsed",
        key=f"input_{feature}",
    )


def render_probability_rows(class_names: list[str], probabilities: np.ndarray) -> None:
    for class_name, probability in zip(class_names, probabilities):
        meta = RISK_META[class_name]
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">{meta['title']}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{probability * 100:.1f}%;background:{meta['color']};"></div>
                </div>
                <div class="prob-value">{probability * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_factors(feature_names: list[str], patient: pd.DataFrame, shap_values: np.ndarray) -> None:
    rows = sorted(
        zip(feature_names, patient.iloc[0].to_numpy(dtype=float), shap_values),
        key=lambda item: abs(item[2]),
        reverse=True,
    )
    for feature, value, contribution in rows:
        color = "#bc617c" if contribution >= 0 else "#328d84"
        direction = "усиливает этот класс" if contribution >= 0 else "ослабляет этот класс"
        st.markdown(
            f"""
            <div class="factor-row" style="--factor-color:{color};">
                <div class="factor-main">
                    <strong>{FEATURE_META[feature]['label']}: {format_value(feature, value)}</strong>
                    <span>{FEATURE_META[feature]['hint']} · {direction}</span>
                </div>
                <div class="factor-score">{contribution:+.3f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result(result: dict[str, object], bundle: dict) -> None:
    class_name = str(result["class_name"])
    risk = RISK_META[class_name]
    probabilities = np.asarray(result["probabilities"], dtype=float)
    class_index = int(result["class_index"])
    score_degrees = probabilities[class_index] * 360
    st.markdown(
        f"""
        <div class="risk-panel">
            <div class="risk-top">
                <div>
                    <div class="risk-status">{risk['status']}</div>
                    <h2 class="risk-title">{risk['title']}</h2>
                    <p class="risk-summary">{risk['summary']}</p>
                </div>
                <div class="score-ring" style="--risk-color:{risk['color']};--score:{score_degrees:.1f}deg;">
                    <div class="score-core">
                        <strong>{probabilities[class_index] * 100:.0f}%</strong>
                        <span>score</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_label("Вероятности", "3 класса")
    render_probability_rows(bundle["class_names"], probabilities)
    section_label("Факторы", "SHAP")
    render_factors(bundle["feature_names"], result["patient"], np.asarray(result["shap_values"], dtype=float))


def create_patient_form(default_doctor: str = "") -> None:
    patients = patients_df()
    with st.form("create_patient_form", clear_on_submit=False):
        st.markdown('<div class="form-section">Карта пациентки</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            patient_id = st.text_input("Внутренний ID", value=next_patient_id(patients))
            age = st.number_input("Возраст", min_value=10, max_value=70, value=29, step=1)
            gravida = st.number_input("Gravida", min_value=0, max_value=20, value=1, step=1)
        with c2:
            initials = st.text_input("Инициалы / код", value="")
            gestational_week = st.number_input("Срок беременности, недель", min_value=4, max_value=45, value=28, step=1)
            parity = st.number_input("Parity", min_value=0, max_value=20, value=0, step=1)
        with c3:
            doctor = st.text_input("Ответственный врач", value=default_doctor)
            status = st.selectbox("Статус", ["Активна", "Наблюдение", "Архив"])
            bmi = st.text_input("ИМТ, кг/м²", placeholder="24.5")
        anamnesis = st.text_input("Анамнез", placeholder="Преэклампсия, хронические заболевания")
        note = st.text_area("Заметка", height=80)
        submitted = st.form_submit_button("Создать карту", type="primary")

    if submitted:
        patient_id = patient_id.strip() or next_patient_id(patients)
        if patient_id in patients["patient_id"].astype(str).tolist():
            st.error("Такой ID уже есть. Выберите другой внутренний ID.")
            return
        append_patient(
            {
                "patient_id": patient_id,
                "initials": initials.strip() or "Без имени",
                "age": str(age),
                "gestational_week": str(gestational_week),
                "gravida": str(gravida),
                "parity": str(parity),
                "doctor": doctor.strip(),
                "status": status,
                "created_at": now_iso(),
                "note": note.strip(),
                "bmi": bmi.strip(),
                "anamnesis": anamnesis.strip(),
            }
        )
        log_event(
            "patient_created",
            target=patient_id,
            details=f"Врач: {doctor.strip() or 'не указан'}",
        )
        st.success(f"Карта {patient_id} создана.")


def patient_selector(patients: pd.DataFrame, key: str) -> str | None:
    if patients.empty:
        return None
    options = [patient_label(row) for _, row in patients.iterrows()]
    selected = st.selectbox("Пациентка", options, key=key)
    return selected.split(" · ", 1)[0]


def render_intake_page(bundle: dict, user: dict[str, str]) -> None:
    patients = patients_df()
    if patients.empty:
        page_head("Новый расчёт", "сначала создайте карту пациентки")
        create_patient_form(user.get("full_name", ""))
        return

    preselect = st.session_state.pop(INTAKE_PRESELECT_KEY, None)
    if preselect:
        preselected_rows = patients.loc[patients["patient_id"].astype(str) == str(preselect)]
        if not preselected_rows.empty:
            st.session_state["intake_patient"] = patient_label(preselected_rows.iloc[0])

    progress_col, caption_col = st.columns([0.86, 0.14], vertical_alignment="center")
    with progress_col:
        st.markdown('<div class="progress-line"><i style="width:100%;"></i></div>', unsafe_allow_html=True)
    with caption_col:
        st.markdown('<div class="progress-caption">Шаг 1 из 1</div>', unsafe_allow_html=True)

    page_head("Новый расчёт риска преэклампсии", "Все данные из рутинного первого скрининга (11–13+6 нед)")

    left, right = st.columns([0.98, 1.02], gap="large")

    with left, st.container(border=True):
        selected_patient = patient_selector(patients, "intake_patient")
        patient_row = patients.loc[patients["patient_id"] == selected_patient].iloc[0]
        default_age = safe_int(patient_row["age"], int(FEATURE_META["Age"]["default"]))
        default_week = safe_int(patient_row["gestational_week"], 28)

        st.markdown('<div class="form-section">Антропометрия и витальные показатели</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age_value = render_feature_input("Age", default_age)
            systolic = render_feature_input("SystolicBP")
            sugar = render_feature_input("BS")
        with c2:
            heart_rate = render_feature_input("HeartRate")
            diastolic = render_feature_input("DiastolicBP")
            body_temp = render_feature_input("BodyTemp")
        bmi_col, week_col = st.columns(2)
        with bmi_col:
            bmi = labeled_text_input(
                "ИМТ",
                unit="кг/м²",
                hint="Индекс массы тела",
                value=str(patient_row.get("bmi") or ""),
                placeholder="24.5",
            )
        with week_col:
            labeled_number_input(
                "Срок беременности",
                unit="нед",
                hint="Срок на момент расчёта",
                value=default_week,
                minimum=4,
                maximum=45,
            )

        st.markdown('<div class="form-section">Биомаркеры (при наличии)</div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            plgf = labeled_text_input("PLGF", unit="пг/мл", hint="Плацентарный фактор роста", placeholder="—")
            sflt1 = labeled_text_input("SFLT-1", unit="пг/мл", hint="Растворимая тирозинкиназа", placeholder="—")
        with b2:
            papp_a = labeled_text_input("PAPP-A", unit="MoM", hint="Белок, связанный с беременностью", placeholder="—")
            map_value = labeled_text_input("MAP", unit="мм рт.ст.", hint="Среднее артериальное давление", placeholder="92")

        st.markdown('<div class="form-section">Анамнез</div>', unsafe_allow_html=True)
        preeclampsia_history = st.checkbox("Преэклампсия в анамнезе")
        chronic_disease = st.checkbox("Хронические заболевания (гипертония, диабет, болезни почек)")

        visit_note = st.text_area("Комментарий к визиту", height=80)
        attach_visit = st.checkbox("Сохранить визит в карту пациентки", value=True)

        submit_col, cancel_col = st.columns(2)
        with submit_col:
            submitted = st.button("Рассчитать риск", type="primary", width="stretch")
        with cancel_col:
            if st.button("Отмена", width="stretch"):
                st.session_state.pop("intake_result", None)
                go_to_page("Дашборд")
                st.rerun()

        st.markdown(
            '<div class="hint-line">Модель использует 6 клинических показателей: возраст, давление, сахар, '
            "температура, пульс. ИМТ, биомаркеры и анамнез сохраняются в карте и в текущий расчёт не входят.</div>",
            unsafe_allow_html=True,
        )

    if submitted:
        values = {
            "Age": float(age_value),
            "SystolicBP": float(systolic),
            "DiastolicBP": float(diastolic),
            "BS": float(sugar),
            "BodyTemp": float(body_temp),
            "HeartRate": float(heart_rate),
        }
        result = predict_risk(values, bundle)
        st.session_state["intake_result"] = result
        if attach_visit:
            probabilities = np.asarray(result["probabilities"], dtype=float)
            class_name = str(result["class_name"])
            anamnesis_notes = []
            if preeclampsia_history:
                anamnesis_notes.append("преэклампсия в анамнезе")
            if chronic_disease:
                anamnesis_notes.append("хронические заболевания")
            append_visit(
                {
                    "visit_id": uuid.uuid4().hex[:10],
                    "patient_id": selected_patient,
                    "recorded_at": now_iso(),
                    **{feature: values[feature] for feature in bundle["feature_names"]},
                    "risk_label": class_name,
                    "risk_probability": f"{probabilities[int(result['class_index'])]:.6f}",
                    "prob_low": f"{probabilities[0]:.6f}",
                    "prob_mid": f"{probabilities[1]:.6f}",
                    "prob_high": f"{probabilities[2]:.6f}",
                    "top_factor": result["top_factor"],
                    "visit_note": visit_note.strip(),
                    "bmi": bmi.strip(),
                    "plgf": plgf.strip(),
                    "papp_a": papp_a.strip(),
                    "sflt1": sflt1.strip(),
                    "map_value": map_value.strip(),
                    "anamnesis": ", ".join(anamnesis_notes),
                }
            )
            log_event(
                "visit_calculated",
                target=selected_patient,
                details=(
                    f"{RISK_META.get(class_name, {}).get('title', class_name)}, "
                    f"score {probabilities[int(result['class_index'])]:.0%}, "
                    f"фактор: {result['top_factor']}"
                ),
            )
            st.success(f"Расчёт сохранён в карту {selected_patient}.")

    with right, st.container(border=True):
        card_head("Результат", "прогноз и объяснение")
        stored_result = st.session_state.get("intake_result")
        if stored_result:
            render_result(stored_result, bundle)
        else:
            st.markdown(
                """
                <div class="risk-panel">
                    <div class="risk-status">Ожидание</div>
                    <h2 class="risk-title">Введите показатели</h2>
                    <p class="risk-summary">После расчёта здесь появятся риск, вероятности по классам и SHAP-факторы.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="clinical-note">Демо-система поддержки решений. Не является медицинским заключением.</div>',
            unsafe_allow_html=True,
        )


def save_uploaded_attachment(patient_id: str, uploaded_file, note: str) -> None:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", uploaded_file.name).strip("_") or "attachment"
    relative = f"{patient_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    data = bytes(uploaded_file.getbuffer())
    if attachments_in_cloud():
        upload_attachment_to_cloud(relative, data)
        stored_path = f"cloud:{relative}"
    else:
        local_path = ATTACHMENTS_DIR / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        stored_path = str(local_path.relative_to(PROJECT_ROOT))
    append_attachment(
        {
            "attachment_id": uuid.uuid4().hex[:10],
            "patient_id": patient_id,
            "filename": uploaded_file.name,
            "stored_path": stored_path,
            "uploaded_at": now_iso(),
            "note": note.strip(),
        }
    )
    log_event("attachment_uploaded", target=patient_id, details=uploaded_file.name)


PATIENTS_PER_PAGE = 8
INTAKE_PRESELECT_KEY = "intake_preselect"

CHART_INK = "#4a3238"
CHART_MUTED = "#806b6c"
CHART_GRID = "#ead6cf"
CHART_ACCENT = "#bd6875"


def latest_visits_by_patient(visits: pd.DataFrame) -> pd.DataFrame:
    columns = ["patient_id", "recorded_at", "risk_label", "risk_probability", "top_factor"]
    if visits.empty:
        return pd.DataFrame(columns=columns)
    latest = visits.sort_values("recorded_at").groupby("patient_id", as_index=False).tail(1)
    return latest[columns]


def render_patient_rows(frame: pd.DataFrame, key_prefix: str) -> None:
    if frame.empty:
        st.markdown('<div class="empty-state">Пациентов по этому фильтру не найдено.</div>', unsafe_allow_html=True)
        return

    with st.container(key=f"head_{key_prefix}"):
        columns = st.columns([1.0, 0.7, 0.7, 1.3, 1.1, 1.3], vertical_alignment="center")
        for column, label in zip(columns, ["ID", "Возраст", "Срок", "Последний расчёт", "Риск", "Действия"]):
            with column:
                st.markdown(f'<span class="cell-head">{label}</span>', unsafe_allow_html=True)

    for _, row in frame.iterrows():
        patient_id = str(row["patient_id"])
        probability = pd.to_numeric(pd.Series([row.get("risk_probability", "")]), errors="coerce").iloc[0]
        recorded = str(row.get("recorded_at", "") or "—")[:10]
        risk_label = str(row.get("risk_label", ""))
        with st.container(key=f"row_{key_prefix}_{patient_id}"):
            columns = st.columns([1.0, 0.7, 0.7, 1.3, 1.1, 1.3], vertical_alignment="center")
            with columns[0]:
                st.markdown(f'<span class="cell-strong">{escape(patient_id)}</span>', unsafe_allow_html=True)
            with columns[1]:
                st.markdown(f'<span class="cell-text">{escape(str(row.get("age") or "—"))} лет</span>', unsafe_allow_html=True)
            with columns[2]:
                st.markdown(f'<span class="cell-text">{escape(str(row.get("gestational_week") or "—"))} нед</span>', unsafe_allow_html=True)
            with columns[3]:
                st.markdown(f'<span class="cell-muted">{escape(recorded)}</span>', unsafe_allow_html=True)
            with columns[4]:
                st.markdown(risk_pill(risk_label, probability) if risk_label else '<span class="cell-muted">нет данных</span>', unsafe_allow_html=True)
            with columns[5]:
                history_col, intake_col = st.columns(2)
                with history_col:
                    if st.button("История", key=f"{key_prefix}_history_{patient_id}"):
                        st.session_state[PATIENT_VIEW_KEY] = patient_id
                        st.rerun()
                with intake_col:
                    if st.button("Новый расчёт", key=f"{key_prefix}_intake_{patient_id}"):
                        st.session_state[INTAKE_PRESELECT_KEY] = patient_id
                        go_to_page("Новый расчёт")
                        st.rerun()


def render_patients_page(user: dict[str, str]) -> None:
    patients = patients_df()
    if not patients.empty and st.session_state.get(PATIENT_VIEW_KEY):
        render_patient_card(user, str(st.session_state[PATIENT_VIEW_KEY]))
        return

    pending_risk = st.session_state.pop(PENDING_RISK_FILTER, None)
    if pending_risk:
        st.session_state["patients_risk_filter"] = pending_risk
    elif "patients_risk_filter" not in st.session_state:
        st.session_state["patients_risk_filter"] = "Все риски"

    page_head("Пациенты", f"{len(patients)} пациентов в базе")
    with st.expander("Новая карта пациентки"):
        create_patient_form(user.get("full_name", ""))

    if patients.empty:
        st.markdown(
            '<div class="panel"><div class="empty-state">Пока нет карт пациенток. Создайте первую карту выше.</div></div>',
            unsafe_allow_html=True,
        )
        return

    visits = visits_df()
    table = patients.merge(latest_visits_by_patient(visits), on="patient_id", how="left").fillna("")

    search_col, risk_col, week_col, period_col = st.columns([1.15, 1.35, 1.15, 1.05], vertical_alignment="center")
    with search_col:
        query = st.text_input("Поиск по ID", placeholder="Поиск по ID...", label_visibility="collapsed", key="patients_search")
    with risk_col:
        risk_filter = st.segmented_control(
            "Риск",
            ["Все риски", "Высокий", "Средний", "Низкий"],
            label_visibility="collapsed",
            key="patients_risk_filter",
        )
    with week_col:
        week_filter = st.segmented_control(
            "Срок",
            ["Любой срок", "≤ 14 нед", "15–28 нед", "> 28 нед"],
            default="Любой срок",
            label_visibility="collapsed",
            key="patients_week_filter",
        )
    with period_col:
        period_filter = st.segmented_control(
            "Период",
            ["За всё время", "Месяц", "Неделя"],
            default="За всё время",
            label_visibility="collapsed",
            key="patients_period_filter",
        )

    filtered = table.copy()
    if query.strip():
        needle = query.strip().lower()
        filtered = filtered.loc[
            filtered["patient_id"].astype(str).str.lower().str.contains(needle, regex=False)
            | filtered["initials"].astype(str).str.lower().str.contains(needle, regex=False)
        ]
    risk_by_filter = {"Высокий": "high risk", "Средний": "mid risk", "Низкий": "low risk"}
    if risk_filter in risk_by_filter:
        filtered = filtered.loc[filtered["risk_label"] == risk_by_filter[risk_filter]]
    elif risk_filter == "Все риски":
        pass
    if week_filter == "≤ 14 нед":
        weeks = pd.to_numeric(filtered["gestational_week"], errors="coerce")
        filtered = filtered.loc[weeks <= 14]
    elif week_filter == "15–28 нед":
        weeks = pd.to_numeric(filtered["gestational_week"], errors="coerce")
        filtered = filtered.loc[(weeks >= 15) & (weeks <= 28)]
    elif week_filter == "> 28 нед":
        weeks = pd.to_numeric(filtered["gestational_week"], errors="coerce")
        filtered = filtered.loc[weeks > 28]
    if period_filter in {"Месяц", "Неделя"}:
        days = 30 if period_filter == "Месяц" else 7
        cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
        recorded = pd.to_datetime(filtered["recorded_at"], errors="coerce")
        filtered = filtered.loc[recorded.notna() & (recorded >= cutoff)]

    filtered = filtered.sort_values(["risk_label", "created_at"], ascending=[True, False])

    total = len(filtered)
    total_pages = max((total + PATIENTS_PER_PAGE - 1) // PATIENTS_PER_PAGE, 1)
    current_page = int(st.session_state.get("patients_page", 1))
    current_page = min(max(current_page, 1), total_pages)
    st.session_state["patients_page"] = current_page
    start = (current_page - 1) * PATIENTS_PER_PAGE
    page_frame = filtered.iloc[start : start + PATIENTS_PER_PAGE]

    with st.container(border=True):
        card_head("Список пациенток", f"показано {len(page_frame)} из {total}")
        render_patient_rows(page_frame, "list")

        pager_left, pager_mid, pager_right = st.columns([1, 0.5, 1], vertical_alignment="center")
        with pager_left:
            st.markdown(
                f'<div class="pager">Показано {len(page_frame)} из {total}</div>',
                unsafe_allow_html=True,
            )
        with pager_mid:
            pager_prev, pager_next = st.columns(2)
            with pager_prev:
                if st.button("‹", key="patients_prev", disabled=current_page <= 1):
                    st.session_state["patients_page"] = current_page - 1
                    st.rerun()
            with pager_next:
                if st.button("›", key="patients_next", disabled=current_page >= total_pages):
                    st.session_state["patients_page"] = current_page + 1
                    st.rerun()
        with pager_right:
            st.markdown(
                f'<div class="pager"><div class="pager-pages"><span class="active">{current_page}</span>'
                f'<span>{total_pages}</span></div></div>',
                unsafe_allow_html=True,
            )


def render_patient_card(user: dict[str, str], patient_id: str) -> None:
    patients = patients_df()
    mask = patients["patient_id"].astype(str) == str(patient_id)
    if not mask.any():
        st.warning("Карта пациентки не найдена.")
        if st.button("← Назад к пациентам"):
            st.session_state.pop(PATIENT_VIEW_KEY, None)
            st.rerun()
        return

    row = patients.loc[mask].iloc[0]
    visits = visits_with_context(patient_id=patient_id)
    attachments = attachments_df()
    patient_attachments = attachments.loc[attachments["patient_id"].astype(str) == str(patient_id)]

    back_col, _ = st.columns([0.2, 0.8])
    with back_col:
        if st.button("← Назад к пациентам"):
            st.session_state.pop(PATIENT_VIEW_KEY, None)
            st.rerun()

    latest_label = str(visits.iloc[0]["risk_label"]) if not visits.empty else ""
    latest_probability = (
        pd.to_numeric(pd.Series([visits.iloc[0]["risk_probability"]]), errors="coerce").iloc[0]
        if not visits.empty
        else None
    )
    initials = str(row.get("initials") or "без имени")
    created = str(row.get("created_at") or "")[:10]
    anamnesis = str(row.get("anamnesis") or "").strip() or "не указан"
    bmi = str(row.get("bmi") or "").strip() or "—"

    st.markdown(
        f"""
        <div class="panel">
            <div class="card-head">
                <div>
                    <h3 style="font-size:1.4rem;">Пациент {escape(str(patient_id))}</h3>
                    <span class="card-sub">Наблюдается с {escape(created or '—')} · {escape(initials)}</span>
                </div>
                {risk_pill(latest_label, latest_probability) if latest_label else ''}
            </div>
            <div class="stat-row">
                <div><span>Возраст</span><strong>{escape(str(row.get("age") or "—"))} лет</strong></div>
                <div><span>ИМТ</span><strong>{escape(bmi)}{' кг/м²' if bmi != '—' else ''}</strong></div>
                <div><span>Текущий срок</span><strong>{escape(str(row.get("gestational_week") or "—"))} нед</strong></div>
                <div><span>Анамнез</span><strong>{escape(anamnesis)}</strong></div>
                <div><span>Расчётов</span><strong>{len(visits)}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if str(row.get("note") or "").strip():
        st.markdown(f'<div class="clinical-note">{escape(str(row["note"]))}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        card_head("Динамика риска по неделям", "доля вероятности высокого риска")
        render_risk_trend(visits)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.container(border=True):
            card_head("История расчётов", f"{len(visits)} записей")
            render_visit_rows(visits, "card")

    with right:
        with st.container(border=True):
            card_head("Вложения", f"{len(patient_attachments)} файлов")
            with st.form("attachment_form"):
                uploaded = st.file_uploader("Прикрепить файл", type=["csv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "txt"])
                attachment_note = st.text_input("Комментарий к файлу")
                attach_submitted = st.form_submit_button("Прикрепить файл")
            if attach_submitted and uploaded is not None:
                save_uploaded_attachment(str(patient_id), uploaded, attachment_note)
                st.success("Файл прикреплён к карте.")
            elif attach_submitted:
                st.warning("Выберите файл для прикрепления.")
            if patient_attachments.empty:
                st.markdown('<div class="empty-state">Вложений пока нет.</div>', unsafe_allow_html=True)
            else:
                display = patient_attachments.sort_values("uploaded_at", ascending=False).rename(
                    columns={"uploaded_at": "Дата", "filename": "Файл", "note": "Комментарий"}
                )
                st.dataframe(display[["Дата", "Файл", "Комментарий"]], width="stretch", hide_index=True)


def render_risk_trend(visits: pd.DataFrame) -> None:
    frame = visits.copy()
    frame["score"] = pd.to_numeric(frame["risk_probability"], errors="coerce")
    frame["дата"] = pd.to_datetime(frame["recorded_at"], errors="coerce")
    frame = frame.loc[frame["score"].notna() & frame["дата"].notna()].sort_values("дата")
    if frame.empty:
        st.markdown('<div class="empty-state">Пока нет расчётов для графика.</div>', unsafe_allow_html=True)
        return
    base = alt.Chart(frame).encode(
        x=alt.X("дата:T", title=None, axis=alt.Axis(format="%d %b", labelColor=CHART_MUTED, grid=False, tickSize=0)),
        y=alt.Y(
            "score:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%", labelColor=CHART_MUTED, gridColor=CHART_GRID, tickSize=0),
        ),
    )
    line = base.mark_line(color=CHART_ACCENT, strokeWidth=2.4).encode(
        tooltip=[
            alt.Tooltip("дата:T", title="Дата", format="%d.%m.%Y"),
            alt.Tooltip("score:Q", title="Риск", format=".0%"),
        ]
    )
    points = base.mark_point(color=CHART_ACCENT, filled=True, size=45)
    threshold = (
        alt.Chart(pd.DataFrame({"порог": [0.5]}))
        .mark_rule(color="#bc617c", strokeDash=[4, 4], strokeWidth=1.4)
        .encode(y="порог:Q")
    )
    st.altair_chart((threshold + line + points).properties(height=230), width="stretch")


def risk_distribution(visits: pd.DataFrame) -> None:
    counts = visits["risk_label"].value_counts().to_dict()
    total = max(sum(counts.values()), 1)
    for class_name in ["low risk", "mid risk", "high risk"]:
        count = int(counts.get(class_name, 0))
        meta = RISK_META[class_name]
        width = count / total * 100
        st.markdown(
            f"""
            <div class="bar-row">
                <span>{meta['title']}</span>
                <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%;background:{meta['color']};"></div></div>
                <span>{count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_queue(visits: pd.DataFrame, patients: pd.DataFrame) -> None:
    if visits.empty:
        st.write("Очередь появится после первых визитов.")
        return
    queue = visits.loc[visits["risk_label"].isin(["high risk", "mid risk"])].copy()
    if queue.empty:
        st.write("Сейчас нет визитов среднего или высокого риска.")
        return
    queue["risk_probability_num"] = pd.to_numeric(queue["risk_probability"], errors="coerce").fillna(0)
    queue = queue.sort_values(["risk_label", "risk_probability_num"], ascending=[True, False]).head(8)
    patient_names = patients.set_index("patient_id")["initials"].to_dict() if not patients.empty else {}
    for _, row in queue.iterrows():
        meta = RISK_META.get(row["risk_label"], RISK_META["mid risk"])
        patient_name = patient_names.get(row["patient_id"], row["patient_id"])
        st.markdown(
            f"""
            <div class="queue-row" style="border-left:4px solid {meta['color']};">
                <div>
                    <strong>{escape(patient_name)} · {meta['title']} · {float(row['risk_probability_num']):.0%}</strong>
                    <span>{escape(row['recorded_at'])} · главный фактор: {escape(row['top_factor'])}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def filter_by_period(visits: pd.DataFrame, period: str) -> pd.DataFrame:
    if visits.empty or period == "За всё время":
        return visits
    parsed = pd.to_datetime(visits["recorded_at"], errors="coerce")
    if period == "Текущий месяц":
        start = pd.Timestamp(datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    else:
        start = pd.Timestamp(datetime.now() - timedelta(days=30))
    return visits.loc[parsed.notna() & (parsed >= start)]


def risk_donut(frame: pd.DataFrame) -> None:
    rows = [
        {"Риск": RISK_META[label]["title"], "color": RISK_META[label]["color"], "Расчётов": int((frame["risk_label"] == label).sum())}
        for label in ["high risk", "mid risk", "low risk"]
    ]
    data = pd.DataFrame(rows)
    data = data.loc[data["Расчётов"] > 0]
    if data.empty:
        st.markdown('<div class="empty-state">Нет расчётов за выбранный период.</div>', unsafe_allow_html=True)
        return
    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=64, outerRadius=98, stroke="#fffdfa", strokeWidth=2)
        .encode(
            theta=alt.Theta("Расчётов:Q"),
            color=alt.Color(
                "Риск:N",
                scale=alt.Scale(domain=data["Риск"].tolist(), range=data["color"].tolist()),
                legend=alt.Legend(orient="bottom", title=None, labelColor=CHART_MUTED, symbolType="circle", columns=3, labelFontSize=12),
            ),
            tooltip=[alt.Tooltip("Риск:N"), alt.Tooltip("Расчётов:Q")],
        )
    )
    st.altair_chart(chart.properties(height=250), width="stretch")


def weekly_bars(frame: pd.DataFrame) -> None:
    parsed = pd.to_datetime(frame["recorded_at"], errors="coerce")
    dated = frame.loc[parsed.notna()].copy()
    dated["Неделя"] = parsed.loc[parsed.notna()].dt.strftime("%d.%m")
    if dated.empty:
        st.markdown('<div class="empty-state">Динамика появится после первых расчётов.</div>', unsafe_allow_html=True)
        return
    weekly = dated.groupby("Неделя", sort=True).size().reset_index(name="Расчётов").tail(8)
    chart = (
        alt.Chart(weekly)
        .mark_bar(color=CHART_INK, size=28, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Неделя:N", title=None, sort=None, axis=alt.Axis(labelColor=CHART_MUTED, grid=False, tickSize=0, labelAngle=0)),
            y=alt.Y("Расчётов:Q", title=None, axis=alt.Axis(labelColor=CHART_MUTED, gridColor=CHART_GRID, tickSize=0)),
            tooltip=[alt.Tooltip("Неделя:N"), alt.Tooltip("Расчётов:Q")],
        )
    )
    st.altair_chart(chart.properties(height=250), width="stretch")


def build_report_html(period: str, frame: pd.DataFrame, user: dict[str, str]) -> str:
    """Печатная сводка (открывается в браузере, сохраняется в PDF через печать)."""
    total = len(frame)
    rows = []
    for label in ["high risk", "mid risk", "low risk"]:
        count = int((frame["risk_label"] == label).sum()) if not frame.empty else 0
        share = f"{count / total:.0%}" if total else "—"
        rows.append(f"<tr><td>{RISK_META[label]['title']}</td><td>{count}</td><td>{share}</td></tr>")
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>VascularAI — отчёт</title>
<style>body{{font-family:Arial,sans-serif;color:#4a3238;padding:32px;}}
h1{{color:#4a3238;font-size:22px;}}table{{border-collapse:collapse;margin-top:16px;}}
td,th{{border-bottom:1px solid #ead6cf;padding:8px 18px 8px 0;text-align:left;}}
th{{color:#806b6c;font-size:12px;text-transform:uppercase;}}
.note{{color:#806b6c;font-size:12px;margin-top:24px;}}</style></head><body>
<h1>VascularAI — отчёт по расчётам</h1>
<p>Период: {escape(period)} · врач: {escape(user.get('full_name') or user.get('username', ''))} · всего расчётов: {total}</p>
<table><tr><th>Уровень риска</th><th>Расчётов</th><th>Доля</th></tr>{''.join(rows)}</table>
<p class="note">Демо-система поддержки решений. Не является медицинским заключением.<br>Для PDF: печать страницы (Ctrl+P) → «Сохранить как PDF».</p>
</body></html>"""


def render_dashboard_page() -> None:
    user = current_user() or {}
    visits = visits_with_context()
    page_head("Отчёты", "Аналитика по вашим расчётам")

    period_col, excel_col, pdf_col = st.columns([1.4, 0.8, 0.9], vertical_alignment="center")
    with period_col:
        period = st.segmented_control(
            "Период",
            ["Текущий месяц", "30 дней", "За всё время"],
            default="Текущий месяц",
            label_visibility="collapsed",
            key="reports_period",
        )
    frame = filter_by_period(visits, period or "Текущий месяц")
    with excel_col:
        st.download_button(
            "Excel (CSV)",
            data=frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="vascularai_otchet.csv",
            mime="text/csv",
            width="stretch",
        )
    with pdf_col:
        st.download_button(
            "PDF (печать)",
            data=build_report_html(period or "Текущий месяц", frame, user).encode("utf-8"),
            file_name="vascularai_otchet.html",
            mime="text/html",
            width="stretch",
        )

    total = len(frame)
    high_count = int((frame["risk_label"] == "high risk").sum()) if total else 0
    mid_count = int((frame["risk_label"] == "mid risk").sum()) if total else 0
    kpi_cards(
        [
            ("va-icon-file", "Всего расчётов", str(total), "за период", False),
            ("va-icon-alert", "Высокий риск", f"{high_count} ({high_count / total:.0%})" if total else "0", "требуют внимания", True),
            ("va-icon-percent", "Средний риск", f"{mid_count} ({mid_count / total:.0%})" if total else "0", "под наблюдением", False),
            ("va-icon-users", "Пациенток", str(frame["patient_id"].nunique() if total else 0), "уникальных карт", False),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        card_head("Распределение по уровням риска")
        risk_donut(frame)
    with right, st.container(border=True):
        card_head("Динамика расчётов по неделям")
        weekly_bars(frame)

    with st.container(border=True):
        card_head("Последние расчёты", f"{total} записей за период")
        render_visit_rows(frame.head(8), "reports")


def visits_with_patient_context() -> pd.DataFrame:
    visits = visits_df()
    patients = patients_df()
    if visits.empty:
        return visits
    context = patients[["patient_id", "initials", "doctor", "age", "gestational_week"]].copy()
    joined = visits.merge(context, on="patient_id", how="left", suffixes=("", "_patient"))
    for column in ["initials", "doctor", "age", "gestational_week"]:
        if column not in joined.columns:
            joined[column] = ""
        joined[column] = joined[column].fillna("")
    joined["doctor"] = joined["doctor"].replace("", "Не указан")
    joined["initials"] = joined["initials"].replace("", "Без имени")
    return joined


def render_doctor_drilldown(users: pd.DataFrame, visits: pd.DataFrame) -> None:
    """Разбор работы конкретного врача: объём, риск, факторы и последние действия."""
    doctors = users.loc[users["role"] == "doctor"].copy()
    if doctors.empty:
        st.write("Врачей пока нет. Создайте аккаунт во вкладке «Управление доступом».")
        return

    options: dict[str, str] = {}
    for _, row in doctors.iterrows():
        username = str(row["username"])
        full_name = str(row.get("full_name") or username)
        options[f"{full_name} · {username}"] = full_name

    selected_label = st.selectbox("Врач", list(options.keys()), key="admin_doctor_drilldown")
    username = selected_label.rsplit(" · ", 1)[-1]
    account = doctors.loc[doctors["username"] == username].iloc[0]
    doctor_name = options[selected_label]
    doctor_keys = {doctor_name.strip().lower(), username.strip().lower()}
    doctor_visits = visits.loc[visits["doctor"].astype(str).str.strip().str.lower().isin(doctor_keys)].copy()

    scores = pd.to_numeric(doctor_visits["risk_probability"], errors="coerce")
    high_count = int((doctor_visits["risk_label"] == "high risk").sum())
    mid_count = int((doctor_visits["risk_label"] == "mid risk").sum())
    last_visit = doctor_visits["recorded_at"].max() if not doctor_visits.empty else "—"
    mean_score = f"{scores.mean():.0%}" if not scores.empty and pd.notna(scores.mean()) else "—"

    section_label(
        "Карточка врача",
        f"{ROLE_LABELS.get(str(account['role']), 'Врач')} · {STATUS_LABELS.get(str(account['status']), '')}",
    )
    metric_cards(
        [
            ("Расчётов", str(len(doctor_visits))),
            ("Пациенток", str(doctor_visits["patient_id"].nunique())),
            ("Средний score", mean_score),
            ("Высокий риск", str(high_count)),
        ]
    )

    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True):
        card_head("Распределение риска")
        risk_distribution(doctor_visits)
        st.markdown(
            f'<div class="clinical-note">Расчёты сопоставлены с врачом по ФИО из карты пациентки '
            f"или по логину «{escape(username)}». Последний расчёт: {escape(str(last_visit))}. "
            f"Средний риск: {mid_count}. Последний вход: {escape(str(account.get('last_login_at') or 'Никогда'))}.</div>",
            unsafe_allow_html=True,
        )

    with right, st.container(border=True):
        card_head("Частые факторы риска")
        if doctor_visits.empty:
            st.write("Расчётов у врача пока нет.")
        else:
            factors = doctor_visits["top_factor"].replace("", "не определён").value_counts().head(5)
            top_count = int(factors.max()) or 1
            for factor, count in factors.items():
                st.markdown(
                    f"""
                    <div class="bar-row">
                        <span>{escape(str(factor))}</span>
                        <div class="bar-track"><div class="bar-fill" style="width:{int(count) / top_count * 100:.1f}%;background:var(--accent);"></div></div>
                        <span>{int(count)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('<div class="form-section">Последние действия врача</div>', unsafe_allow_html=True)
        events = activity_df()
        doctor_events = events.loc[events["actor"] == username].sort_values("event_at", ascending=False).head(6)
        if doctor_events.empty:
            st.write("Событий пока нет.")
        else:
            display = doctor_events.rename(
                columns={"event_at": "Время", "action": "Действие", "target": "Объект", "details": "Детали"}
            )
            display["Действие"] = display["Действие"].map(lambda value: ACTION_LABELS.get(value, value))
            st.dataframe(display[["Время", "Действие", "Объект", "Детали"]], width="stretch", hide_index=True)


def render_admin_analytics_page() -> None:
    users = users_df()
    visits = visits_with_patient_context()
    doctors = users.loc[users["role"] == "doctor"].copy()
    active_doctors = int(((doctors["status"] == "active")).sum()) if not doctors.empty else 0
    high_count = int((visits["risk_label"] == "high risk").sum()) if not visits.empty else 0
    mid_count = int((visits["risk_label"] == "mid risk").sum()) if not visits.empty else 0

    metric_cards(
        [
            ("Врачей", str(len(doctors))),
            ("Активных врачей", str(active_doctors)),
            ("Всего расчётов", str(len(visits))),
            ("Высокий риск", str(high_count)),
        ]
    )

    left, right = st.columns([0.92, 1.08], gap="large")
    with left, st.container(border=True):
        card_head("Распределение риска")
        risk_distribution(visits)

        st.markdown('<div class="form-section">Статусы аккаунтов</div>', unsafe_allow_html=True)
        if users.empty:
            st.write("Аккаунтов пока нет.")
        else:
            account_status = users["status"].map(lambda value: STATUS_LABELS.get(value, value)).value_counts()
            for label, count in account_status.items():
                st.markdown(
                    f"""
                    <div class="bar-row">
                        <span>{escape(str(label))}</span>
                        <div class="bar-track"><div class="bar-fill" style="width:{count / max(len(users), 1) * 100:.1f}%;background:var(--nav);"></div></div>
                        <span>{int(count)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right, st.container(border=True):
        card_head("Активность врачей")
        if visits.empty:
            st.write("Расчётов пока нет.")
        else:
            visits["risk_probability_num"] = pd.to_numeric(visits["risk_probability"], errors="coerce").fillna(0)
            doctor_summary = (
                visits.groupby("doctor", dropna=False)
                .agg(
                    Расчётов=("visit_id", "count"),
                    Пациенток=("patient_id", "nunique"),
                    Высокий_риск=("risk_label", lambda values: int((values == "high risk").sum())),
                    Средний_риск=("risk_label", lambda values: int((values == "mid risk").sum())),
                    Последний_расчёт=("recorded_at", "max"),
                )
                .reset_index()
                .rename(columns={"doctor": "Врач", "Высокий_риск": "Высокий риск", "Средний_риск": "Средний риск"})
                .sort_values(["Расчётов", "Высокий риск"], ascending=False)
            )
            st.dataframe(doctor_summary, width="stretch", hide_index=True)

        st.markdown('<div class="form-section">Динамика расчётов</div>', unsafe_allow_html=True)
        if visits.empty:
            st.write("Динамика появится после первых расчётов.")
        else:
            dates = pd.to_datetime(visits["recorded_at"], errors="coerce")
            weekly = (
                visits.assign(week=dates.dt.to_period("W").astype(str))
                .loc[dates.notna()]
                .groupby("week")
                .size()
                .tail(8)
            )
            if weekly.empty:
                st.write("Нет корректных дат для графика.")
            else:
                max_count = int(weekly.max()) or 1
                for week, count in weekly.items():
                    st.markdown(
                        f"""
                        <div class="bar-row">
                            <span>{escape(str(week))}</span>
                            <div class="bar-track"><div class="bar-fill" style="width:{int(count) / max_count * 100:.1f}%;background:var(--accent);"></div></div>
                            <span>{int(count)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    render_doctor_drilldown(users, visits)


def render_admin_activity_page() -> None:
    users = users_df()
    visits = visits_with_patient_context()
    attachments = attachments_df()
    activity = activity_df()

    metric_cards(
        [
            ("Событий в журнале", str(len(activity))),
            ("Входов", str(int((users["last_login_at"] != "").sum())) if not users.empty else "0"),
            ("Расчётов", str(len(visits))),
            ("Вложений", str(len(attachments))),
        ]
    )

    left, right = st.columns([1, 1], gap="large")
    with left, st.container(border=True):
        card_head("Последние входы")
        if users.empty:
            st.write("Нет аккаунтов.")
        else:
            login_events = users[["username", "full_name", "role", "status", "last_login_at"]].copy()
            login_events["role"] = login_events["role"].map(lambda value: ROLE_LABELS.get(value, value))
            login_events["status"] = login_events["status"].map(lambda value: STATUS_LABELS.get(value, value))
            login_events["last_login_at"] = login_events["last_login_at"].replace("", "Никогда")
            login_events = login_events.rename(
                columns={
                    "username": "Логин",
                    "full_name": "Имя",
                    "role": "Роль",
                    "status": "Статус",
                    "last_login_at": "Последний вход",
                }
            )
            st.dataframe(login_events, width="stretch", hide_index=True)

    with right, st.container(border=True):
        card_head("Последние расчёты")
        if visits.empty:
            st.write("Расчётов пока нет.")
        else:
            recent = visits.sort_values("recorded_at", ascending=False).head(12).copy()
            recent["risk_label"] = recent["risk_label"].map(lambda value: RISK_META.get(value, {}).get("title", value))
            recent["risk_probability"] = pd.to_numeric(recent["risk_probability"], errors="coerce").map(
                lambda value: f"{value:.0%}" if pd.notna(value) else ""
            )
            recent = recent.rename(
                columns={
                    "recorded_at": "Дата",
                    "patient_id": "ID пациентки",
                    "initials": "Пациентка",
                    "doctor": "Врач",
                    "risk_label": "Риск",
                    "risk_probability": "Score",
                    "top_factor": "Фактор",
                }
            )
            st.dataframe(
                recent[["Дата", "ID пациентки", "Пациентка", "Врач", "Риск", "Score", "Фактор"]],
                width="stretch",
                hide_index=True,
            )

    with st.container(border=True):
        card_head("Вложения")
        if attachments.empty:
            st.write("Вложений пока нет.")
        else:
            display = attachments.sort_values("uploaded_at", ascending=False).head(20).rename(
                columns={
                    "uploaded_at": "Дата",
                    "patient_id": "ID пациентки",
                    "filename": "Файл",
                    "note": "Комментарий",
                    "stored_path": "Путь",
                }
            )
            st.dataframe(display[["Дата", "ID пациентки", "Файл", "Комментарий", "Путь"]], width="stretch", hide_index=True)

    st.markdown('<div class="form-section">Журнал действий</div>', unsafe_allow_html=True)
    if activity.empty:
        st.write(
            "Событий пока нет. Журнал наполняется автоматически: входы и выходы, расчёты риска, "
            "создание карт и аккаунтов, вложения и изменения доступов."
        )
    else:
        data = activity.copy()
        data["event_dt"] = pd.to_datetime(data["event_at"], errors="coerce")
        f1, f2, f3 = st.columns([0.8, 1.1, 1.1])
        with f1:
            period_label = st.selectbox("Период", ["Все время", "7 дней", "30 дней"], key="activity_period")
        with f2:
            action_labels = ["Все действия"] + sorted({ACTION_LABELS.get(value, value) for value in data["action"]})
            action_label = st.selectbox("Действие", action_labels, key="activity_action")
        with f3:
            actor_labels = ["Все пользователи"] + sorted(data["actor"].astype(str).unique().tolist())
            actor_label = st.selectbox("Пользователь", actor_labels, key="activity_actor")

        if period_label != "Все время":
            days = 7 if period_label == "7 дней" else 30
            cutoff = pd.Timestamp(datetime.now() - timedelta(days=days))
            data = data.loc[data["event_dt"].fillna(pd.Timestamp("1970-01-01")) >= cutoff]
        if action_label != "Все действия":
            data = data.loc[data["action"].map(lambda value: ACTION_LABELS.get(value, value)) == action_label]
        if actor_label != "Все пользователи":
            data = data.loc[data["actor"].astype(str) == actor_label]

        if data.empty:
            st.write("Под выбранные фильтры событий не найдено.")
        else:
            display = data.sort_values("event_at", ascending=False).head(100).copy()
            display["action"] = display["action"].map(lambda value: ACTION_LABELS.get(value, value))
            display["actor_role"] = display["actor_role"].replace("", "—")
            display = display.rename(
                columns={
                    "event_at": "Время",
                    "actor": "Пользователь",
                    "actor_role": "Роль",
                    "action": "Действие",
                    "target": "Объект",
                    "details": "Детали",
                }
            )
            st.dataframe(
                display[["Время", "Пользователь", "Роль", "Действие", "Объект", "Детали"]],
                width="stretch",
                hide_index=True,
            )
            st.caption(f"Показано {len(display)} из {len(data)} событий.")
        st.download_button(
            "Скачать журнал CSV",
            data=activity.to_csv(index=False).encode("utf-8-sig"),
            file_name="vascularai_activity.csv",
            mime="text/csv",
        )


def render_settings_page(user: dict[str, str]) -> None:
    users = users_df()
    mask = users["username"].str.lower() == normalize_username(user.get("username", ""))
    if not mask.any():
        st.error("Аккаунт не найден.")
        return
    row_index = users.index[mask][0]
    account = users.loc[row_index]

    page_head("Настройки", "Профиль врача, безопасность и уведомления")

    full_name_value = str(account.get("full_name") or "")
    name_parts = full_name_value.split(" ", 1)
    last_name_value = name_parts[0] if name_parts else ""
    first_name_value = name_parts[1] if len(name_parts) > 1 else ""
    specialty_value = str(account.get("specialty") or "")
    clinic_value = str(account.get("clinic") or "")
    role_label = ROLE_LABELS.get(str(account.get("role")), "Врач")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left, st.container(border=True):
        st.markdown(
            f"""
            <div class="card-head">
                <div style="display:flex;align-items:center;gap:.75rem;">
                    <span class="avatar avatar-light" style="width:3.1rem;height:3.1rem;flex:0 0 3.1rem;font-size:1rem;">
                        {escape(initials_of(full_name_value or str(account.get('username', ''))))}
                    </span>
                    <div>
                        <h3>{escape(full_name_value or str(account.get("username", "")))}</h3>
                        <span class="card-sub">{escape(specialty_value or role_label)}{' · ' + escape(clinic_value) if clinic_value else ''}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("profile_settings_form"):
            first_col, second_col = st.columns(2)
            with first_col:
                first_name = st.text_input("Имя", value=first_name_value)
                email = st.text_input("Email", value=str(account.get("email") or ""), placeholder="doctor@clinic.kz")
                clinic = st.text_input("Клиника", value=clinic_value, placeholder="Городской перинатальный центр")
            with second_col:
                last_name = st.text_input("Фамилия", value=last_name_value)
                phone = st.text_input("Телефон", value=str(account.get("phone") or ""), placeholder="+7 700 000 00 00")
                specialty = st.text_input("Специальность", value=specialty_value, placeholder="Акушер-гинеколог")
            profile_saved = st.form_submit_button("Сохранить", type="primary")

        if profile_saved:
            new_full_name = " ".join(part for part in [last_name.strip(), first_name.strip()] if part)
            users.loc[row_index, "full_name"] = new_full_name or str(account.get("username", ""))
            users.loc[row_index, "email"] = email.strip()
            users.loc[row_index, "phone"] = phone.strip()
            users.loc[row_index, "clinic"] = clinic.strip()
            users.loc[row_index, "specialty"] = specialty.strip()
            write_users(users)
            st.session_state["auth_user"] = public_user(users.loc[row_index])
            log_event("profile_updated", target=str(account.get("username", "")), details=f"ФИО: {new_full_name}")
            st.success("Профиль обновлён.")
            st.rerun()
        st.markdown('<div class="form-section">Смена пароля</div>', unsafe_allow_html=True)
        st.markdown('<p class="card-sub">Пароль хранится только в виде хэша.</p>', unsafe_allow_html=True)
        with st.form("password_settings_form"):
            current_password = st.text_input("Текущий пароль", type="password")
            password_col, confirm_col = st.columns(2)
            with password_col:
                new_password = st.text_input("Новый пароль", type="password")
            with confirm_col:
                new_password_confirm = st.text_input("Подтверждение пароля", type="password")
            password_saved = st.form_submit_button("Изменить пароль")

        if password_saved:
            if not verify_password(current_password, str(account.get("password_hash", ""))):
                st.error("Текущий пароль указан неверно.")
            elif len(new_password) < 6:
                st.error("Новый пароль должен быть не короче 6 символов.")
            elif new_password != new_password_confirm:
                st.error("Пароли не совпадают.")
            else:
                users.loc[row_index, "password_hash"] = hash_password(new_password)
                write_users(users)
                log_event("password_changed", target=str(account.get("username", "")))
                st.success("Пароль обновлён.")

    with right, st.container(border=True):
        card_head("Уведомления", "каналы оповещения врача")
        notify_email = st.toggle(
            "Email-уведомления",
            value=str(account.get("notify_email") or "") in {"1", "true", "True"},
            help="Получать отчёты и алерты по email",
        )
        notify_sms = st.toggle(
            "SMS-уведомления",
            value=str(account.get("notify_sms") or "") in {"1", "true", "True"},
            help="Экстренные алерты о высоком риске",
        )
        if st.button("Сохранить уведомления", width="stretch"):
            users.loc[row_index, "notify_email"] = "1" if notify_email else ""
            users.loc[row_index, "notify_sms"] = "1" if notify_sms else ""
            write_users(users)
            st.success("Настройки уведомлений сохранены.")
        st.markdown(
            '<div class="clinical-note">Выбранные каналы сохраняются в профиле: отправка писем и SMS '
            "подключается на этапе пилота с клиникой.</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="form-section">Язык интерфейса</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="lang-row"><span class="active">Русский</span>'
            '<span class="off">O\'zbek · скоро</span><span class="off">English · скоро</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="clinical-note">Интерфейс сейчас доступен на русском языке; узбекский и английский — в работе.</div>',
            unsafe_allow_html=True,
        )


def render_admin_page(user: dict[str, str]) -> None:
    users = users_df()
    active_count = int((users["status"] == "active").sum()) if not users.empty else 0
    admin_count = int((users["role"] == "admin").sum()) if not users.empty else 0
    never_logged = int((users["last_login_at"] == "").sum()) if not users.empty else 0

    metric_cards(
        [
            ("Аккаунтов", str(len(users))),
            ("Активных", str(active_count)),
            ("Админов", str(admin_count)),
            ("Без входа", str(never_logged)),
        ]
    )

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        with st.form("create_user_form"):
            st.subheader("Создать аккаунт")
            username = st.text_input("Логин", placeholder="doctor01")
            full_name = st.text_input("ФИО / отделение", placeholder="Акушер-гинеколог")
            role_label = st.selectbox("Роль", list(ROLE_VALUES.keys()))
            status_label = st.selectbox("Статус", list(STATUS_VALUES.keys()))
            password = st.text_input("Временный пароль", type="password")
            password_confirm = st.text_input("Повторите пароль", type="password")
            submitted = st.form_submit_button("Создать аккаунт")

        if submitted:
            if password != password_confirm:
                st.error("Пароли не совпадают.")
            else:
                ok, message = create_user_account(
                    username,
                    full_name,
                    ROLE_VALUES[role_label],
                    STATUS_VALUES[status_label],
                    password,
                )
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with st.form("manage_user_form"):
            st.subheader("Изменить доступ")
            options = users["username"].astype(str).tolist()
            selected_username = st.selectbox("Аккаунт", options)
            selected_row = users.loc[users["username"] == selected_username].iloc[0]
            current_role_label = ROLE_LABELS.get(selected_row["role"], "Врач")
            current_status_label = STATUS_LABELS.get(selected_row["status"], "Активен")
            new_role_label = st.selectbox(
                "Новая роль",
                list(ROLE_VALUES.keys()),
                index=list(ROLE_VALUES.keys()).index(current_role_label),
            )
            new_status_label = st.selectbox(
                "Новый статус",
                list(STATUS_VALUES.keys()),
                index=list(STATUS_VALUES.keys()).index(current_status_label),
            )
            new_password = st.text_input("Новый пароль, если нужно сбросить", type="password")
            updated = st.form_submit_button("Сохранить изменения")

        if updated:
            new_role = ROLE_VALUES[new_role_label]
            new_status = STATUS_VALUES[new_status_label]
            if selected_username == user["username"] and (new_role != "admin" or new_status != "active"):
                st.error("Нельзя снять права или отключить текущий админ-аккаунт.")
            elif new_password and len(new_password) < 6:
                st.error("Новый пароль должен быть не короче 6 символов.")
            else:
                updated_users = users.copy()
                mask = updated_users["username"] == selected_username
                updated_users.loc[mask, "role"] = new_role
                updated_users.loc[mask, "status"] = new_status
                if new_password:
                    updated_users.loc[mask, "password_hash"] = hash_password(new_password)
                active_admins = updated_users.loc[
                    (updated_users["role"] == "admin") & (updated_users["status"] == "active")
                ]
                if active_admins.empty:
                    st.error("Должен остаться хотя бы один активный администратор.")
                else:
                    write_users(updated_users)
                    log_event(
                        "account_updated",
                        target=selected_username,
                        details=(
                            f"Роль: {ROLE_LABELS.get(new_role, new_role)}, "
                            f"статус: {STATUS_LABELS.get(new_status, new_status)}"
                            + (", пароль сброшен" if new_password else "")
                        ),
                    )
                    st.success("Доступ обновлён.")
                    st.rerun()

    with right, st.container(border=True):
        card_head("Мониторинг аккаунтов", "пароли хранятся только в виде хэшей")
        display = users[["username", "full_name", "role", "status", "created_at", "last_login_at"]].copy()
        display["role"] = display["role"].map(lambda value: ROLE_LABELS.get(value, value))
        display["status"] = display["status"].map(lambda value: STATUS_LABELS.get(value, value))
        display["last_login_at"] = display["last_login_at"].replace("", "Никогда")
        display = display.rename(
            columns={
                "username": "Логин",
                "full_name": "Имя",
                "role": "Роль",
                "status": "Статус",
                "created_at": "Создан",
                "last_login_at": "Последний вход",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)
        st.markdown(
            '<div class="clinical-note">Пароли не показываются: в таблице хранится только хэш. '
            "Для врача можно выдать временный пароль и заменить его через сброс.</div>",
            unsafe_allow_html=True,
        )


ADMIN_TABS = ["Управление доступом", "Аналитика врачей", "Журнал действий"]


def render_admin_workspace(user: dict[str, str]) -> None:
    """Админ-панель: управление доступами, аналитика врачей и журнал действий в одном месте."""
    users = users_df()
    visits = visits_with_patient_context()
    attachments = attachments_df()
    events = activity_df()
    doctors = users.loc[users["role"] == "doctor"] if not users.empty else users
    active_doctors = int((doctors["status"] == "active").sum()) if not doctors.empty else 0
    high_count = int((visits["risk_label"] == "high risk").sum()) if not visits.empty else 0
    full_name = user.get("full_name") or user.get("username", "")

    page_head("Админ-панель", f"{full_name}, управление доступами, аналитика врачей и журнал действий")
    kpi_cards(
        [
            ("va-icon-users", "Аккаунтов", str(len(users)), f"{len(doctors)} врачей", False),
            ("va-icon-shield", "Активных врачей", str(active_doctors), "с доступом", False),
            ("va-icon-file", "Расчётов всего", str(len(visits)), f"высокий риск {high_count}", True),
            ("va-icon-chart", "Событий в журнале", str(len(events)), f"{len(attachments)} вложений", False),
        ]
    )

    tabs = st.tabs(ADMIN_TABS)
    with tabs[0]:
        st.markdown(
            '<div class="tab-note">Создание аккаунтов, роли, статусы и сброс паролей врачей.</div>',
            unsafe_allow_html=True,
        )
        render_admin_page(user)
    with tabs[1]:
        st.markdown(
            '<div class="tab-note">Расчёты по врачам, распределение риска и динамика по неделям.</div>',
            unsafe_allow_html=True,
        )
        render_admin_analytics_page()
    with tabs[2]:
        st.markdown(
            '<div class="tab-note">Входы в систему, последние расчёты и загруженные файлы.</div>',
            unsafe_allow_html=True,
        )
        render_admin_activity_page()


def main() -> None:
    add_style()
    ensure_store()
    ensure_auth_store()

    user = current_user()
    if user is None:
        render_login_page()
        return
    user = refresh_current_user(user)
    if user is None:
        st.warning("Сессия завершена. Войдите снова.")
        render_login_page()
        return

    bundle: dict = {"metrics": {}}
    if not is_admin(user):
        try:
            bundle = load_bundle()
        except FileNotFoundError:
            st.error("Модель не найдена. Запустите обучение: python src/train_model.py")
            st.stop()

    pages = (
        ["Админ-панель", "Настройки"]
        if is_admin(user)
        else ["Дашборд", "Новый расчёт", "Пациенты", "Отчёты", "Настройки"]
    )
    if st.session_state.get(NAV_KEY) not in pages:
        st.session_state[NAV_KEY] = pages[0]

    page = st.session_state.get(NAV_KEY) or pages[0]
    render_sidebar(user, pages)
    home_page = "Админ-панель" if is_admin(user) else "Дашборд"
    crumbs: list[tuple[str, str | None]] = [(home_page, home_page), (page, None)]
    if page == home_page:
        crumbs = [(page, None)]
    if page == "Пациенты" and st.session_state.get(PATIENT_VIEW_KEY):
        crumbs = [(home_page, home_page), ("Пациенты", "Пациенты"), ("История пациента", None)]
    query = render_topbar(user, page, crumbs)
    if query:
        matched_patient = find_patient_by_query(query)
        if matched_patient:
            st.session_state[PATIENT_VIEW_KEY] = matched_patient
            st.session_state[NAV_KEY] = "Пациенты"
            st.rerun()
        else:
            st.info(f"Пациент по запросу «{query}» не найден.")

    if is_admin(user):
        if page == "Настройки":
            render_settings_page(user)
        else:
            render_admin_workspace(user)
        return

    if page == "Дашборд":
        render_home_page(bundle, user)
    elif page == "Новый расчёт":
        render_intake_page(bundle, user)
    elif page == "Пациенты":
        render_patients_page(user)
    elif page == "Отчёты":
        render_dashboard_page()
    elif page == "Настройки":
        render_settings_page(user)
    else:
        render_home_page(bundle, user)


if __name__ == "__main__":
    main()
