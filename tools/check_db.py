"""Проверка подключения к внешней БД (Supabase / Turso) до деплоя.

Как пользоваться (PowerShell):

    $env:DATABASE_URL="postgresql://postgres.<ref>:<пароль>@aws-0-<регион>.pooler.supabase.com:5432/postgres"
    python tools/check_db.py

Либо положите строку в `.streamlit/secrets.toml` (файл в .gitignore) и запустите без переменной.
Скрипт не печатает пароль.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent


def load_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        return url
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        match = re.search(r'DATABASE_URL\s*=\s*["\']([^"\']+)["\']', secrets.read_text(encoding="utf-8"))
        if match:
            return match.group(1).strip()
    return ""


def to_sqlalchemy_url(url: str) -> str:
    """Те же преобразования, что делает приложение."""
    if url.startswith(("libsql://", "http://", "https://")):
        url = "sqlite+libsql://" + url.split("://", 1)[1]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://") :]
    if url.startswith("postgresql+psycopg://") and "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def main() -> int:
    raw = load_url()
    if not raw:
        print("DATABASE_URL не найден: задайте переменную окружения или .streamlit/secrets.toml")
        return 2

    # Разбор для понятных подсказок (пароль не выводим).
    clean = raw.replace("postgresql+psycopg://", "postgresql://").replace("postgres://", "postgresql://")
    parts = urlsplit(clean)
    user = (parts.username or "")
    password = (parts.password or "")
    host = parts.hostname or ""
    port = parts.port or ""
    print(f"host: {host}:{port} | user: {user} | пароль: {'задан' if password else 'ПУСТОЙ'}")

    hints: list[str] = []
    if "pooler.supabase.com" in host:
        if "." not in user:
            hints.append(
                "Для Session pooler имя пользователя должно быть с кодом проекта: "
                "postgres.<ref> (например postgres.qroufctfsccgjmtnebqb), а не просто postgres."
            )
        if "[YOUR-PASSWORD]" in raw or password in ("", "YOUR-PASSWORD"):
            hints.append("Похоже, в строке остался плейсхолдер пароля — вставьте реальный пароль базы.")
    for character in "@:#/?%":
        if character in password:
            hints.append(
                f"В пароле есть символ «{character}» — его нужно закодировать в URL "
                "(например @ → %40) или сбросить пароль на состоящий из букв и цифр."
            )
            break

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("Нет SQLAlchemy: python -m pip install -r requirements.txt")
        return 2

    url = to_sqlalchemy_url(raw)
    driver = url.split("://", 1)[0].split("+")[-1]
    try:
        engine = create_engine(url, pool_pre_ping=True, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as error:
        print(f"\n✗ Подключиться не удалось ({type(error).__name__}): {str(error)[:400]}")
        print(f"\nУстановлен ли драйвер «{driver}»? postgres → psycopg[binary], libsql → sqlalchemy-libsql")
        for hint in hints:
            print(f"• {hint}")
        return 1

    print("\n✓ Подключение работает — строку можно вставлять в Secrets")
    try:
        with engine.connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar()
        print(f"  {str(version)[:80]}")
    except Exception:
        print(f"  (версию сервера получить не удалось, но соединение установлено)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
