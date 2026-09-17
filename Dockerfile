# Образ кабинета VascularAI (Streamlit).
# Подходит для Railway, Fly.io, Google Cloud Run, Render (Docker) и обычного VPS.
# Данные живут во внешней БД (Supabase/Turso), поэтому диск контейнера не нужен.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp нужен xgboost, curl — для проверки здоровья
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8501}/_stcore/health" || exit 1

# Порт из переменной PORT (так делают Railway и Cloud Run); локально это 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
