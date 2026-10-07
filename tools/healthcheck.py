"""Read-only container readiness: HTTP, database and required schema."""
import os
import urllib.request
from sqlalchemy import create_engine, inspect, text


def main():
    with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4) as response:
        if response.status != 200:
            return 1
    engine = create_engine(os.environ['DATABASE_URL'], connect_args={'connect_timeout': 4})
    try:
        with engine.connect() as connection:
            connection.execute(text('SELECT 1'))
            schema = inspect(connection)
            if not {'patients', 'visits', 'users', 'activity', 'attachments'} <= set(schema.get_table_names()):
                return 1
            if 'gestational_week' not in {column['name'] for column in schema.get_columns('visits')}:
                return 1
    finally:
        engine.dispose()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('Application or database is not ready.')
        raise SystemExit(1)
