"""Create server secrets once: python tools/setup_server.py (never overwrites .env)."""
from pathlib import Path
import os
import secrets

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT / '.env'
    if destination.exists():
        print('.env already exists; existing passwords have been preserved.')
        return
    contents = '\n'.join([
        '# Generated server secrets. Keep this file private and back it up.',
        f'POSTGRES_PASSWORD={secrets.token_hex(32)}',
        f'ADMIN_PASSWORD={secrets.token_hex(16)}',
        f'COOKIE_SECRET={secrets.token_hex(32)}',
        '',
    ])
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as output:
        output.write(contents)
    print('Created .env. Initial login: admin. Read ADMIN_PASSWORD from .env on the server.')
    print('Next: docker compose up -d --build')


if __name__ == '__main__':
    main()
