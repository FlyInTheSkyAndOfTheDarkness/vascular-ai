"""Consistent application backup: temporarily stops cabinet writes, restarts in finally."""
from datetime import datetime, timezone
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-validation', action='store_true', help='Back up only the isolated local validation stack.')
    args = parser.parse_args()
    destination = ROOT / 'backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination.mkdir(parents=True, exist_ok=False)
    compose = ['docker', 'compose']
    if args.local_validation:
        compose += ['-p', 'vascularai-validation', '-f', 'compose.yaml', '-f', 'deploy/compose.local.yaml']
    subprocess.run(compose + ['stop', 'cabinet'], cwd=ROOT, check=True)
    try:
        with (destination / 'database.dump').open('wb') as output:
            subprocess.run(compose + ['exec', '-T', 'db', 'pg_dump', '-U', 'vascularai',
                                     '-d', 'vascularai', '-Fc'], cwd=ROOT, stdout=output, check=True)
        with (destination / 'attachments.tar.gz').open('wb') as output:
            subprocess.run(compose + ['run', '--rm', '--no-deps', '-T', 'cabinet',
                                     'tar', '-czf', '-', '-C', '/app/data/clinic', 'attachments'],
                           cwd=ROOT, stdout=output, check=True)
    finally:
        subprocess.run(compose + ['start', 'cabinet'], cwd=ROOT, check=True)
    (destination / 'COMPLETE').write_text('Database and attachments saved. Keep .env separately.\n', encoding='utf-8')
    print(f'Backup completed: {destination}')


if __name__ == '__main__':
    main()
