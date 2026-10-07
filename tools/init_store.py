"""Initialize/migrate the configured database before accepting browser sessions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app

app.check_deployment_settings()
app.ensure_store()
app.ensure_auth_store()
app.load_bundle()
print('Database, initial administrator and model are ready.')
