"""Run on SQLite by default; TEST_DATABASE_URL enables isolated PostgreSQL schemas."""
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

import pandas as pd
from sqlalchemy import create_engine, inspect, text

import app


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        server = os.environ.get('TEST_DATABASE_URL')
        if server:
            self.schema = 'va_test_' + uuid.uuid4().hex
            base = create_engine(server)
            with base.begin() as conn:
                conn.execute(text(f'CREATE SCHEMA "{self.schema}"'))
            def cleanup():
                with base.begin() as conn:
                    conn.execute(text(f'DROP SCHEMA "{self.schema}" CASCADE'))
                base.dispose()
            self.stack.callback(cleanup)
            self.engine = create_engine(server, connect_args={'options': f'-csearch_path={self.schema}'})
        else:
            temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
            self.engine = create_engine('sqlite:///' + str(Path(temporary) / 'test.db'))
        self.stack.callback(self.engine.dispose)
        self.state = {}
        self.ready = set()
        self.stack.enter_context(patch.object(app, 'engine_or_none', return_value=self.engine))
        self.stack.enter_context(patch.object(app, 'db_engine', return_value=self.engine))
        self.stack.enter_context(patch.object(app, 'database_url', return_value=str(self.engine.url)))
        self.stack.enter_context(patch.object(app, 'state_obj', side_effect=lambda key, factory: self.state.setdefault(key, factory())))
        self.stack.enter_context(patch.object(app, 'tables_ready', return_value=self.ready))
        self.stack.enter_context(patch.object(app, 'secret_value', side_effect=lambda name, default='': default))
        app.read_table_cached.clear()
        self.stack.callback(app.read_table_cached.clear)
        app.ensure_store();app.ensure_auth_store()
        admin=app.public_user(app.users_df().iloc[0])
        self.stack.enter_context(patch.object(app, 'current_user', return_value=admin))
        app.append_patient({'patient_id':'TEST-BASE'})

    def test_migration_preserves_old_visits_and_adds_week(self):
        from sqlalchemy import text
        with self.engine.begin() as conn: conn.execute(text('DROP TABLE visits'))
        self.ready.clear()
        app.append_patient({'patient_id':'PT-0001'})
        legacy = [column for column in app.VISIT_COLUMNS if column != 'gestational_week']
        with self.engine.begin() as conn:
            definition = ', '.join(f'"{column}" TEXT' for column in legacy)
            conn.execute(text(f'CREATE TABLE visits ({definition})'))
            conn.execute(text("INSERT INTO visits (visit_id, patient_id) VALUES ('old', 'PT-0001')"))
        app.ensure_store()
        app.ensure_store()
        with self.engine.connect() as conn:
            self.assertIn('gestational_week', {c['name'] for c in inspect(conn).get_columns('visits')})
        self.assertEqual(app.visits_df().iloc[0]['visit_id'], 'old')
        app.append_visit({'patient_id':'TEST-BASE','visit_id': 'new', 'patient_id': 'PT-0001', 'gestational_week': '25'})
        self.assertEqual(len(app.visits_df()), 2)
        self.assertEqual(app.visits_df().iloc[-1]['gestational_week'], '25')

    def test_stale_snapshot_cannot_remove_another_sessions_new_visit(self):
        app.ensure_store()
        app.append_visit({'patient_id':'TEST-BASE','visit_id': 'one'})
        stale = app.visits_df()
        app.append_visit({'patient_id':'TEST-BASE','visit_id': 'two'})
        stale.loc[0, 'visit_note'] = 'updated'
        app.write_csv(stale, app.VISITS_PATH, app.VISIT_COLUMNS)
        self.assertEqual(set(app.visits_df().visit_id), {'one', 'two'})

    def test_conflicting_updates_are_rejected_without_losing_new_value(self):
        app.ensure_store()
        app.ensure_auth_store()
        first = app.users_df()
        stale = app.users_df()
        first.loc[0, 'full_name'] = 'First edit'
        app.write_users(first)
        stale.loc[0, 'full_name'] = 'Stale edit'
        with self.assertRaises(app.StorageError):
            app.write_users(stale)
        self.assertEqual(app.users_df().iloc[0]['full_name'], 'First edit')

    def test_duplicate_id_is_rejected(self):
        app.ensure_store()
        app.append_patient({'patient_id': 'PT-0001'})
        with self.assertRaises(app.StorageError):
            app.append_patient({'patient_id': 'PT-0001'})
        self.assertEqual(len(app.patients_df()), 2)

    def test_conflicting_batch_rolls_back_prior_updates(self):
        app.ensure_store()
        app.append_visit({'patient_id':'TEST-BASE','visit_id': 'one'})
        app.append_visit({'patient_id':'TEST-BASE','visit_id': 'two'})
        stale = app.visits_df()
        current = app.visits_df()
        current.loc[current.visit_id == 'two', 'visit_note'] = 'Other session'
        app.write_csv(current, app.VISITS_PATH, app.VISIT_COLUMNS)
        stale['visit_note'] = 'Stale batch'
        with self.assertRaises(app.StorageError):
            app.write_csv(stale, app.VISITS_PATH, app.VISIT_COLUMNS)
        saved = app.visits_df().set_index('visit_id')
        self.assertEqual(saved.loc['one', 'visit_note'], '')
        self.assertEqual(saved.loc['two', 'visit_note'], 'Other session')

    def test_failed_read_does_not_return_empty_data(self):
        with patch.object(app, 'read_table_cached', side_effect=OSError('offline')):
            with self.assertRaises(app.StorageError):
                app.users_df()

    def test_failed_transaction_rolls_back_and_raises(self):
        app.ensure_store()
        app.append_visit({'patient_id':'TEST-BASE','visit_id': 'keep'})
        frame = app.visits_df()
        frame.loc[0, 'visit_note'] = 'new note'
        with patch.object(self.engine, 'begin', side_effect=OSError('offline')):
            with self.assertRaises(app.StorageError):
                app.write_csv(frame, app.VISITS_PATH, app.VISIT_COLUMNS)
        self.assertEqual(app.visits_df().iloc[0]['visit_note'], '')


class DeploymentSettingsTests(unittest.TestCase):
    def test_configured_database_failure_never_falls_back_to_csv(self):
        with patch.object(app, 'database_url', return_value='postgresql+psycopg://invalid'), \
             patch.object(app, 'db_status', return_value={}), \
             patch.object(app, 'db_engine', side_effect=OSError('offline')):
            with self.assertRaises(app.StorageError):
                app.engine_or_none()

    def test_production_requires_persistent_database_and_attachments(self):
        with patch.object(app, 'production_mode', return_value=True), \
             patch.object(app, 'database_url', return_value=''):
            with self.assertRaises(app.StorageError):
                app.check_deployment_settings()
        with patch.object(app, 'production_mode', return_value=True), \
             patch.object(app, 'database_url', return_value='postgresql+psycopg://db'), \
             patch.object(app, 'attachments_in_cloud', return_value=False), \
             patch.object(app, 'secret_value', return_value='0'):
            with self.assertRaises(app.StorageError):
                app.check_deployment_settings()

    def test_production_bootstrap_requires_nondefault_password(self):
        with patch.object(app, 'production_mode', return_value=True), \
             patch.object(app, 'secret_value', return_value='admin123'):
            with self.assertRaises(app.StorageError):
                app.seed_admin_row()


if __name__ == '__main__':
    unittest.main()
