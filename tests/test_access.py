import io
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import app
import cabinet_ui
from tests.test_storage_db import DatabaseTests


class AccessTests(unittest.TestCase):
    def setUp(self):
        DatabaseTests.setUp(self)
        from sqlalchemy import text
        with self.engine.begin() as conn:conn.execute(text('DELETE FROM patients'))
        app.bump_data_revision()

    def actors(self):
        app.ensure_store();app.ensure_auth_store()
        for username in ['doctor_a','doctor_b']:
            app.create_user_account(username,'Same Name','doctor','active','synthetic123')
        self.accounts=app.users_df().set_index('username').to_dict('index')
        for name,user in self.accounts.items():user['username']=name
        self.actor=self.accounts['doctor_a']
        self.stack.enter_context(patch.object(app,'current_user',side_effect=lambda:self.actor))

    def populate(self):
        self.actors()
        app.append_patient({'patient_id':'A','initials':'PRIVATE A','owner_user_id':self.accounts['doctor_b']['user_id'],'doctor':'Same Name'})
        app.append_visit({'visit_id':'VA','patient_id':'A','visit_note':'PRIVATE VISIT'})
        app.append_attachment({'attachment_id':'FA','patient_id':'A','stored_path':'private.txt','filename':'PRIVATE FILE'})
        app.log_event('patient_created',target='A',actor=self.actor)
        self.actor=self.accounts['doctor_b']

    def test_second_doctor_starts_empty_and_cannot_read_raw_clinical_tables(self):
        self.populate()
        for path,cols in [(app.PATIENTS_PATH,app.PATIENT_COLUMNS),(app.VISITS_PATH,app.VISIT_COLUMNS),(app.ATTACHMENTS_PATH,app.ATTACHMENT_COLUMNS),(app.ACTIVITY_PATH,app.ACTIVITY_COLUMNS)]:
            frame=app.read_csv(path,cols)
            self.assertTrue(frame.empty)
            self.assertEqual(frame.attrs.get('original_rows'),{})
        app.append_patient({'patient_id':'B'})
        self.assertEqual(app.patients_df().patient_id.tolist(),['B'])
        self.actor=self.accounts['doctor_a']
        self.assertEqual(app.patients_df().patient_id.tolist(),['A'])
        self.assertEqual(app.patients_df().iloc[0].owner_user_id,self.actor['user_id'])
        self.assertEqual(app.visits_df().visit_id.tolist(),['VA'])

    def test_direct_foreign_writes_uploads_and_downloads_are_denied(self):
        self.populate()
        with self.assertRaises(PermissionError):app.append_visit({'visit_id':'evil','patient_id':'A'})
        with self.assertRaises(PermissionError):app.append_attachment({'attachment_id':'evil','patient_id':'A'})
        upload=io.BytesIO(b'synthetic');upload.name='test.txt'
        with self.assertRaises(app.StorageError):app.save_uploaded_attachment('A',upload,'')
        row=app.read_csv_unscoped(app.ATTACHMENTS_PATH,app.ATTACHMENT_COLUMNS).iloc[0]
        with self.assertRaises(PermissionError):cabinet_ui.attachment_bytes(app,row)
        foreign=app.read_csv_unscoped(app.PATIENTS_PATH,app.PATIENT_COLUMNS)
        foreign['note']='tampered'
        with self.assertRaises(PermissionError):app.write_csv(foreign,app.PATIENTS_PATH,app.PATIENT_COLUMNS)
        self.assertEqual(len(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS)),1)

    def test_owner_cannot_be_changed_or_spoofed_and_same_name_gives_no_access(self):
        self.populate();self.actor=self.accounts['doctor_a']
        row=app.patients_df();row['owner_user_id']=self.accounts['doctor_b']['user_id']
        with self.assertRaises(PermissionError):app.write_csv(row,app.PATIENTS_PATH,app.PATIENT_COLUMNS)
        self.actor=self.accounts['doctor_b'];self.actor['role']='admin'
        self.assertTrue(app.patients_df().empty)

    def test_admin_can_read_all_but_anonymous_and_blocked_cannot(self):
        self.populate();self.actor=self.accounts['admin']
        self.assertEqual(app.patients_df().patient_id.tolist(),['A'])
        self.actor=None
        self.assertTrue(app.patients_df().empty)
        with self.assertRaises(PermissionError):app.append_patient({'patient_id':'evil'})
        self.actor=self.accounts['doctor_a']
        users=app.users_df();users.loc[users.username=='doctor_a','status']='blocked';app.write_users(users)
        self.assertTrue(app.patients_df().empty)

    def test_migration_uses_creation_event_only_and_is_idempotent(self):
        self.actors()
        for pid in ['known','unknown','ambiguous']:
            app.append_patient({'patient_id':pid,'doctor':'Same Name'})
        engine=app.engine_or_none()
        if engine is not None:
            from sqlalchemy import text
            with engine.begin() as conn:conn.execute(text("UPDATE patients SET owner_user_id=''"))
            app.bump_data_revision()
        else:
            rows=app.read_csv_unscoped(app.PATIENTS_PATH,app.PATIENT_COLUMNS);rows['owner_user_id']='';rows.to_csv(app.PATIENTS_PATH,index=False)
        app.log_event('patient_created',target='known',actor=self.actor)
        app.log_event('patient_created',target='ambiguous',actor=self.actor)
        app.log_event('patient_created',target='ambiguous',actor=self.accounts['doctor_b'])
        self.assertEqual(app.migrate_patient_owners(),{'assigned':1,'unassigned':2,'total':3})
        self.assertEqual(app.migrate_patient_owners()['assigned'],0)
        self.assertEqual(app.patients_df().patient_id.tolist(),['known'])
        self.actor=self.accounts['doctor_b'];self.assertTrue(app.patients_df().empty)
        self.actor=self.accounts['admin'];self.assertEqual(len(app.patients_df()),3)

    def test_scoped_csv_edit_preserves_another_doctors_rows(self):
        self.populate();app.append_patient({'patient_id':'B'})
        own=app.patients_df();own['note']='OWN';app.write_csv(own,app.PATIENTS_PATH,app.PATIENT_COLUMNS)
        raw=app.read_csv_unscoped(app.PATIENTS_PATH,app.PATIENT_COLUMNS)
        self.assertEqual(set(raw.patient_id),{'A','B'})
        self.assertEqual(raw.set_index('patient_id').loc['A','note'],'')

    def test_identity_switch_discards_results_forms_and_notices(self):
        state={'_clinical_identity':('A','doctor'),'intake_result':{'private':True},'patient_view':'A','_notice':'private','ui_language':'kk','auth_user':{'user_id':'B','role':'doctor'}}
        with patch.object(app.st,'session_state',state):app.reset_clinical_session(state['auth_user'])
        self.assertNotIn('intake_result',state);self.assertNotIn('patient_view',state);self.assertNotIn('_notice',state)
        self.assertEqual(state['ui_language'],'kk')

    def test_legacy_foreign_activity_is_not_leaked_in_notifications(self):
        self.populate()
        app.log_event('visit_calculated',target='A',actor=self.actor)
        self.assertTrue(app.activity_df().empty)


class CsvAccessTests(AccessTests):
    def setUp(self):
        super().setUp()
        root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(app,'engine_or_none',return_value=None))
        self.stack.enter_context(patch.multiple(app,CLINIC_DIR=root/'clinic',AUTH_DIR=root/'auth',
            PATIENTS_PATH=root/'clinic/patients.csv',VISITS_PATH=root/'clinic/visits.csv',
            ATTACHMENTS_PATH=root/'clinic/attachments.csv',ACTIVITY_PATH=root/'clinic/activity.csv',USERS_PATH=root/'auth/users.csv'))
        app.ensure_store();app.ensure_auth_store()


if __name__=='__main__':unittest.main()
