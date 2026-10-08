"""Local checks: python -m unittest discover -s tests -v.

Storage is redirected to a temporary directory; real clinic data is never changed.
"""
from contextlib import ExitStack
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest
import xgboost as xgb

import app
from src.train_model import CLASS_TO_ID, FEATURES, TARGET, load_dataset


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(app.MODEL_PATH)
        cls.data = load_dataset()

    def test_saved_holdout_metrics_are_reproducible(self):
        _, x_test, _, y_test = train_test_split(
            self.data[FEATURES], self.data[TARGET].map(CLASS_TO_ID),
            test_size=0.2, random_state=42, stratify=self.data[TARGET].map(CLASS_TO_ID),
        )
        predicted = self.bundle['model'].predict(x_test)
        self.assertAlmostEqual(accuracy_score(y_test, predicted), self.bundle['metrics']['accuracy'])
        self.assertEqual(confusion_matrix(y_test, predicted).tolist(), self.bundle['metrics']['confusion_matrix'])

    def test_probabilities_are_finite_normalized_and_match_classes(self):
        model = self.bundle['model']
        probs = model.predict_proba(self.data[FEATURES])
        self.assertEqual(probs.shape, (len(self.data), 3))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(probs.argmax(axis=1), model.predict(self.data[FEATURES]))
        self.assertEqual(self.bundle['class_names'], list(CLASS_TO_ID))

    def test_shap_reconstructs_model_margins(self):
        model = self.bundle['model']
        patient = self.data[FEATURES].iloc[:50]
        matrix = xgb.DMatrix(patient)
        booster = model.get_booster()
        contribs = booster.predict(matrix, pred_contribs=True)
        margins = booster.predict(matrix, output_margin=True)
        np.testing.assert_allclose(contribs.sum(axis=2), margins, atol=5e-6)
        for class_index in range(3):
            actual = app.prediction_shap_values(model, patient.iloc[:1], class_index)
            np.testing.assert_allclose(actual, contribs[0, class_index, :-1])

    def test_application_predictions_match_batch_model(self):
        sample = self.data[FEATURES].sample(30, random_state=42)
        expected = self.bundle['model'].predict_proba(sample)
        for values, probs in zip(sample.to_dict('records'), expected):
            result = app.predict_risk(values, self.bundle)
            np.testing.assert_allclose(result['probabilities'], probs)
            self.assertEqual(result['class_name'], self.bundle['class_names'][int(probs.argmax())])
            self.assertEqual(len(result['shap_values']), 6)


class IsolatedStore(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=app.PROJECT_ROOT / 'data')))
        clinic, auth = root / 'clinic', root / 'auth'
        self.stack.enter_context(patch.multiple(
            app, CLINIC_DIR=clinic, AUTH_DIR=auth, ATTACHMENTS_DIR=clinic / 'attachments',
            PATIENTS_PATH=clinic / 'patients.csv', VISITS_PATH=clinic / 'visits.csv',
            ATTACHMENTS_PATH=clinic / 'attachments.csv', ACTIVITY_PATH=clinic / 'activity.csv',
            USERS_PATH=auth / 'users.csv',
        ))
        self.stack.enter_context(patch.object(app, 'engine_or_none', return_value=None))
        self.stack.enter_context(patch.object(app, 'database_url', return_value=''))
        self.stack.enter_context(patch.object(app, 'attachments_in_cloud', return_value=False))
        self.stack.enter_context(patch.object(app, 'secret_value', side_effect=lambda name, default='': default))
        app.ensure_store()
        app.ensure_auth_store()

    def patient(self, patient_id='PT-0001', age='29'):
        row = dict.fromkeys(app.PATIENT_COLUMNS, '')
        row.update(patient_id=patient_id, initials='TEST', age=age, gestational_week='12',
                   doctor='Test Doctor', created_at=app.now_iso(), status='Активна')
        owner = getattr(self, 'patient_owner', app.public_user(app.users_df().iloc[0]))
        with patch.object(app, 'current_user', return_value=owner):app.append_patient(row)

    def doctor(self):
        with patch.object(app, 'log_event'):
            ok, message = app.create_user_account('testdoctor', 'Test Doctor', 'doctor', 'active', 'testpass123')
        self.assertTrue(ok, message)
        return app.authenticate('testdoctor', 'testpass123')


class StorageAuthTests(IsolatedStore):
    def test_password_hash_and_rejection(self):
        hashed = app.hash_password('testpass123')
        self.assertTrue(app.verify_password('testpass123', hashed))
        self.assertFalse(app.verify_password('incorrect', hashed))
        self.assertFalse(app.verify_password('testpass123', 'broken'))
        self.assertNotEqual(hashed, app.hash_password('testpass123'))

    def test_login_and_blocked_user(self):
        user = self.doctor()
        self.assertNotIn('password_hash', user)
        self.assertIsNone(app.authenticate('testdoctor', 'incorrect'))
        self.assertIsNotNone(app.authenticate(' TESTDOCTOR ', 'testpass123'))
        users = app.users_df()
        users.loc[users.username == 'testdoctor', 'status'] = 'blocked'
        app.write_users(users)
        self.assertIsNone(app.authenticate('testdoctor', 'testpass123'))

    def test_sessions_expire_and_reject_unknown_tokens(self):
        user = self.doctor()
        token = app.issue_session_token(user['user_id'])
        self.assertEqual(app.user_from_session_token(token)['username'], 'testdoctor')
        self.assertIsNone(app.user_from_session_token('unknown'))
        users = app.users_df()
        users.loc[users.username == 'testdoctor', 'session_expires_at'] = '2000-01-01 00:00:00'
        app.write_users(users)
        self.assertIsNone(app.user_from_session_token(token))

    def test_patient_round_trip_and_ids(self):
        self.patient()
        self.assertEqual(app.read_csv_unscoped(app.PATIENTS_PATH,app.PATIENT_COLUMNS).iloc[0]['age'], '29')
        self.assertEqual(app.next_patient_id(app.read_csv_unscoped(app.PATIENTS_PATH,app.PATIENT_COLUMNS)), 'PT-0002')

    def test_latest_visit_selected_by_date(self):
        rows = [dict(patient_id='PT-0001', recorded_at=date, risk_label=risk,
                     risk_probability='.8', top_factor='Age') for date, risk in
                [('2026-10-05 10:00:00', 'high risk'), ('2026-10-01 10:00:00', 'low risk')]]
        self.assertEqual(app.latest_visits_by_patient(pd.DataFrame(rows)).iloc[0]['risk_label'], 'high risk')

    def test_report_period_filter(self):
        frame = pd.DataFrame({'recorded_at': [app.now_iso(), '2000-01-01 00:00:00', 'invalid']})
        self.assertEqual(len(app.filter_by_period(frame, 'Текущий месяц')), 1)
        self.assertEqual(len(app.filter_by_period(frame, 'За всё время')), 3)

    def test_attachment_round_trip(self):
        self.patient()
        attachment = io.BytesIO(b'synthetic test attachment')
        attachment.name = 'test.txt'
        with patch.object(app, 'log_event'):
            with patch.object(app,'current_user',return_value=app.public_user(app.users_df().iloc[0])):
                app.save_uploaded_attachment('PT-0001', attachment, 'test')
        saved = app.read_csv_unscoped(app.ATTACHMENTS_PATH,app.ATTACHMENT_COLUMNS).iloc[0]
        self.assertEqual(saved['filename'], 'test.txt')
        self.assertEqual((app.PROJECT_ROOT / saved['stored_path']).read_bytes(), attachment.getvalue())

    def test_risk_trend_uses_high_risk_probability(self):
        frame = pd.DataFrame({'recorded_at': ['2026-10-01 10:00:00'],
                              'risk_probability': ['.95'], 'prob_high': ['.02']})
        with patch.object(app.st, 'altair_chart') as rendered:
            app.render_risk_trend(frame)
        chart = rendered.call_args.args[0]
        self.assertEqual(chart.layer[1].data['score'].tolist(), [.02])


class InterfaceTests(IsolatedStore):
    def start(self, role='doctor', patient=True):
        user = self.doctor() if role == 'doctor' else app.authenticate('admin', app.DEFAULT_ADMIN_PASSWORD)
        self.patient_owner = user
        if patient:
            self.patient()
        at = AppTest.from_string('import app\napp.main()', default_timeout=30)
        at.session_state['auth_user'] = user
        at.run()
        self.assertFalse(list(at.exception))
        return at

    def navigate(self, at, page):
        at.session_state[app.NAV_KEY] = page
        at.run()
        self.assertFalse(list(at.exception))
        return at

    def click(self, at, label):
        next(button for button in at.button if button.label == label).click().run()
        self.assertFalse(list(at.exception))

    def measurements(self, at, values=None):
        values = values or [29, 130, 70, 6.7, 98, 78]
        for feature, value in zip(FEATURES, values):
            at.text_input(key='input_'+feature+('_celsius' if feature=='BodyTemp' else '')).set_value(str((value-32)*5/9 if feature=='BodyTemp' else value))

    def test_login_and_invalid_password(self):
        at = AppTest.from_string('import app\napp.main()', default_timeout=30).run()
        at.text_input(key='login_username').set_value('admin')
        at.text_input(key='login_password').set_value('incorrect')
        self.click(at, 'Войти')
        self.assertTrue(list(at.error))
        at.text_input(key='login_password').set_value(app.DEFAULT_ADMIN_PASSWORD)
        self.click(at, 'Войти')
        self.assertEqual(at.session_state['auth_user']['role'], 'admin')

    def test_every_role_page_in_three_languages(self):
        for role in ['doctor', 'admin']:
            at = self.start(role=role, patient=role=='doctor')
            pages = ['Дашборд','Новый расчёт','Пациенты','Отчёты','Настройки'] if role=='doctor' else ['Админ-панель','Настройки']
            for language in ['ru','en','kk']:
                at.selectbox(key='ui_language').set_value(language).run()
                for page in pages:
                    self.navigate(at,page)

    def test_empty_store_patient_creation_updates_view_and_notice(self):
        at = self.navigate(self.start(patient=False), 'Новый расчёт')
        self.click(at,'Создать карту')
        self.assertEqual(len(app.read_csv_unscoped(app.PATIENTS_PATH,app.PATIENT_COLUMNS)),1)
        self.assertEqual(at.session_state[app.NAV_KEY],'Пациенты')
        self.assertTrue(list(at.success))

    def test_missing_measurements_do_not_save(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        self.click(at,'Рассчитать риск')
        self.assertTrue(list(at.error))
        self.assertTrue(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).empty)

    def test_three_live_profiles_and_no_duplicate_on_rerun_or_double_submit(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        for values,label in [([15,78,49,7.5,98,77],'low risk'),([29,130,70,6.7,98,78],'mid risk'),([25,140,100,15,98.6,70],'high risk')]:
            self.measurements(at,values)
            self.click(at,'Рассчитать риск')
            result=at.session_state['intake_result']
            self.assertEqual(result['class_name'],label)
            saved=app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).iloc[-1]
            np.testing.assert_allclose([float(saved[f]) for f in FEATURES],values)
            np.testing.assert_allclose([float(saved[c]) for c in ['prob_low','prob_mid','prob_high']],result['probabilities'],atol=5e-7)
        at.run()
        self.click(at,'Рассчитать риск')
        self.assertEqual(len(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS)),3)
        self.assertTrue(list(at.success))

    def test_invalid_text_never_reuses_old_temperature(self):
        at=self.navigate(self.start(),'Новый расчёт')
        self.measurements(at)
        self.click(at,'Рассчитать риск')
        for value in ['98.6','34.9','40.6','NaN','inf','','not a number']:
            at.text_input(key='input_BodyTemp_celsius').set_value(value)
            with patch.object(app,'append_visit') as saved:
                self.click(at,'Рассчитать риск')
                saved.assert_not_called()
            self.assertTrue(list(at.error))
            self.assertFalse(at.session_state.filtered_state.get('intake_result'))
        self.assertEqual(len(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS)),1)

    def test_celsius_comma_input_converts_once_for_model_and_storage(self):
        at=self.navigate(self.start(),'Новый расчёт');self.measurements(at)
        temperature=at.text_input(key='input_BodyTemp_celsius')
        self.assertIn('°C',temperature.label)
        temperature.set_value('36,6');self.click(at,'Рассчитать риск')
        saved=app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).iloc[-1]
        self.assertAlmostEqual(float(saved['BodyTemp']),97.88)
        result=at.session_state['intake_result']
        self.assertAlmostEqual(float(result['patient'].iloc[0]['BodyTemp']),97.88)
        expected=app.predict_risk(dict(zip(FEATURES,[29,130,70,6.7,97.88,78])),app.load_bundle())
        np.testing.assert_allclose(result['probabilities'],expected['probabilities'])

    def test_celsius_history_and_exports_do_not_modify_fahrenheit_records(self):
        import cabinet_ui
        self.assertEqual(cabinet_ui.display_measurement('BodyTemp','98.6'),'37 °C')
        self.assertEqual(cabinet_ui.display_measurement('BodyTemp','97.88'),'36.6 °C')
        frame=pd.DataFrame([{'BodyTemp':'98.6','risk_label':'low risk'}])
        for lang in ['ru','en','kk']:
            with patch.object(cabinet_ui.st,'session_state',{'ui_language':lang}):
                exported=cabinet_ui.export_frame(frame)
                column=next(c for c in exported.columns if '(°C)' in c)
                self.assertEqual(exported.iloc[0][column],37.0)
                self.assertNotIn('°F',' '.join(exported.columns))
        self.assertEqual(frame.iloc[0]['BodyTemp'],'98.6')

    def test_celsius_bounds_are_valid_for_model(self):
        import cabinet_ui
        for c,f in [('35',95),('40.5',104.9),('37',98.6)]:
            raw=dict(zip(FEATURES,['29','130','70','6.7',c,'78']))
            values,errors=cabinet_ui.validate_measurements(raw,app.FEATURE_META)
            self.assertFalse(errors);self.assertAlmostEqual(values['BodyTemp'],f)

    def test_invalid_pressure_does_not_save(self):
        at=self.navigate(self.start(),'Новый расчёт')
        self.measurements(at,[29,80,100,6.7,98,78])
        self.click(at,'Рассчитать риск')
        self.assertTrue(list(at.error));self.assertTrue(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).empty)

    def test_unsaved_calculation(self):
        at=self.navigate(self.start(),'Новый расчёт')
        self.measurements(at)
        at.checkbox(key='attach_visit').uncheck()
        self.click(at,'Рассчитать риск')
        self.assertTrue(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).empty)
        self.assertIn('class_name',at.session_state['intake_result'])

    def test_patient_switch_resets_measurements_and_result(self):
        at=self.start();self.patient('PT-0002','40')
        self.navigate(at,'Новый расчёт');self.measurements(at);self.click(at,'Рассчитать риск')
        at.selectbox(key='intake_patient').set_value('PT-0002').run()
        self.assertFalse(list(at.exception))
        self.assertEqual(at.text_input(key='input_Age').value,'40')
        self.assertEqual(at.text_input(key='input_BodyTemp_celsius').value,'')
        self.assertFalse(at.session_state.filtered_state.get('intake_result'))

    def test_optional_values_saved_without_changing_prediction(self):
        at=self.navigate(self.start(),'Новый расчёт');self.measurements(at)
        self.click(at,'Рассчитать риск');previous=at.session_state['intake_result']['probabilities'].copy()
        for key,value in [('plgf','15'),('papp_a','0.2'),('sflt1','5000'),('map_value','125'),('bmi','35'),('week','25')]:
            at.text_input(key='extra_'+key).set_value(value)
        at.checkbox(key='extra_preeclampsia').check()
        self.click(at,'Рассчитать риск')
        np.testing.assert_array_equal(at.session_state['intake_result']['probabilities'],previous)
        saved=app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).iloc[-1]
        self.assertEqual(saved['gestational_week'],'25');self.assertEqual(saved['plgf'],'15')
        self.assertIn('преэклампсия',saved['anamnesis'])

    def test_write_failure_never_reports_saved_result(self):
        at=self.navigate(self.start(),'Новый расчёт');self.measurements(at)
        with patch.object(app,'append_visit',side_effect=app.StorageError('Unavailable')):
            self.click(at,'Рассчитать риск')
        self.assertTrue(list(at.error));self.assertFalse(list(at.success));self.assertTrue(app.read_csv_unscoped(app.VISITS_PATH,app.VISIT_COLUMNS).empty)

    def test_admin_account_switch_loads_correct_privileges(self):
        self.doctor();at=self.start(role='admin',patient=False)
        at.selectbox(key='manage_account').set_value('testdoctor').run()
        self.assertEqual(at.selectbox(key='role_testdoctor').value,'doctor')
        at.selectbox(key='manage_account').set_value('admin').run()
        self.assertEqual(at.selectbox(key='role_admin').value,'admin')
        at.selectbox(key='role_admin').set_value('doctor')
        self.click(at,'Сохранить')
        self.assertTrue(list(at.error))
        self.assertEqual(app.users_df().set_index('username').loc['admin','role'],'admin')

    def test_admin_analytics_empty(self):
        self.doctor();at=self.start(role='admin',patient=False)
        at.radio(key='admin_section').set_value('analytics').run()
        self.assertFalse(list(at.exception))

    def test_admin_tab_survives_settings_and_language_change(self):
        at=self.start(role='admin',patient=False)
        at.radio(key='admin_section').set_value('activity').run()
        self.navigate(at,'Настройки')
        at.selectbox(key='ui_language').set_value('kk').run()
        self.navigate(at,'Админ-панель')
        self.assertEqual(at.radio(key='admin_section').value,'activity')

    def test_locale_catalogue_has_matching_placeholders(self):
        from cabinet_i18n import CATALOGUE
        from string import Formatter
        for key,values in CATALOGUE.items():
            self.assertEqual(set(values),{'ru','en','kk'},key)
            placeholders=[{field for _,field,_,_ in Formatter().parse(value) if field is not None} for value in values.values()]
            self.assertEqual(placeholders[0],placeholders[1],key)
            self.assertEqual(placeholders[0],placeholders[2],key)


if __name__ == '__main__':
    unittest.main()
