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
        app.append_patient(row)

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
        self.assertEqual(app.patients_df().iloc[0]['age'], '29')
        self.assertEqual(app.next_patient_id(app.patients_df()), 'PT-0002')

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
            app.save_uploaded_attachment('PT-0001', attachment, 'test')
        saved = app.attachments_df().iloc[0]
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
        if patient:
            self.patient()
        user = self.doctor() if role == 'doctor' else app.authenticate('admin', app.DEFAULT_ADMIN_PASSWORD)
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

    def test_login_form_and_invalid_password(self):
        at = AppTest.from_string('import app\napp.main()', default_timeout=30).run()
        self.assertFalse(list(at.exception))
        at.text_input[0].set_value('admin')
        at.text_input[1].set_value('incorrect')
        self.click(at, 'Войти')
        self.assertTrue(list(at.error))
        at.text_input[1].set_value(app.DEFAULT_ADMIN_PASSWORD)
        self.click(at, 'Войти')
        self.assertEqual(at.session_state['auth_user']['role'], 'admin')

    def test_doctor_pages(self):
        at = self.start()
        for page in ['Дашборд', 'Новый расчёт', 'Пациенты', 'Отчёты', 'Настройки']:
            with self.subTest(page=page):
                self.navigate(at, page)

    def test_admin_pages(self):
        at = self.start(role='admin')
        for page in ['Админ-панель', 'Настройки']:
            self.navigate(at, page)

    def test_admin_analytics_with_doctor_but_no_patients_or_visits(self):
        self.doctor()
        at = self.start(role='admin', patient=False)
        self.assertFalse(list(at.exception))

    def test_empty_store_patient_creation(self):
        at = self.start(patient=False)
        self.navigate(at, 'Новый расчёт')
        self.click(at, 'Создать карту')
        self.assertEqual(len(app.patients_df()), 1)

    def test_calculation_persistence_and_patient_history(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        self.click(at, 'Рассчитать риск')
        self.assertTrue(list(at.success))
        saved = app.visits_df().iloc[0]
        expected = app.predict_risk({f: float(saved[f]) for f in FEATURES}, app.load_bundle())
        self.assertEqual(saved['risk_label'], expected['class_name'])
        np.testing.assert_allclose([float(saved[c]) for c in ['prob_low', 'prob_mid', 'prob_high']],
                                   expected['probabilities'], atol=5e-7)
        at.run()
        self.assertEqual(len(app.visits_df()), 1, 'Reruns must not duplicate saved visits')
        at.session_state[app.PATIENT_VIEW_KEY] = 'PT-0001'
        self.navigate(at, 'Пациенты')
        self.navigate(at, 'Отчёты')

    def test_unsaved_calculation(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        next(c for c in at.checkbox if c.label == 'Сохранить визит в карту пациентки').uncheck()
        self.click(at, 'Рассчитать риск')
        self.assertTrue(app.visits_df().empty)
        self.assertIn('class_name', at.session_state['intake_result'])

    def test_patient_switch_resets_age_and_previous_result(self):
        at = self.start()
        self.patient('PT-0002', '40')
        self.navigate(at, 'Новый расчёт')
        self.click(at, 'Рассчитать риск')
        select = at.selectbox(key='intake_patient')
        select.select(select.options[1]).run()
        self.assertFalse(list(at.exception))
        self.assertEqual(at.number_input(key='input_Age').value, 40)
        self.assertFalse(at.session_state.filtered_state.get('intake_result'))

    def test_changed_measurement_invalidates_previous_result(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        self.click(at, 'Рассчитать риск')
        at.number_input(key='input_SystolicBP').set_value(150).run()
        self.assertFalse(list(at.exception))
        self.assertFalse(at.session_state.filtered_state.get('intake_result'))
        self.assertEqual(len(app.visits_df()), 1)

    def test_intake_heading_matches_training_target(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        headings = '\n'.join(m.value for m in at.markdown if m.value.startswith('<div class="page-head">'))
        self.assertTrue(headings)
        self.assertNotIn('риска преэклампсии', headings)

    def test_repeated_calculations_use_current_inputs_and_change_result(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        # Recorded input profiles exercise all three classes in a single UI session.
        profiles = [
            [15, 78, 49, 7.5, 98.0, 77],
            [29, 130, 70, 6.7, 98.0, 78],
            [25, 140, 100, 15.0, 98.6, 70],
        ]
        observed = []
        with patch.object(app, 'predict_risk', wraps=app.predict_risk) as predict:
            for index, values in enumerate(profiles):
                for feature, value in zip(FEATURES, values):
                    at.number_input(key=f'input_{feature}').set_value(value)
                self.click(at, 'Рассчитать риск')
                self.assertEqual(predict.call_count, index + 1)
                self.assertEqual(predict.call_args.args[0], dict(zip(FEATURES, values)))
                result = at.session_state['intake_result']
                np.testing.assert_allclose(result['patient'].iloc[0], values)
                saved = app.visits_df().iloc[-1]
                np.testing.assert_allclose([float(saved[f]) for f in FEATURES], values)
                np.testing.assert_allclose(
                    [float(saved[c]) for c in ['prob_low', 'prob_mid', 'prob_high']],
                    result['probabilities'], atol=5e-7,
                )
                observed.append(result['class_name'])
                probabilities_html = '\n'.join(m.value for m in at.markdown if 'prob-value' in m.value)
                for probability in result['probabilities']:
                    self.assertIn(f'{probability * 100:.0f}%', probabilities_html)
        self.assertEqual(observed, ['low risk', 'mid risk', 'high risk'])
        self.assertEqual(len(app.visits_df()), 3)

    def test_biomarkers_and_history_are_saved_but_do_not_affect_model(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        self.click(at, 'Рассчитать риск')
        previous = at.session_state['intake_result']['probabilities'].copy()
        for label, value in [('PLGF', '15'), ('PAPP-A', '0.2'), ('SFLT-1', '5000'), ('MAP', '125'), ('ИМТ', '35')]:
            next(w for w in at.text_input if w.label == label).set_value(value)
        next(w for w in at.checkbox if w.label == 'Преэклампсия в анамнезе').check()
        next(w for w in at.number_input if w.label == 'Срок беременности').set_value(25)
        self.click(at, 'Рассчитать риск')
        np.testing.assert_array_equal(at.session_state['intake_result']['probabilities'], previous)
        saved = app.visits_df().iloc[-1]
        self.assertEqual(saved['plgf'], '15')
        self.assertEqual(saved['papp_a'], '0.2')
        self.assertEqual(saved['sflt1'], '5000')
        self.assertEqual(saved['map_value'], '125')
        self.assertEqual(saved['bmi'], '35')
        self.assertIn('преэклампсия', saved['anamnesis'])
        self.assertEqual(saved['gestational_week'], '25')

    def test_invalid_pressure_never_calculates_or_saves(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        self.click(at, 'Рассчитать риск')
        for systolic, diastolic in [(80, 100), (100, 100)]:
            at.number_input(key='input_SystolicBP').set_value(systolic)
            at.number_input(key='input_DiastolicBP').set_value(diastolic)
            with patch.object(app, 'append_visit') as save:
                self.click(at, 'Рассчитать риск')
                save.assert_not_called()
            self.assertTrue(list(at.error))
            self.assertFalse(list(at.success))
            self.assertFalse(at.session_state.filtered_state.get('intake_result'))
        self.assertEqual(len(app.visits_df()), 1)

    def test_visit_week_survives_reload_and_appears_in_history(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        next(w for w in at.number_input if w.label == 'Срок беременности').set_value(25)
        self.click(at, 'Рассчитать риск')
        self.assertEqual(app.visits_df().iloc[0]['gestational_week'], '25')
        at.session_state[app.PATIENT_VIEW_KEY] = 'PT-0001'
        self.navigate(at, 'Пациенты')
        self.assertIn('25 нед', '\n'.join(m.value for m in at.markdown))

    def test_write_error_is_not_reported_as_success(self):
        at = self.navigate(self.start(), 'Новый расчёт')
        with patch.object(app, 'append_visit', side_effect=app.StorageError('Storage unavailable')):
            self.click(at, 'Рассчитать риск')
        self.assertTrue(list(at.error))
        self.assertFalse(list(at.success))
        self.assertTrue(app.visits_df().empty)


if __name__ == '__main__':
    unittest.main()
