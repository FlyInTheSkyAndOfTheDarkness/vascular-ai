"""Browser smoke test against the isolated local Docker stack only.

Requires playwright and a locally installed Chrome. Never points at production.
"""
import json
from pathlib import Path
import secrets

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local_validation'


def main():
    expect.set_options(timeout=30000)
    env = dict(line.split('=', 1) for line in (ROOT / '.env').read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    account = 'qa_' + secrets.token_hex(4)
    password = secrets.token_hex(12)
    report = {'account': account, 'checks': []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='chrome', headless=True,
            args=['--host-resolver-rules=MAP med-it.localhost 127.0.0.1, MAP cabinet.med-it.localhost 127.0.0.1'])
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.set_default_timeout(30000)
        try:
            page.goto('http://med-it.localhost:18080/', wait_until='domcontentloaded')
            expect(page.get_by_role('heading', level=1)).to_contain_text('материнского риска')
            page.locator('button.text-button.login').click()
            page.wait_for_url('http://cabinet.med-it.localhost:18080/**')
            page.get_by_label('Логин', exact=True).fill('admin')
            page.get_by_label('Пароль', exact=True).fill(env['ADMIN_PASSWORD'])
            page.get_by_role('button', name='Войти', exact=True).click()
            expect(page.get_by_role('heading', name='Админ-панель', exact=True)).to_be_visible()
            report['checks'].append('landing_to_admin_login_via_proxy_and_websocket')

            page.get_by_label('Логин', exact=True).fill(account)
            page.get_by_label('ФИО / отделение', exact=True).fill('QA Synthetic Doctor')
            page.get_by_label('Временный пароль', exact=True).fill(password)
            page.get_by_label('Повторите пароль', exact=True).fill(password)
            page.get_by_role('button', name='Создать аккаунт', exact=True).click()
            # Wait for the rerun before leaving; a new option proves the account was stored.
            expect(page.get_by_text('Такой логин уже существует.', exact=True)).to_have_count(0)
            page.wait_for_timeout(2000)
            expect(page.get_by_test_id('stException')).to_have_count(0)
            page.get_by_role('button', name='Выход', exact=True).click()
            expect(page.get_by_role('button', name='Войти', exact=True)).to_be_visible()
            page.wait_for_timeout(700)
            page.get_by_label('Логин', exact=True).fill(account)
            page.get_by_label('Пароль', exact=True).fill(password)
            page.get_by_role('button', name='Войти', exact=True).click()
            sidebar = page.get_by_test_id('stSidebar')
            expect(sidebar.get_by_role('button', name='Новый расчёт', exact=True)).to_be_visible()
            report['checks'].append('create_doctor_and_login')

            sidebar.get_by_role('button', name='Пациенты', exact=True).click()
            page.get_by_text('Новая карта пациентки', exact=True).click()
            page.get_by_label('Внутренний ID', exact=True).fill('QA-' + account)
            page.get_by_label('Инициалы / код', exact=True).fill('SYNTHETIC QA')
            page.get_by_role('button', name='Создать карту', exact=True).click()
            expect(page.get_by_text('Карта QA-' + account + ' создана.', exact=True)).to_be_visible()
            sidebar.get_by_role('button', name='Новый расчёт', exact=True).click()
            patient_select = page.get_by_role('combobox', name='Пациентка', exact=True)
            patient_select.click()
            page.get_by_role('option').filter(has_text='QA-' + account).click()
            names = ['Возраст пациентки, лет', 'Систолическое давление, мм рт. ст.',
                     'Диастолическое давление, мм рт. ст.', 'Сахар крови, ммоль/л',
                     'Температура тела, F', 'Пульс в покое, уд/мин']
            results = []
            for values, expected in [([15, 78, 49, 7.5, 98, 77], 'Низкий риск'),
                                     ([29, 130, 70, 6.7, 98, 78], 'Средний риск'),
                                     ([25, 140, 100, 15, 98.6, 70], 'Высокий риск')]:
                for name, value in zip(names, values):
                    field = page.get_by_role('spinbutton', name=name, exact=True)
                    field.fill(str(value))
                    field.press('Tab')
                week = page.get_by_role('spinbutton', name='Срок беременности', exact=True)
                week.fill('25')
                week.press('Tab')
                page.get_by_role('button', name='Рассчитать риск', exact=True).click()
                expect(page.locator('.risk-title')).to_have_text(expected)
                results.append({'input': values, 'class': expected, 'probabilities': page.locator('.prob-value').all_text_contents()})
            report['results'] = results
            report['checks'].append('three_different_live_calculations')
            page.get_by_role('spinbutton', name=names[1], exact=True).fill('80')
            page.get_by_role('spinbutton', name=names[1], exact=True).press('Tab')
            page.get_by_role('button', name='Рассчитать риск', exact=True).click()
            expect(page.get_by_text('Систолическое давление должно быть выше диастолического. Проверьте введённые значения.', exact=True)).to_be_visible()
            report['checks'].append('invalid_pressure_rejected')
            sidebar.get_by_role('button', name='Пациенты', exact=True).click()
            expect(page.get_by_role('heading', name='Пациенты', exact=True)).to_be_visible()
            expect(page.get_by_test_id('stException')).to_have_count(0)
            search = page.get_by_placeholder('Поиск по ID...')
            search.fill('QA-' + account)
            search.press('Enter')
            expect(page.get_by_role('button', name='История', exact=True)).to_have_count(1)
            page.get_by_role('button', name='История', exact=True).first.click()
            expect(page.get_by_role('heading', name='Пациент QA-' + account, exact=True)).to_be_visible()
            expect(page.get_by_text('25 нед', exact=True).first).to_be_visible()
            page.locator('input[type=file]').set_input_files({
                'name': 'synthetic.txt', 'mimeType': 'text/plain', 'buffer': b'VascularAI synthetic deployment check',
            })
            page.get_by_label('Комментарий к файлу', exact=True).fill('Synthetic deployment check')
            page.get_by_role('button', name='Прикрепить файл', exact=True).click()
            expect(page.get_by_text('Файл прикреплён к карте.', exact=True)).to_be_visible()
            report['checks'].append('visit_week_and_attachment_upload')
            sidebar.get_by_role('button', name='Отчёты', exact=True).click()
            expect(page.get_by_role('heading', name='Отчёты', exact=True)).to_be_visible()
            expect(page.get_by_test_id('stException')).to_have_count(0)
            page.screenshot(path=str(REPORT / 'browser-report.png'), full_page=True)
            report['checks'].append('patients_and_reports')
            report['passed'] = True
        except Exception as error:
            page.screenshot(path=str(REPORT / 'browser-failure.png'), full_page=True)
            (REPORT / 'browser-failure.txt').write_text(page.locator('body').inner_text(), encoding='utf-8')
            report['passed'] = False
            message = str(error).replace(env['ADMIN_PASSWORD'], '[redacted]').replace(password, '[redacted]')
            raise RuntimeError(message) from None
        finally:
            (REPORT / 'browser_smoke.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            browser.close()
    print('Browser smoke passed:', ', '.join(report['checks']))


if __name__ == '__main__':
    main()
