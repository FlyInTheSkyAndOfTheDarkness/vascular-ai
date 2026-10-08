"""Real browser acceptance against the isolated local Compose stack only.

python tools/browser_smoke.py
Uses local .env credentials, synthetic data and http://cabinet.med-it.localhost:18080.
Never accepts a production URL. Requires Playwright and Chrome.
"""
import json
from pathlib import Path
import secrets
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/local_validation'

def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    env=dict(line.split('=',1) for line in (ROOT/'.env').read_text().splitlines() if line and not line.startswith('#') and '=' in line)
    account='qa_'+secrets.token_hex(4)
    password=secrets.token_hex(16)
    patient='QA-'+account
    checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='chrome',headless=True,args=['--no-proxy-server','--host-resolver-rules=MAP cabinet.med-it.localhost 127.0.0.1','--enable-unsafe-swiftshader'])
        page=browser.new_page(viewport={'width':1440,'height':1000});page.set_default_timeout(30000)
        expect.set_options(timeout=30000)
        try:
            page.goto('http://cabinet.med-it.localhost:18080/')
            viewer=page.frame_locator('iframe[srcdoc*="VascularAI 3D model"]')
            expect(viewer.locator('#viewer')).to_have_class('viewer ready')
            expect(viewer.locator('#status')).to_have_text('3D готово')
            page.get_by_label('Логин',exact=True).fill('admin')
            page.get_by_label('Пароль',exact=True).fill(env['ADMIN_PASSWORD'])
            page.get_by_role('button',name='Войти',exact=True).click()
            expect(page.get_by_role('heading',name='Админ-панель',exact=True)).to_be_visible()
            if not page.get_by_label('Логин *',exact=True).is_visible():
                page.get_by_text('Создать аккаунт',exact=True).first.click()
            page.get_by_label('Логин *',exact=True).fill(account)
            page.get_by_label('ФИО / отделение',exact=True).fill('Synthetic QA Doctor')
            page.get_by_label('Временный пароль *',exact=True).fill(password)
            page.get_by_label('Повторите пароль *',exact=True).fill(password)
            page.get_by_role('button',name='Создать аккаунт',exact=True).click()
            expect(page.get_by_test_id('stAlert').filter(has_text=account)).to_be_visible()
            page.get_by_test_id('stSidebar').get_by_role('button',name='Выйти',exact=True).click()
            page.get_by_label('Логин',exact=True).fill(account)
            page.get_by_label('Пароль',exact=True).fill(password)
            page.get_by_role('button',name='Войти',exact=True).click()
            sidebar=page.get_by_test_id('stSidebar')
            sidebar.get_by_role('button',name='Пациентки',exact=True).click()
            expect(page.get_by_role('heading',name='Пациентки',exact=True)).to_be_visible()
            # A newly created doctor has an empty list and an expanded creation form.
            expect(page.get_by_label('ID пациентки *',exact=True)).to_be_visible()
            page.get_by_label('ID пациентки *',exact=True).fill(patient)
            page.get_by_label('Инициалы / код',exact=True).fill('SYNTHETIC QA')
            page.get_by_role('button',name='Создать карту',exact=True).click()
            expect(page.get_by_test_id('stAlert').filter(has_text=patient)).to_be_visible()
            page.get_by_role('button',name='Новый расчёт',exact=True).last.click()
            page.get_by_role('button',name='Рассчитать риск',exact=True).click()
            expect(page.get_by_text('Исправьте отмеченные ошибки. Расчёт не сохранён.',exact=True)).to_be_visible()
            names=['Возраст пациентки','Систолическое давление','Диастолическое давление','Сахар крови','Температура тела','Пульс в покое']
            for values,result in [([15,78,49,7.5,98,77],'Низкий риск'),([29,130,70,6.7,98,78],'Средний риск'),([25,140,100,15,98.6,70],'Высокий риск')]:
                for name,value in zip(names,values):page.get_by_role('textbox',name=name+' *',exact=False).fill(str((value-32)*5/9 if name=='Температура тела' else value))
                page.get_by_role('button',name='Рассчитать риск',exact=True).click()
                expect(page.locator('.va-risk')).to_have_text(result)
                expect(page.get_by_test_id('stAlert').filter(has_text=patient)).to_be_visible()
            page.get_by_role('textbox',name='Температура тела *',exact=False).fill('98.6')
            page.get_by_role('button',name='Рассчитать риск',exact=True).click()
            expect(page.get_by_text('Исправьте отмеченные ошибки. Расчёт не сохранён.',exact=True)).to_be_visible()
            expect(page.get_by_text('Сохранённый расчёт',exact=True)).to_have_count(0)
            checks += ['actual_3d_model','account_and_patient_notifications','required_fields','three_live_profiles','invalid_temperature_rejected']
            for width in [1440,768,390]:
                page.set_viewport_size({'width':width,'height':1000 if width==1440 else 844})
                page.wait_for_timeout(600)
                close=page.get_by_test_id('stSidebarCollapseButton')
                box=sidebar.bounding_box()
                if width<1000 and close.is_visible() and box and box['x']>=0:close.click()
                page.get_by_test_id('stMain').evaluate('(e)=>e.scrollTop=0')
                page.screenshot(path=str(REPORT/f'cabinet-{width}.png'))
                assert not page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
            checks.append('responsive_widths')
            expect(page.get_by_test_id('stException')).to_have_count(0)
            (REPORT/'browser_smoke.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2),encoding='utf-8')
        except Exception as exc:
            page.screenshot(path=str(REPORT/'browser-failure.png'))
            message=str(exc).replace(password,'[redacted]').replace(env['ADMIN_PASSWORD'],'[redacted]')
            raise RuntimeError(message) from None
        finally:browser.close()
    print('Browser smoke passed:',', '.join(checks))

if __name__=='__main__':main()
