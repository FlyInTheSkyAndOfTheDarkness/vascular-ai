"""Responsive, localized workspace. Persistence and model stay in app.py."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import uuid
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from cabinet_i18n import LANGUAGES, CATALOGUE, tr

PAGES = {'Дашборд':'dashboard', 'Новый расчёт':'intake', 'Пациенты':'patients',
         'Отчёты':'reports', 'Настройки':'settings', 'Админ-панель':'admin_panel'}
FEATURES = ['Age','SystolicBP','DiastolicBP','BS','BodyTemp','HeartRate']
EVENTS = {'login':'login_event','logout':'logout','account_created':'account_created_event',
          'account_updated':'account_updated','profile_updated':'profile_updated',
          'password_changed':'password_changed','patient_created':'patient_created_event',
          'visit_calculated':'visit_calculated','attachment_uploaded':'attachment_uploaded'}
RISKS = ['low risk', 'mid risk', 'high risk']
COLORS = {'low risk':'#15766c','mid risk':'#926b08','high risk':'#b33e55'}

CSS = '''
<style>
:root {--va-ink:#302d38;--va-muted:#6a6571;--va-accent:#a44f65;--va-line:#e9e3e6;}
.stApp {background:#f7f6f8;color:var(--va-ink);}
html,body,.stApp,h1,h2,h3,p,label,button,input,textarea,select,.va-table {font-family:Arial,sans-serif;}
[data-testid="stMainBlockContainer"] {max-width:1440px;padding:4rem 2rem 4rem;}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stMainMenu"],[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid var(--va-line);}
[data-testid="stSidebarUserContent"] {padding-top:1.2rem;}
h1 {font-size:1.9rem !important;line-height:1.25 !important;letter-spacing:-.025em;}
h2 {font-size:1.35rem !important;line-height:1.35 !important;}
h3 {font-size:1.12rem !important;line-height:1.4 !important;}
h1,h2,h3,p,button,label {overflow-wrap:anywhere;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;background:white;}
[data-testid="stForm"] {background:#fff;border:1px solid var(--va-line);border-radius:16px;padding:1.25rem;}
button[kind="primary"],button[kind="primaryFormSubmit"] {background:var(--va-accent);border-color:var(--va-accent);color:white;}
[data-testid="stSidebar"] button {text-align:left;justify-content:flex-start;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {font-size:1rem;}
[data-testid="InputInstructions"] {display:none;}
.va-brand {font-weight:800;font-size:1.3rem;letter-spacing:-.025em;color:#283650;margin:0;}
.va-eyebrow {color:var(--va-muted);font-size:.82rem;margin:.25rem 0 1rem;}
.va-metrics {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin:.5rem 0 1.5rem;}
.va-metric {background:#fff;border:1px solid var(--va-line);border-radius:14px;padding:1.1rem;min-width:0;}
.va-metric strong {display:block;font-size:1.8rem;margin:.15rem 0 .35rem;}
.va-metric span {font-size:.9rem;color:var(--va-muted);overflow-wrap:anywhere;}
.va-table-wrap {overflow-x:auto;border:1px solid var(--va-line);border-radius:12px;background:#fff;margin:.6rem 0 1rem;}
.va-table {border-collapse:collapse;width:100%;font-size:.9rem;text-align:left;}
.va-table th {background:#f3f0f3;color:#625767;font-weight:600;white-space:nowrap;}
.va-table th,.va-table td {padding:.8rem 1rem;border-bottom:1px solid #efebee;vertical-align:top;}
.va-table td {min-width:95px;max-width:380px;overflow-wrap:anywhere;}
.va-table tr:last-child td {border-bottom:0;}
.va-table caption {text-align:left;padding:.7rem 1rem;font-weight:600;}
.va-risk {display:inline-block;padding:.35rem .65rem;border-radius:8px;font-weight:650;border:1px solid currentColor;line-height:1.45;}
.va-required {padding:.8rem 1rem;border-left:4px solid #a44f65;background:#faf1f4;border-radius:6px;margin-bottom:1rem;}
.va-result {padding:1rem 0;}
.va-result h2 {margin:.5rem 0;}
.va-prob {display:grid;grid-template-columns:minmax(95px,1fr) 2fr 55px;gap:.8rem;align-items:center;margin:1rem 0;}
.va-track {height:9px;border-radius:8px;background:#ece8ec;overflow:hidden;}
.va-track i {display:block;height:100%;border-radius:8px;}
.va-safe {color:var(--va-muted);font-size:.86rem;line-height:1.6;}
[data-testid="stFileUploader"] button[kind="secondary"], [data-testid="stFileUploader"] button[kind="secondary"] * {font-size:0 !important;}
[data-testid="stFileUploader"] button[kind="secondary"]::after {content:var(--upload-label);font-size:.9rem;}
[data-testid="stFileUploaderDropzoneInstructions"] {display:none;}
@media(max-width:1100px) {
 [data-testid="stMainBlockContainer"] {padding:4rem 1rem 3rem;}
 [data-testid="stMain"] [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 [data-testid="stMain"] [data-testid="stHorizontalBlock"]>[data-testid="stColumn"] {width:100% !important;flex:1 1 100% !important;min-width:0 !important;}
}
@media(max-width:600px) {
 .va-metrics {grid-template-columns:1fr;gap:.65rem;}
 .va-metric {padding:.9rem;}
 .va-metric strong {font-size:1.5rem;}
 h1 {font-size:1.5rem !important;}
 [data-testid="stMainBlockContainer"] {padding:4rem .85rem 2rem;}
 .va-prob {grid-template-columns:1fr 1fr 45px;gap:.4rem;font-size:.85rem;}
}
@media(prefers-reduced-motion:reduce) {* {animation:none !important;transition:none !important;scroll-behavior:auto !important;}}
</style>
'''

def unit(feature):
    return {'Age':tr('years'),'SystolicBP':tr('pressure_unit'),'DiastolicBP':tr('pressure_unit'),
            'BS':tr('glucose_unit'),'BodyTemp':'°C','HeartRate':tr('pulse_unit')}[feature]

def temperature_celsius(fahrenheit):
    return (float(fahrenheit)-32)*5/9


def measurement_bounds(feature, meta):
    if feature == 'BodyTemp':
        # Whole tenths inside the model's supported Fahrenheit range.
        return (math.ceil(temperature_celsius(meta['min'])*10)/10,
                math.floor(temperature_celsius(meta['max'])*10)/10)
    return meta['min'], meta['max']


def display_measurement(feature, value):
    if feature == 'BodyTemp':
        try:
            value=temperature_celsius(value)
            if not math.isfinite(value):return '—'
            return f'{value:.2f}'.rstrip('0').rstrip('.')+' °C'
        except (ValueError, TypeError):return '—'
    return str(value)+' '+unit(feature)


def formatter(mapping=None):
    # Freeze labels for this widget render, including deferred serialization.
    lang=st.session_state.get('ui_language','ru')
    labels={key:values[lang] for key,values in CATALOGUE.items()}
    mapping=mapping or {}
    return lambda value:labels.get(mapping.get(value,value),str(value))

def timezone_label():
    offset=datetime.now().astimezone().strftime('%z')
    return 'UTC'+offset[:3]+':'+offset[3:]

def numeric(raw, minimum, maximum, *, required=True):
    """Validate the exact submitted string, including empty/NaN/infinity/comma."""
    raw = str(raw).strip()
    if not raw and not required:
        return None
    if not raw:
        raise ValueError('missing')
    try:
        value = float(raw.replace(',', '.'))
    except (ValueError, TypeError):
        raise ValueError('invalid_number') from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError('invalid_number')
    return value

def validate_measurements(raw, meta):
    values, errors = {}, []
    for feature in FEATURES:
        entry = meta[feature]
        minimum,maximum=measurement_bounds(feature,entry)
        try:
            value=numeric(raw.get(feature,''),minimum,maximum)
            # The model and persisted BodyTemp remain in Fahrenheit. Convert once.
            values[feature] = value*9/5+32 if feature=='BodyTemp' else value
        except ValueError as exc:
            errors.append((str(exc),feature,minimum,maximum))
    if not errors and values['SystolicBP'] <= values['DiastolicBP']:
        errors.append(('pressure_error','',0,0))
    return values, errors

def table(headers, rows, caption=''):
    if not rows:
        st.info(tr('empty'))
        return
    html = '<div class="va-table-wrap" tabindex="0"><table class="va-table">'
    if caption:
        html += '<caption>'+escape(caption)+'</caption>'
    html += '<thead><tr>'+''.join('<th scope="col">'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'
    html += ''.join('<tr>'+''.join('<td>'+escape(str(v) if str(v) else '—')+'</td>' for v in row)+'</tr>' for row in rows)
    st.html(html+'</tbody></table></div>')

def metrics(items):
    st.html('<div class="va-metrics">'+''.join('<div class="va-metric"><span>'+escape(tr(label))+'</span><strong>'+escape(str(value))+'</strong></div>' for label,value in items)+'</div>')

def risk(label):
    return tr(label) if label in RISKS else tr('no_visit')

def risk_badge(label):
    st.html('<span class="va-risk" style="color:'+COLORS.get(label,'#6a6571')+'">'+escape(risk(label))+'</span>')

def go(a, page, patient=None, new=False):
    st.session_state[a.NAV_KEY] = page
    st.session_state.pop('_notice',None)
    st.session_state.pop('_visit_id',None)
    if patient is not None:
        st.session_state[a.PATIENT_VIEW_KEY] = patient
    elif page == 'Пациенты':
        st.session_state.pop(a.PATIENT_VIEW_KEY,None)
    if new:
        st.session_state.pop('intake_result',None)
        st.session_state.pop('_saved_digest',None)
        for key in list(st.session_state):
            if key.startswith('input_') or key.startswith('extra_'):
                del st.session_state[key]
        if patient:
            st.session_state['_intake_patient'] = patient
    st.rerun()

def notify(key, **values):
    st.session_state['_notice'] = {'key':key,'values':values,'fresh':True}
    st.rerun()

def clear_notice():
    st.session_state.pop('_notice',None)

def submit(label, **kwargs):
    return st.form_submit_button(label,on_click=clear_notice,**kwargs)

def show_notice():
    notice = st.session_state.get('_notice')
    if notice:
        message = tr(notice['key'],**notice['values'])
        if notice['fresh']:
            st.toast(message,icon='✅',duration=2)
            notice['fresh'] = False
        st.success(message)
        if st.button(tr('acknowledge'),key='_dismiss',type='tertiary'):
            st.session_state.pop('_notice',None)
            st.rerun()

def language_changed():
    st.query_params['lang'] = st.session_state['ui_language']

def render(a):
    try:
        a.check_deployment_settings();a.ensure_store();a.ensure_auth_store()
        identity_user=a.current_user()
        if identity_user:identity_user=a.refresh_current_user(identity_user)
        a.reset_clinical_session(identity_user)
    except PermissionError:
        st.error(tr('access_denied'))
    except a.StorageError:
        st.error(tr('storage_error'));return
    if 'ui_language' not in st.session_state:
        lang = st.query_params.get('lang','ru')
        st.session_state['ui_language'] = lang if lang in LANGUAGES else 'ru'
    st.html(CSS)
    st.html('<style>:root{--upload-label:'+json.dumps(tr('upload'),ensure_ascii=False)+';}</style>')
    top, language = st.columns([3,1],vertical_alignment='center')
    with top:
        st.html('<p class="va-brand">VascularAI</p><p class="va-eyebrow">'+escape(tr('clinical_workspace'))+'</p>')
    with language:
        st.selectbox('Язык / Language / Тіл',list(LANGUAGES),format_func=LANGUAGES.get,
                     key='ui_language',on_change=language_changed)
    try:
        a.check_deployment_settings();a.ensure_store();a.ensure_auth_store()
        user = a.current_user()
        if user:
            user = a.refresh_current_user(user)
        if not user:
            login(a)
            return
        sidebar(a,user)
        show_notice()
        page = st.session_state.get(a.NAV_KEY,'Админ-панель' if a.is_admin(user) else 'Дашборд')
        if a.is_admin(user):
            if page == 'Настройки':settings(a,user)
            else:admin(a,user)
        elif page == 'Новый расчёт':intake(a,user)
        elif page == 'Пациенты':patients(a,user)
        elif page == 'Отчёты':reports(a)
        elif page == 'Настройки':settings(a,user)
        else:overview(a,user)
    except a.StorageError:
        st.error(tr('storage_error'))

def login(a):
    form_col, model_col = st.columns([1,1.1],gap='large')
    with form_col:
        st.title(tr('welcome'))
        st.write(tr('login_hint'))
        with st.form('login_form'):
            username = st.text_input(tr('login'),key='login_username')
            password = st.text_input(tr('password'),type='password',key='login_password')
            submitted = submit(tr('signin'),type='primary',width='stretch')
        st.caption(tr('session_hint'))
        if submitted:
            user = a.authenticate(username,password)
            if not user:
                st.error(tr('invalid_login'))
            else:
                a.set_browser_session(a.issue_session_token(user['user_id']))
                st.session_state['auth_user'] = user
                a.log_event('login',target=user['username'],actor=user)
                st.rerun()
    with model_col:
        html = a.MODEL_VIEWER_PATH.read_text(encoding='utf-8').replace('<head>','<head><base href="/app/static/landing/">',1)
        for old,key in [('3D не загрузилось — показано изображение модели','model_error'),('Maternal risk model','model_title'),
                        ('Повторить загрузку','model_retry'),('Подготовка 3D','model_preparing'),('Загрузка 3D','model_loading'),
                        ('3D готово','model_ready'),('3D недоступно','model_unavailable')]:
            html = html.replace(old,tr(key))
        # A stationary model for users who request less motion.
        html = html.replace('const t = time * 0.001;', 'const t = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : time * 0.001;')
        st.iframe(html,height=420,width='stretch',tab_index=-1)
        st.caption(tr('model_limits'))

def sidebar(a,user):
    admin_user = a.is_admin(user)
    pages = ['Админ-панель','Настройки'] if admin_user else list(PAGES)[:5]
    if st.session_state.get(a.NAV_KEY) not in pages:
        st.session_state[a.NAV_KEY] = pages[0]
    a.apply_page_from_query(pages)
    with st.sidebar:
        st.markdown('### VascularAI')
        st.caption(tr('admin' if admin_user else 'doctor'))
        st.write(user.get('full_name') or user['username'])
        st.divider()
        for page in pages:
            if st.button(tr(PAGES[page]),key='nav_'+PAGES[page],width='stretch',
                         type='primary' if st.session_state[a.NAV_KEY]==page else 'secondary'):
                go(a,page,new=page=='Новый расчёт')
        st.divider()
        with st.popover(tr('notifications'),width='stretch'):
            events = a.activity_df()
            events = events.loc[events['actor']==user['username']].sort_values('event_at',ascending=False).head(8)
            if events.empty:st.caption(tr('notification_empty'))
            for _,event in events.iterrows():
                st.write(tr(EVENTS.get(event['action'],'event_other')))
                st.caption(str(event['event_at'])+' '+timezone_label()+' · '+str(event['target']))
        if st.button(tr('signout'),width='stretch'):
            a.log_event('logout',target=user['username'],actor=user)
            a.clear_browser_session(user)
            lang = st.session_state.get('ui_language','ru')
            st.session_state.clear()
            st.session_state['ui_language'] = lang
            st.rerun()

def visit_rows(a,visits,key,limit=10):
    if visits.empty:
        st.info(tr('no_visit'));return
    ordered=visits.sort_values('recorded_at',ascending=False)
    if key in ('history','report') and len(ordered)>limit:
        page=st.selectbox(tr('page'),list(range(1,(len(ordered)+limit-1)//limit+1)),key=key+'_page')
        recent=ordered.iloc[(page-1)*limit:page*limit]
    else:
        recent=ordered.head(limit)
    for _,row in recent.iterrows():
        with st.container(border=True):
            info,action = st.columns([3,1],vertical_alignment='center')
            with info:
                st.markdown('**'+escape(str(row['patient_id']))+'** · '+str(row['recorded_at'])+' '+timezone_label())
                risk_badge(str(row['risk_label']))
            with action:
                if st.button(tr('open_visit'),key=key+'_'+str(row['visit_id']),width='stretch'):
                    st.session_state['_visit_id'] = str(row['visit_id'])
                    st.session_state[a.PATIENT_VIEW_KEY] = str(row['patient_id'])
                    st.session_state[a.NAV_KEY] = 'Пациенты'
                    st.session_state.pop('_notice',None)
                    st.rerun()

def overview(a,user):
    st.title(tr('dashboard'))
    st.caption(user.get('full_name') or user['username'])
    visits = a.visits_df()
    metrics([('total_patients',len(a.patients_df())),('total_visits',len(visits)),('high_count',int((visits['risk_label']=='high risk').sum()))])
    if st.button(tr('intake'),key='overview_intake',type='primary'):go(a,'Новый расчёт',new=True)
    st.subheader(tr('recent'))
    visit_rows(a,visits,'recent')
    st.caption(tr('model_limits'))

def create_patient(a,user):
    st.subheader(tr('new_patient'))
    st.caption(tr('required_hint'))
    with st.form('create_patient_form',clear_on_submit=False):
        c1,c2 = st.columns(2)
        with c1:
            patient_id = st.text_input(tr('patient_id')+' *',value=a.next_patient_id(a.patients_df()),key='new_patient_id')
            initials = st.text_input(tr('initials'),key='new_patient_initials')
            age = st.text_input(tr('Age')+' *',value='29',key='new_patient_age')
            week = st.text_input(tr('week'),key='new_patient_week')
        with c2:
            doctor = st.text_input(tr('responsible'),value=user.get('full_name') or user['username'],key='new_patient_doctor')
            status = st.selectbox(tr('status'),['Активна','Наблюдение','Архив'],
                                 format_func=formatter({'Активна':'patient_active','Наблюдение':'patient_followup','Архив':'patient_archive'}))
            gravida = st.text_input(tr('gravida'),key='new_gravida')
            parity = st.text_input(tr('parity'),key='new_parity')
        bmi = st.text_input(tr('bmi'),key='new_bmi')
        anamnesis = st.text_area(tr('anamnesis'),key='new_anamnesis')
        note = st.text_area(tr('note'),key='new_note')
        submitted = submit(tr('create_patient'),type='primary')
    if submitted:
        errors=[]
        if not patient_id.strip():errors.append(tr('missing',field=tr('patient_id')))
        if patient_id.strip() in a.patients_df()['patient_id'].tolist():errors.append(tr('patient_duplicate'))
        for key,value,lo,hi,required in [('Age',age,10,70,True),('week',week,4,45,False),('gravida',gravida,0,20,False),('parity',parity,0,20,False),('bmi',bmi,5,100,False)]:
            try:numeric(value,lo,hi,required=required)
            except ValueError as exc:errors.append(tr(str(exc),field=tr(key),minimum=lo,maximum=hi))
        if errors:
            for message in errors:st.error(message)
            return
        a.append_patient({'patient_id':patient_id.strip(),'initials':initials.strip(),'age':age.strip(),
                          'gestational_week':week.strip(),'gravida':gravida.strip(),'parity':parity.strip(),
                          'doctor':doctor.strip(),'status':status,'created_at':a.now_iso(),'note':note.strip(),
                          'bmi':bmi.strip(),'anamnesis':anamnesis.strip()})
        a.log_event('patient_created',target=patient_id.strip())
        st.session_state[a.PATIENT_VIEW_KEY] = patient_id.strip()
        st.session_state[a.NAV_KEY] = 'Пациенты'
        notify('patient_created',id=patient_id.strip())

def intake(a,user):
    st.title(tr('intake_title'))
    st.caption(tr('intake_hint'))
    patients_frame = a.patients_df()
    if patients_frame.empty:
        st.info(tr('patient_hint'));create_patient(a,user);return
    options = patients_frame['patient_id'].astype(str).tolist()
    names = {str(row['patient_id']):str(row['patient_id'])+' · '+str(row['initials']) for _,row in patients_frame.iterrows()}
    wanted = st.session_state.pop('_intake_patient',None)
    if wanted in options:st.session_state['intake_patient']=wanted
    selected = st.selectbox(tr('patient')+' *',options,format_func=names.get,key='intake_patient',placeholder=tr('choose'))
    patient_row = patients_frame.loc[patients_frame['patient_id']==selected].iloc[0]
    if st.session_state.get('intake_active_patient') != selected:
        for key in list(st.session_state):
            if key.startswith('input_') or key.startswith('extra_'):del st.session_state[key]
        st.session_state.pop('intake_result',None)
        st.session_state.pop('_saved_digest',None)
        st.session_state['intake_active_patient'] = selected
    st.html('<div class="va-required"><strong>'+escape(tr('required'))+'</strong><br>'+escape(tr('required_hint'))+'</div>')
    raw={};extra={}
    with st.form('risk_form',enter_to_submit=False):
        columns=st.columns(2)
        for index,feature in enumerate(FEATURES):
            meta=a.FEATURE_META[feature]
            minimum,maximum=measurement_bounds(feature,meta)
            with columns[index%2]:
                raw[feature]=st.text_input(tr(feature)+' * ('+unit(feature)+')',
                    value=str(patient_row['age']) if feature=='Age' else '',key='input_'+feature+('_celsius' if feature=='BodyTemp' else ''),
                    help=tr('range_hint',minimum=minimum,maximum=maximum,unit=unit(feature)))
                if feature=='BodyTemp':st.caption(tr('temp_hint'))
        with st.expander(tr('optional')):
            st.caption(tr('optional_hint'))
            c1,c2=st.columns(2)
            for index,key in enumerate(['bmi','week','plgf','sflt1','papp_a','map_value']):
                with (c1 if index%2==0 else c2):
                    default=str(patient_row.get('bmi','')) if key=='bmi' else str(patient_row.get('gestational_week','')) if key=='week' else ''
                    extra[key]=st.text_input(tr(key),value=default,key='extra_'+key)
            preeclampsia=st.checkbox(tr('preeclampsia'),key='extra_preeclampsia')
            chronic=st.checkbox(tr('chronic'),key='extra_chronic')
            note=st.text_area(tr('visit_note'),key='extra_note')
        attach=st.checkbox(tr('attach_visit'),value=True,key='attach_visit')
        submitted=submit(tr('calculate'),type='primary',width='stretch')
    if st.button(tr('cancel'),key='intake_cancel'):go(a,'Дашборд')
    if submitted:
        st.session_state.pop('intake_result',None)
        values,errors=validate_measurements(raw,a.FEATURE_META)
        for key,lo,hi in [('bmi',5,100),('week',4,45),('plgf',0,1000000),('sflt1',0,1000000),('papp_a',0,1000000),('map_value',0,300)]:
            try:numeric(extra[key],lo,hi,required=False)
            except ValueError as exc:errors.append((str(exc),key,lo,hi))
        if errors:
            st.error(tr('form_errors'))
            for code,field,lo,hi in errors:
                st.error(tr(code,field=tr(field) if field else '',minimum=lo,maximum=hi))
            return
        try:
            with st.spinner(tr('calculating')):result=a.predict_risk(values,a.load_bundle())
        except FileNotFoundError:
            st.error(tr('model_missing'));return
        digest=hashlib.sha256(json.dumps([selected,values,extra,note,preeclampsia,chronic],sort_keys=True).encode()).hexdigest()
        if attach and digest!=st.session_state.get('_saved_digest'):
            probs=np.asarray(result['probabilities'],dtype=float)
            a.append_visit({'visit_id':uuid.uuid4().hex[:10],'patient_id':selected,'recorded_at':a.now_iso(),**values,
                'risk_label':result['class_name'],'risk_probability':f"{probs[int(result['class_index'])]:.6f}",
                'prob_low':f'{probs[0]:.6f}','prob_mid':f'{probs[1]:.6f}','prob_high':f'{probs[2]:.6f}',
                'top_factor':result['top_factor'],'visit_note':note.strip(),
                'bmi':extra['bmi'].strip(),'plgf':extra['plgf'].strip(),'papp_a':extra['papp_a'].strip(),
                'sflt1':extra['sflt1'].strip(),'map_value':extra['map_value'].strip(),
                'anamnesis':', '.join(x for x,flag in [('преэклампсия в анамнезе',preeclampsia),('хронические заболевания',chronic)] if flag),
                'gestational_week':extra['week'].strip()})
            a.log_event('visit_calculated',target=selected,details=str(result['class_name']))
            st.session_state['_saved_digest']=digest
        st.session_state['intake_result']=result
        st.session_state['_result_saved']=attach
        notify('calculation_saved' if attach else 'calculation_ready',id=selected)
    if st.session_state.get('intake_result'):
        st.divider()
        st.subheader(tr('result_saved' if st.session_state.get('_result_saved') else 'result_unsaved'))
        render_result(st.session_state['intake_result'])
    st.caption(tr('disclaimer'))

def render_result(result):
    st.caption(tr('result_hint'))
    risk_badge(str(result['class_name']))
    st.subheader(tr('probabilities'))
    for key,prob in zip(RISKS,result['probabilities']):
        st.html('<div class="va-prob"><span>'+escape(tr(key))+'</span><span class="va-track"><i style="background:'+COLORS[key]+';width:'+str(float(prob)*100)+'%"></i></span><strong>'+f'{float(prob):.1%}'+'</strong></div>')
    frame=result['patient']
    shap=np.asarray(result['shap_values'])
    table([tr('measurement'),tr('value'),tr('contribution')],
          [[tr(feature),display_measurement(feature,frame.iloc[0][feature]),f'{float(shap[index]):+.3f}'] for index,feature in enumerate(FEATURES)],tr('saved_values'))

def patients(a,user):
    st.title(tr('patients'))
    selected=st.session_state.get(a.PATIENT_VIEW_KEY)
    frame=a.patients_df()
    if selected and selected in frame['patient_id'].tolist():
        patient_card(a,user,frame.loc[frame['patient_id']==selected].iloc[0]);return
    with st.expander(tr('new_patient'),expanded=frame.empty):create_patient(a,user)
    if frame.empty:st.info(tr('patient_hint'));return
    query=st.text_input(tr('search'),key='patients_search')
    selected_risk=st.selectbox(tr('risk'),['all']+RISKS,format_func=formatter(),key='patients_risk')
    visits=a.visits_df().sort_values('recorded_at',ascending=False)
    latest=visits.drop_duplicates('patient_id').set_index('patient_id')['risk_label'].to_dict()
    if query.strip():
        needle=query.strip().casefold()
        frame=frame.loc[frame['patient_id'].astype(str).str.casefold().str.contains(needle,regex=False)|frame['initials'].astype(str).str.casefold().str.contains(needle,regex=False)]
    if selected_risk!='all':frame=frame.loc[frame['patient_id'].map(latest)==selected_risk]
    if frame.empty:st.info(tr('no_matches'));return
    page_count=max(1,(len(frame)+7)//8)
    page=st.selectbox(tr('page'),list(range(1,page_count+1)),key='patients_pagination')
    chunk=frame.iloc[(page-1)*8:page*8]
    st.caption(tr('count_shown',shown=len(chunk),total=len(frame)))
    for _,row in chunk.iterrows():
        pid=str(row['patient_id'])
        with st.container(border=True):
            st.subheader(pid+' · '+str(row['initials']))
            st.caption(tr('Age')+': '+str(row['age'])+' · '+tr('responsible')+': '+str(row['doctor']))
            risk_badge(latest.get(pid,''))
            c1,c2=st.columns(2)
            with c1:
                if st.button(tr('open_patient'),key='patient_'+pid,width='stretch'):go(a,'Пациенты',patient=pid)
            with c2:
                if st.button(tr('intake'),key='intake_'+pid,width='stretch'):go(a,'Новый расчёт',patient=pid,new=True)

def attachment_bytes(a,row):
    allowed=a.attachments_df()
    matches=allowed.loc[allowed['attachment_id']==str(row['attachment_id'])]
    if len(matches)!=1:raise PermissionError('Attachment is unavailable')
    row=matches.iloc[0]
    a.require_patient_access(str(row['patient_id']))
    path=str(row['stored_path'])
    if path.startswith('cloud:'):
        import requests
        relative=path.removeprefix('cloud:')
        if '..' in Path(relative).parts:raise ValueError('unsafe attachment path')
        base=a.secret_value('SUPABASE_URL').rstrip('/')
        bucket=a.secret_value('SUPABASE_BUCKET',a.ATTACHMENT_BUCKET)
        response=requests.get(f'{base}/storage/v1/object/authenticated/{quote(bucket,safe="")}/{quote(relative,safe="/")}',
                              headers={'Authorization':'Bearer '+a.secret_value('SUPABASE_SERVICE_KEY')},timeout=30)
        response.raise_for_status();return response.content
    resolved=(a.PROJECT_ROOT/path).resolve()
    if not resolved.is_relative_to(a.ATTACHMENTS_DIR.resolve()):raise ValueError('unsafe attachment path')
    return resolved.read_bytes()

def patient_card(a,user,row):
    pid=str(row['patient_id'])
    a.require_patient_access(pid)
    if st.button('← '+tr('patients'),key='back_patients'):go(a,'Пациенты')
    st.subheader(pid+' · '+str(row['initials']))
    table([tr('measurement'),tr('value')],[[tr(label),row.get(column,'') or '—'] for label,column in
          [('Age','age'),('week','gestational_week'),('bmi','bmi'),('responsible','doctor'),('anamnesis','anamnesis'),('note','note')]])
    if st.button(tr('intake'),key='card_intake',type='primary'):go(a,'Новый расчёт',patient=pid,new=True)
    visits=a.visits_df();visits=visits.loc[visits['patient_id']==pid].sort_values('recorded_at',ascending=False)
    visit_id=st.session_state.get('_visit_id')
    if visit_id and visit_id in visits['visit_id'].tolist():
        selected=visits.loc[visits['visit_id']==visit_id].iloc[0]
        with st.container(border=True):
            st.subheader(tr('visit')+' · '+str(selected['recorded_at'])+' '+timezone_label())
            risk_badge(str(selected['risk_label']))
            table([tr('measurement'),tr('value')],[[tr(f),display_measurement(f,selected[f])] for f in FEATURES],tr('view_details'))
            table([tr('risk'),tr('confidence')],[[tr(key),f'{float(selected[col]):.1%}'] for key,col in zip(RISKS,['prob_low','prob_mid','prob_high'])])
            for key,column in [('week','gestational_week'),('bmi','bmi'),('visit_note','visit_note'),('anamnesis','anamnesis')]:
                if str(selected.get(column,'')):st.write(tr(key)+': '+str(selected[column]))
    if not visits.empty:
        st.subheader(tr('trend'))
        data=pd.DataFrame({'date':pd.to_datetime(visits['recorded_at'],errors='coerce'),
                           'probability':pd.to_numeric(visits['prob_high'],errors='coerce')}).dropna().sort_values('date')
        chart=alt.Chart(data).mark_line(point=True,color='#a44f65').encode(
            x=alt.X('date:T',title=(tr('date')+' ('+timezone_label()+')'),axis=alt.Axis(format='%d.%m %H:%M',labelAngle=-20)),
            y=alt.Y('probability:Q',title=tr('high risk'),scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format='%')),
            tooltip=[alt.Tooltip('date:T',title=(tr('date')+' ('+timezone_label()+')'),format='%d.%m.%Y %H:%M'),alt.Tooltip('probability:Q',title=tr('high risk'),format='.1%')]
        ).properties(height=210,usermeta={'embedOptions':{'actions':False}})
        st.altair_chart(chart,use_container_width=True)
    st.subheader(tr('visits'));visit_rows(a,visits,'history',limit=20)
    st.subheader(tr('attachments'))
    with st.form('attachment_form_'+pid,clear_on_submit=True):
        uploaded=st.file_uploader(tr('upload'),type=['csv','xlsx','xls','pdf','png','jpg','jpeg','txt'],max_upload_size=20,key='upload_'+pid)
        st.caption(tr('upload_hint'))
        note=st.text_input(tr('note'),key='file_note_'+pid)
        submitted=submit(tr('attach'),type='primary')
    if submitted:
        if uploaded is None:st.error(tr('select_file'))
        elif uploaded.size>20*1024*1024:st.error(tr('file_limit'))
        else:
            a.save_uploaded_attachment(pid,uploaded,note)
            notify('attachment_saved')
    attachments=a.attachments_df();attachments=attachments.loc[attachments['patient_id']==pid]
    if attachments.empty:st.caption(tr('empty'))
    for _,attachment in attachments.iterrows():
        with st.container(border=True):
            st.write(str(attachment['filename']))
            st.caption(str(attachment['uploaded_at'])+' '+timezone_label()+' · '+str(attachment['note']))
            try:
                data=attachment_bytes(a,attachment)
                st.download_button(tr('download'),data,file_name=str(attachment['filename']),key='file_'+str(attachment['attachment_id']),on_click='ignore')
            except Exception:st.error(tr('file_unavailable'))

def filtered_period(frame,column,key):
    period=st.selectbox(tr('period'),['all_time','month','week_period'],format_func=formatter(),key=key)
    if period!='all_time':
        dates=pd.to_datetime(frame[column],errors='coerce')
        cutoff=pd.Timestamp(datetime.now()-timedelta(days=30 if period=='month' else 7))
        frame=frame.loc[dates>=cutoff]
    return frame

def export_frame(visits):
    columns=['patient_id','recorded_at',*FEATURES,'risk_label','prob_low','prob_mid','prob_high','gestational_week','visit_note']
    frame=visits.reindex(columns=columns).copy()
    frame['BodyTemp']=((pd.to_numeric(frame['BodyTemp'],errors='coerce')-32)*5/9).round(2)
    frame['risk_label']=frame['risk_label'].map(risk)
    return frame.rename(columns={'patient_id':tr('patient_id'),'recorded_at':(tr('date')+' ('+timezone_label()+')'),**{f:tr(f)+' ('+unit(f)+')' for f in FEATURES},
        'risk_label':tr('risk'),'prob_low':tr('low risk'),'prob_mid':tr('mid risk'),'prob_high':tr('high risk'),
        'gestational_week':tr('week'),'visit_note':tr('visit_note')})

def reports(a):
    st.title(tr('reports'))
    frame=filtered_period(a.visits_df(),'recorded_at','report_period')
    metrics([('total_visits',len(frame)),('total_patients',frame['patient_id'].nunique()),('high_count',int((frame['risk_label']=='high risk').sum()))])
    table([tr('risk'),tr('count')],[[tr(r),int((frame['risk_label']==r).sum())] for r in RISKS],tr('distribution'))
    if not frame.empty:
        export=export_frame(frame)
        excel=io.BytesIO()
        # Export strings as literals to avoid spreadsheet formula execution.
        with pd.ExcelWriter(excel,engine='openpyxl') as writer:
            export.to_excel(writer,index=False,sheet_name='VascularAI')
            for row in writer.book.active:
                for cell in row:
                    if cell.data_type=='f':cell.data_type='s'
        safe_csv=export.map(lambda x:"'"+x if isinstance(x,str) and x[:1] in ('=','+','-','@') else x)
        print_html='<!doctype html><html lang="'+st.session_state['ui_language']+'"><meta charset="utf-8"><title>VascularAI</title><style>body{font:14px Arial;padding:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}h1{font-size:24px}</style><h1>'+escape(tr('reports'))+'</h1>'+export.to_html(index=False,escape=True)+'</html>'
        c1,c2,c3=st.columns(3)
        with c1:st.download_button(tr('export_csv'),safe_csv.to_csv(index=False).encode('utf-8-sig'),'vascularai.csv','text/csv',on_click=lambda:st.toast(tr('export_ready')))
        with c2:st.download_button(tr('export_xlsx'),excel.getvalue(),'vascularai.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',on_click=lambda:st.toast(tr('export_ready')))
        with c3:st.download_button(tr('export_print'),print_html,'vascularai.html','text/html',on_click=lambda:st.toast(tr('export_ready')))
        st.caption(tr('print_hint'))
    st.subheader(tr('recent'));visit_rows(a,frame,'report',limit=20)

def settings(a,user):
    st.title(tr('settings'))
    users=a.users_df();mask=users['username']==user['username'];index=users.index[mask][0];row=users.loc[index]
    st.subheader(tr('profile'))
    with st.form('profile_settings_form'):
        name=st.text_input(tr('full_name'),value=str(row['full_name']),help=tr('changed_profile_hint'))
        c1,c2=st.columns(2);values={}
        for n,key in enumerate(['email','phone','clinic','specialty']):
            with (c1 if n%2==0 else c2):values[key]=st.text_input(tr(key),value=str(row.get(key,'') or ''))
        saved=submit(tr('save'),type='primary')
    if saved:
        users.loc[index,'full_name']=name.strip() or row['username']
        for key,value in values.items():users.loc[index,key]=value.strip()
        a.write_users(users)
        st.session_state['auth_user']=a.public_user(users.loc[index])
        a.log_event('profile_updated',target=user['username'])
        notify('profile_saved')
    st.subheader(tr('security'))
    with st.form('password_settings_form',clear_on_submit=True):
        current=st.text_input(tr('current_password'),type='password')
        new=st.text_input(tr('new_password'),type='password')
        confirmation=st.text_input(tr('confirm_password'),type='password')
        saved=submit(tr('change_password'))
    if saved:
        if not a.verify_password(current,str(row['password_hash'])):st.error(tr('password_wrong'))
        elif len(new)<6:st.error(tr('password_short'))
        elif new!=confirmation:st.error(tr('password_mismatch'))
        else:
            users.loc[index,'password_hash']=a.hash_password(new)
            a.write_users(users);a.log_event('password_changed',target=user['username'])
            notify('password_saved')
    st.subheader(tr('notifications'));st.info(tr('notifications_hint'))

def admin(a,user):
    st.title(tr('admin_panel'))
    users=a.users_df();visits=a.visits_df()
    metrics([('accounts',len(users)),('doctor',int((users['role']=='doctor').sum())),('total_visits',len(visits))])
    sections=['access','analytics','activity']
    saved_section=st.session_state.get('_admin_tab','access')
    section=st.radio(tr('admin_panel'),sections,index=sections.index(saved_section),format_func=formatter(),horizontal=True,key='admin_section',label_visibility='collapsed')
    st.session_state['_admin_tab']=section
    if section=='access':admin_access(a,user,users)
    elif section=='analytics':admin_analytics(a,users,visits)
    else:admin_activity(a)

def admin_access(a,user,users):
    st.subheader(tr('accounts'))
    table([tr('login'),tr('full_name'),tr('role'),tr('status'),tr('last_login')],
          [[row['username'],row['full_name'],tr(row['role']),tr(row['status']),row['last_login_at'] or tr('never')] for _,row in users.iterrows()])
    with st.expander(tr('create_account'),expanded=len(users)<2):
        with st.form('create_user_form',clear_on_submit=True):
            username=st.text_input(tr('login')+' *',help=tr('username_rules'))
            name=st.text_input(tr('full_name'))
            c1,c2=st.columns(2)
            with c1:role=st.selectbox(tr('role'),['doctor','admin'],format_func=formatter())
            with c2:status=st.selectbox(tr('status'),['active','blocked'],format_func=formatter())
            password=st.text_input(tr('temporary_password')+' *',type='password')
            confirmation=st.text_input(tr('confirm_password')+' *',type='password')
            submitted=submit(tr('create_account'),type='primary')
        if submitted:
            if not a.is_valid_username(a.normalize_username(username)):st.error(tr('username_rules'))
            elif len(password)<6:st.error(tr('password_short'))
            elif password!=confirmation:st.error(tr('password_mismatch'))
            elif a.normalize_username(username) in users['username'].str.lower().tolist():st.error(tr('account_duplicate'))
            else:
                ok,_=a.create_user_account(username,name,role,status,password)
                if ok:notify('account_created',id=a.normalize_username(username))
                else:st.error(tr('storage_error'))
    st.subheader(tr('edit_account'))
    selected=st.selectbox(tr('account'),users['username'].astype(str).tolist(),key='manage_account',placeholder=tr('choose'))
    row=users.loc[users['username']==selected].iloc[0]
    if st.session_state.get('_manage_selected') != selected:
        for prefix in ['role_','status_','reset_']:st.session_state.pop(prefix+selected,None)
        st.session_state['_manage_selected']=selected
    st.caption(tr('account_hint'))
    # Selection is outside the form. Keys include the account to prevent stale privileges.
    with st.form('manage_'+selected):
        role=st.selectbox(tr('new_role'),['doctor','admin'],format_func=formatter(),index=['doctor','admin'].index(row['role']),key='role_'+selected)
        status=st.selectbox(tr('new_status'),['active','blocked'],format_func=formatter(),index=['active','blocked'].index(row['status']),key='status_'+selected)
        password=st.text_input(tr('reset_password'),type='password',key='reset_'+selected)
        submitted=submit(tr('save'),type='primary')
    if submitted:
        updated=users.copy();mask=updated['username']==selected
        if selected==user['username'] and (role!='admin' or status!='active'):st.error(tr('self_admin'));return
        if password and len(password)<6:st.error(tr('password_short'));return
        updated.loc[mask,['role','status']]=[role,status]
        if not ((updated['role']=='admin')&(updated['status']=='active')).any():st.error(tr('last_admin'));return
        if password:
            updated.loc[mask,'password_hash']=a.hash_password(password)
            updated.loc[mask,['session_hash','session_expires_at']]=''
        a.write_users(updated);a.log_event('account_updated',target=selected)
        notify('access_saved')

def admin_analytics(a,users,visits):
    st.subheader(tr('analytics'))
    table([tr('risk'),tr('count')],[[tr(r),int((visits['risk_label']==r).sum())] for r in RISKS],tr('distribution'))
    doctors=users.loc[users['role']=='doctor']
    if doctors.empty:st.info(tr('empty'));return
    names={str(row['username']):str(row['full_name'])+' · '+str(row['username']) for _,row in doctors.iterrows()}
    selected=st.selectbox(tr('doctor'),list(names),format_func=names.get,key='analytics_doctor')
    events=a.activity_df();events=events.loc[events['actor']==selected].sort_values('event_at',ascending=False)
    calculated=events.loc[events['action']=='visit_calculated']
    # Attribute work to the immutable login in audit events, not mutable profile names.
    metrics([('total_visits',len(calculated)),('total_patients',calculated['target'].nunique()),('activity',len(events))])
    event_table(events.head(20))

def event_table(events):
    table([(tr('date')+' ('+timezone_label()+')'),tr('actor'),tr('action'),tr('target')],
          [[row['event_at'],row['actor'],tr(EVENTS.get(row['action'],'event_other')),row['target']] for _,row in events.iterrows()])

def admin_activity(a):
    st.subheader(tr('activity'))
    events=filtered_period(a.activity_df(),'event_at','event_period')
    actor=st.selectbox(tr('actor'),['all']+sorted(events['actor'].unique().tolist()),format_func=lambda x,all_label=tr('all'):all_label if x=='all' else x)
    if actor!='all':events=events.loc[events['actor']==actor]
    action=st.selectbox(tr('action'),['all']+sorted(events['action'].unique().tolist()),format_func=formatter({**EVENTS,'all':'all'}))
    if action!='all':events=events.loc[events['action']==action]
    events=events.sort_values('event_at',ascending=False)
    event_table(events.head(100))
    st.caption(tr('count_shown',shown=min(100,len(events)),total=len(events)))
    export=pd.DataFrame([[row['event_at'],row['actor'],tr(EVENTS.get(row['action'],'event_other')),row['target']] for _,row in events.iterrows()],columns=[(tr('date')+' ('+timezone_label()+')'),tr('actor'),tr('action'),tr('target')])
    st.download_button(tr('export_csv'),export.to_csv(index=False).encode('utf-8-sig'),'vascularai-activity.csv','text/csv',on_click='ignore')
