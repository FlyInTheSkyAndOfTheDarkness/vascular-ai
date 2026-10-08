"""Complete, explicit UI catalogue. Stored clinical values are never translated."""
import streamlit as st

LANGUAGES = {'ru': 'Русский', 'en': 'English', 'kk': 'Қазақша'}
SOURCE = '''
cabinet|Кабинет VascularAI|VascularAI workspace|VascularAI кабинеті
dashboard|Обзор|Overview|Шолу
intake|Новый расчёт|New assessment|Жаңа есептеу
patients|Пациентки|Patients|Пациенттер
reports|Отчёты|Reports|Есептер
settings|Настройки|Settings|Баптаулар
admin|Администратор|Administrator|Әкімші
doctor|Врач|Doctor|Дәрігер
admin_panel|Админ-панель|Administration|Әкімші панелі
access|Управление доступом|Account access|Қолжетімділікті басқару
analytics|Аналитика врачей|Doctor analytics|Дәрігерлер аналитикасы
activity|Журнал действий|Activity log|Әрекеттер журналы
login|Логин|Username|Логин
password|Пароль|Password|Құпиясөз
signin|Войти|Sign in|Кіру
signout|Выйти|Sign out|Шығу
welcome|Вход в систему|Sign in to your workspace|Жүйеге кіру
login_hint|Используйте аккаунт врача или администратора.|Use your doctor or administrator account.|Дәрігер немесе әкімші тіркелгісін пайдаланыңыз.
session_hint|Сессия сохраняется 14 дней. Кнопка «Выйти» завершает её.|Your session lasts 14 days. Sign out to end it.|Сессия 14 күн сақталады. Оны аяқтау үшін «Шығу» түймесін басыңыз.
invalid_login|Неверный логин или пароль, либо аккаунт отключён.|Incorrect credentials or an inactive account.|Логин не құпиясөз қате немесе тіркелгі өшірілген.
choose|Выберите|Select|Таңдаңыз
save|Сохранить|Save|Сақтау
cancel|Отмена|Cancel|Бас тарту
back|Назад|Back|Артқа
search|Поиск по ID или имени|Search by ID or name|ID немесе аты бойынша іздеу
all|Все|All|Барлығы
empty|Данных пока нет.|No data yet.|Әзірге деректер жоқ.
no_matches|По вашему запросу ничего не найдено.|No matching records.|Сұрауыңыз бойынша ештеңе табылмады.
clear_filters|Сбросить фильтры|Clear filters|Сүзгілерді тазалау
required|Обязательные показатели|Required measurements|Міндетті көрсеткіштер
required_hint|Все поля со знаком * обязательны. Введите фактические измерения.|All fields marked * are required. Enter actual measurements.|* белгісі бар өрістер міндетті. Нақты өлшемдерді енгізіңіз.
optional|Дополнительные сведения · необязательно|Additional information · optional|Қосымша мәліметтер · міндетті емес
optional_hint|Сохраняются в истории визита и не влияют на расчёт модели.|Saved with the visit; not used by the prediction model.|Қабылдау тарихында сақталады және модель есебіне әсер етпейді.
intake_title|Новый расчёт материнского риска|New maternal risk assessment|Аналық тәуекелді жаңа есептеу
intake_hint|Для оценки нужны шесть показателей и выбранная карта пациентки.|Select a patient and enter six measurements.|Пациент картасын таңдап, алты көрсеткішті енгізіңіз.
Age|Возраст пациентки|Patient age|Пациенттің жасы
SystolicBP|Систолическое давление|Systolic blood pressure|Систолалық қан қысымы
DiastolicBP|Диастолическое давление|Diastolic blood pressure|Диастолалық қан қысымы
BS|Сахар крови|Blood glucose|Қандағы қант
BodyTemp|Температура тела|Body temperature|Дене температурасы
HeartRate|Пульс в покое|Resting heart rate|Тыныштықтағы пульс
years|лет|years|жас
pressure_unit|мм рт. ст.|mmHg|мм сын. бағ.
glucose_unit|ммоль/л|mmol/L|ммоль/л
pulse_unit|уд/мин|bpm|соғ/мин
temp_hint|Введите температуру в градусах Цельсия (°C), например 36,6.|Enter the temperature in degrees Celsius (°C), for example 36.6.|Температураны Цельсий градусымен (°C) енгізіңіз, мысалы 36,6.
range_hint|Допустимый диапазон: {minimum}–{maximum} {unit}.|Accepted range: {minimum}–{maximum} {unit}.|Рұқсат етілген аралық: {minimum}–{maximum} {unit}.
invalid_number|{field}: введите число в диапазоне {minimum}–{maximum}.|{field}: enter a number between {minimum} and {maximum}.|{field}: {minimum}–{maximum} аралығындағы санды енгізіңіз.
missing|{field}: обязательное поле.|{field}: this field is required.|{field}: міндетті өріс.
pressure_error|Систолическое давление должно быть выше диастолического.|Systolic pressure must be higher than diastolic pressure.|Систолалық қысым диастолалық қысымнан жоғары болуы керек.
form_errors|Исправьте отмеченные ошибки. Расчёт не сохранён.|Correct the errors below. No assessment was saved.|Төмендегі қателерді түзетіңіз. Есеп сақталмады.
calculate|Рассчитать риск|Calculate risk|Тәуекелді есептеу
calculating|Выполняется расчёт…|Calculating…|Есептелуде…
attach_visit|Сохранить визит в карту пациентки|Save assessment to patient record|Есепті пациент картасына сақтау
calculation_saved|Расчёт сохранён в карту {id}.|Assessment saved to patient {id}.|Есеп {id} картасына сақталды.
calculation_ready|Расчёт готов. Результат не сохранён в карту.|Assessment complete. Result not saved to the record.|Есеп дайын. Нәтиже картаға сақталмады.
result|Результат расчёта|Assessment result|Есептеу нәтижесі
result_hint|Результат относится к отправленным значениям ниже. После изменения формы выполните новый расчёт.|This result uses the submitted values shown below. Recalculate after changing the form.|Нәтиже төмендегі жіберілген мәндерге негізделген. Өрістерді өзгертсеңіз, қайта есептеңіз.
probabilities|Вероятности классов|Class probabilities|Санаттардың ықтималдығы
low risk|Низкий риск|Low risk|Төмен тәуекел
mid risk|Средний риск|Moderate risk|Орташа тәуекел
high risk|Высокий риск|High risk|Жоғары тәуекел
confidence|Вероятность выбранного класса|Predicted class probability|Таңдалған санаттың ықтималдығы
factors|Вклад показателей в результат|Feature contributions|Көрсеткіштердің нәтижеге үлесі
contribution|Вклад SHAP|SHAP contribution|SHAP үлесі
measurement|Показатель|Measurement|Көрсеткіш
value|Значение|Value|Мәні
disclaimer|Демо-система поддержки решений. Результат не является медицинским заключением.|Decision support demo. This result is not a medical diagnosis.|Шешім қабылдауды қолдау деможүйесі. Нәтиже медициналық қорытынды емес.
model_limits|Модель оценивает общий материнский риск. Клиническая валидация для диагностики преэклампсии не выполнена.|The model estimates general maternal risk. It has not been clinically validated for preeclampsia diagnosis.|Модель жалпы аналық тәуекелді бағалайды. Преэклампсияны анықтау үшін клиникалық валидациядан өтпеген.
patient|Пациентка|Patient|Пациент
patient_id|ID пациентки|Patient ID|Пациент ID
initials|Инициалы / код|Initials / code|Инициалдар / код
create_patient|Создать карту|Create patient record|Пациент картасын жасау
new_patient|Новая карта пациентки|New patient record|Жаңа пациент картасы
patient_created|Карта {id} создана.|Patient {id} created.|{id} картасы жасалды.
patient_duplicate|Карта с таким ID уже существует.|This patient ID already exists.|Мұндай ID бар карта бұрыннан бар.
patient_hint|Создайте карту, чтобы сохранять расчёты и вложения.|Create a record to save assessments and attachments.|Есептер мен файлдарды сақтау үшін карта жасаңыз.
history|Карта и история|Record and history|Карта және тарих
visits|История расчётов|Assessment history|Есептеулер тарихы
visit|Расчёт|Assessment|Есептеу
open_visit|Открыть расчёт|Open assessment|Есепті ашу
open_patient|Открыть карту|Open record|Картаны ашу
week|Срок беременности, недель|Gestational age, weeks|Жүктілік мерзімі, апта
bmi|ИМТ, кг/м²|BMI, kg/m²|ДСИ, кг/м²
gravida|Число беременностей|Number of pregnancies|Жүктілік саны
parity|Число родов|Number of births|Босану саны
responsible|Ответственный врач|Responsible doctor|Жауапты дәрігер
status|Статус|Status|Күйі
active|Активен|Active|Белсенді
blocked|Отключён|Disabled|Өшірілген
patient_active|Активна|Active|Белсенді
patient_followup|Наблюдение|Follow-up|Бақылауда
patient_archive|Архив|Archived|Мұрағат
anamnesis|Анамнез|Medical history|Анамнез
note|Примечание|Note|Ескертпе
visit_note|Комментарий к визиту|Visit note|Қабылдау туралы түсініктеме
preeclampsia|Преэклампсия в анамнезе|History of preeclampsia|Анамнездегі преэклампсия
chronic|Хронические заболевания|Chronic conditions|Созылмалы аурулар
biomarkers|Биомаркеры|Biomarkers|Биомаркерлер
plgf|PLGF, пг/мл|PLGF, pg/mL|PLGF, пг/мл
sflt1|SFLT-1, пг/мл|SFLT-1, pg/mL|SFLT-1, пг/мл
papp_a|PAPP-A, MoM|PAPP-A, MoM|PAPP-A, MoM
map_value|Среднее артериальное давление, мм рт. ст.|Mean arterial pressure, mmHg|Орташа артериялық қысым, мм сын. бағ.
date|Дата и время|Date and time|Күні мен уақыты
created|Создан|Created|Жасалған
last_login|Последний вход|Last sign-in|Соңғы кіру
never|Ещё не входил|Never signed in|Әлі кірмеген
attachments|Вложения|Attachments|Тіркемелер
file|Файл|File|Файл
upload|Выбрать файл|Choose file|Файлды таңдау
upload_hint|PDF, изображения, таблицы и текст. До 20 МБ на файл.|PDF, images, spreadsheets and text. Up to 20 MB per file.|PDF, суреттер, кестелер және мәтін. Бір файлға 20 МБ дейін.
attach|Прикрепить файл|Attach file|Файлды тіркеу
attachment_saved|Файл прикреплён к карте.|File attached to the record.|Файл картаға тіркелді.
select_file|Выберите файл и дождитесь окончания загрузки.|Choose a file and wait for the upload to finish.|Файлды таңдап, жүктелу аяқталғанша күтіңіз.
download|Скачать|Download|Жүктеп алу
file_unavailable|Файл недоступен. Обратитесь к администратору.|File unavailable. Contact your administrator.|Файл қолжетімсіз. Әкімшіге хабарласыңыз.
total_visits|Всего расчётов|Total assessments|Барлық есептеулер
total_patients|Пациенток|Patients|Пациенттер
high_count|Расчётов с высоким риском|High-risk assessments|Жоғары тәуекелді есептер
recent|Последние расчёты|Recent assessments|Соңғы есептеулер
distribution|Распределение по классам|Distribution by class|Санаттар бойынша үлестірім
trend|Вероятность высокого риска по визитам|High-risk probability by visit|Қабылдаулар бойынша жоғары тәуекел ықтималдығы
count|Количество|Count|Саны
period|Период|Period|Кезең
all_time|За всё время|All time|Барлық уақыт
month|Последние 30 дней|Last 30 days|Соңғы 30 күн
week_period|Последние 7 дней|Last 7 days|Соңғы 7 күн
export_csv|Скачать CSV|Download CSV|CSV жүктеу
export_xlsx|Скачать Excel|Download Excel|Excel жүктеу
export_print|Версия для печати (HTML)|Print version (HTML)|Басып шығару нұсқасы (HTML)
print_hint|Откройте HTML-файл и выберите «Печать → Сохранить как PDF» в браузере.|Open the HTML file and use Print → Save as PDF in your browser.|HTML файлын ашып, браузерде «Басып шығару → PDF ретінде сақтау» пәрменін таңдаңыз.
export_ready|Файл подготовлен к скачиванию.|File ready to download.|Файл жүктеп алуға дайын.
profile|Профиль|Profile|Профиль
full_name|ФИО / отделение|Full name / department|Аты-жөні / бөлімше
email|Электронная почта|Email|Электрондық пошта
phone|Телефон|Phone|Телефон
clinic|Клиника|Clinic|Клиника
specialty|Специальность|Specialty|Мамандығы
profile_saved|Профиль обновлён.|Profile updated.|Профиль жаңартылды.
security|Смена пароля|Change password|Құпиясөзді өзгерту
current_password|Текущий пароль|Current password|Қазіргі құпиясөз
new_password|Новый пароль|New password|Жаңа құпиясөз
confirm_password|Повторите пароль|Confirm password|Құпиясөзді қайталаңыз
change_password|Изменить пароль|Change password|Құпиясөзді өзгерту
password_wrong|Текущий пароль указан неверно.|Current password is incorrect.|Қазіргі құпиясөз қате.
password_short|Пароль должен содержать не менее 6 символов.|Password must contain at least 6 characters.|Құпиясөз кемінде 6 таңбадан тұруы керек.
password_mismatch|Пароли не совпадают.|Passwords do not match.|Құпиясөздер сәйкес келмейді.
password_saved|Пароль обновлён.|Password updated.|Құпиясөз жаңартылды.
notifications|Уведомления|Notifications|Хабарландырулар
notifications_hint|Подтверждения действий показываются в кабинете. Отправка писем и SMS пока не подключена.|Action confirmations appear in the workspace. Email and SMS delivery is not connected yet.|Әрекеттердің растауы кабинетте көрсетіледі. Электрондық пошта мен SMS жіберу әзірге қосылмаған.
notification_empty|Новых действий пока нет.|No recent activity.|Әзірге жаңа әрекеттер жоқ.
accounts|Аккаунты|Accounts|Тіркелгілер
create_account|Создать аккаунт|Create account|Тіркелгі жасау
account_created|Аккаунт {id} создан.|Account {id} created.|{id} тіркелгісі жасалды.
account_duplicate|Такой логин уже существует.|This username already exists.|Бұл логин бұрыннан бар.
username_rules|Логин: 3–32 символа, латинские буквы, цифры, точка, дефис или подчёркивание.|Username: 3–32 Latin letters, digits, dots, hyphens or underscores.|Логин: 3–32 латын әрпі, сан, нүкте, дефис немесе астын сызу таңбасы.
role|Роль|Role|Рөл
temporary_password|Временный пароль|Temporary password|Уақытша құпиясөз
edit_account|Изменить доступ|Edit account access|Қолжетімділікті өзгерту
account|Аккаунт|Account|Тіркелгі
new_role|Новая роль|New role|Жаңа рөл
new_status|Новый статус|New status|Жаңа күй
reset_password|Новый пароль · оставьте пустым, чтобы сохранить текущий|New password · leave empty to keep the current one|Жаңа құпиясөз · қазіргісін сақтау үшін бос қалдырыңыз
access_saved|Доступ обновлён.|Account access updated.|Қолжетімділік жаңартылды.
self_admin|Нельзя отключить текущий аккаунт или снять у него права администратора.|You cannot disable your own account or remove your administrator role.|Өз тіркелгіңізді өшіруге немесе әкімші рөлін алып тастауға болмайды.
last_admin|Должен остаться хотя бы один активный администратор.|At least one active administrator must remain.|Кемінде бір белсенді әкімші қалуы керек.
action|Действие|Action|Әрекет
actor|Пользователь|User|Пайдаланушы
target|Объект|Target|Нысан
login_event|Вход в систему|Signed in|Жүйеге кірді
logout|Выход из системы|Signed out|Жүйеден шықты
account_created_event|Создан аккаунт|Account created|Тіркелгі жасалды
account_updated|Изменён доступ|Account access changed|Қолжетімділік өзгертілді
profile_updated|Обновлён профиль|Profile updated|Профиль жаңартылды
password_changed|Изменён пароль|Password changed|Құпиясөз өзгертілді
patient_created_event|Создана карта пациентки|Patient record created|Пациент картасы жасалды
visit_calculated|Выполнен расчёт|Assessment completed|Есептеу орындалды
attachment_uploaded|Добавлено вложение|File attached|Файл тіркелді
event_other|Действие в системе|System activity|Жүйедегі әрекет
page|Страница|Page|Бет
of|из|of|/
storage_error|Не удалось сохранить или загрузить данные. Повторите попытку. Успешное сохранение не подтверждено.|Unable to save or load data. Please retry. A successful save has not been confirmed.|Деректерді сақтау немесе жүктеу мүмкін болмады. Қайталап көріңіз. Сәтті сақталғаны расталмады.
model_missing|Модель недоступна. Обратитесь к администратору.|Model unavailable. Contact your administrator.|Модель қолжетімсіз. Әкімшіге хабарласыңыз.
model_title|Модель материнского риска|Maternal risk model|Аналық тәуекел моделі
model_loading|Загрузка 3D|Loading 3D|3D жүктелуде
model_ready|3D готово|3D ready|3D дайын
model_unavailable|3D недоступно|3D unavailable|3D қолжетімсіз
model_retry|Повторить загрузку|Retry loading|Қайта жүктеу
model_preparing|Подготовка 3D|Preparing 3D|3D дайындалуда
model_error|3D не загрузилось — показано изображение модели|3D failed to load — showing model image|3D жүктелмеді — модель суреті көрсетілді
clinical_workspace|Клинический кабинет|Clinical workspace|Клиникалық кабинет
detail|Сведения|Details|Мәліметтер
no_visit|Расчётов пока нет|No assessments yet|Әзірге есептеулер жоқ
count_shown|Показано {shown} из {total}|Showing {shown} of {total}|{total} жазбаның {shown} көрсетілді
view_details|Показатели расчёта|Assessment measurements|Есептеу көрсеткіштері
saved_values|Сохранённые значения|Saved measurements|Сақталған мәндер
filters|Фильтры|Filters|Сүзгілер
risk|Уровень риска|Risk class|Тәуекел деңгейі
account_hint|Выберите аккаунт выше: форма подставляет его текущую роль и статус.|Select an account above to load its current role and status.|Ағымдағы рөлі мен күйін жүктеу үшін жоғарыдан тіркелгіні таңдаңыз.
acknowledge|Закрыть уведомление|Dismiss notification|Хабарландыруды жабу
session_expired|Сессия завершена. Войдите снова.|Session ended. Sign in again.|Сессия аяқталды. Қайта кіріңіз.
file_limit|Размер файла превышает 20 МБ.|File exceeds 20 MB.|Файл көлемі 20 МБ-тан асады.
optional_suffix|необязательно|optional|міндетті емес
result_unsaved|Предварительный результат · не сохранён|Preview result · not saved|Алдын ала нәтиже · сақталмаған
result_saved|Сохранённый расчёт|Saved assessment|Сақталған есеп
changed_profile_hint|Имя для отображения. История пациенток сохраняется.|Display name. Patient history is preserved.|Көрсетілетін аты-жөні. Пациенттер тарихы сақталады.
'''
CATALOGUE = {}
for line in SOURCE.strip().splitlines():
    key, *values = line.split('|')
    assert len(values) == 3, key
    assert key not in CATALOGUE, key
    CATALOGUE[key] = dict(zip(LANGUAGES, values))

def tr(key, **values):
    lang = st.session_state.get('ui_language', 'ru')
    return CATALOGUE[key][lang].format(**values)

CATALOGUE["access_denied"] = {"ru": "Нет доступа к этой записи. Обновите страницу.", "en": "You do not have access to this record. Refresh the page.", "kk": "Бұл жазбаға қол жеткізуге рұқсат жоқ. Бетті жаңартыңыз."}
