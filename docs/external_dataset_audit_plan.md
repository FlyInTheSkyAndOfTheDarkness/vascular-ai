# План изучения внешних датасетов

Цель: перед включением новых maternal-health датасетов понять, можно ли их безопасно объединять с текущим UCI Maternal Health Risk Data Set.

## Кандидаты

| Приоритет | Датасет | Ссылка | Зачем смотреть | Риск |
| --- | --- | --- | --- | --- |
| 1 | Mendeley `ckhgtgdg4f` | https://data.mendeley.com/datasets/ckhgtgdg4f/1 | Более 6000 записей, близкие vitals, 3 класса риска, дополнительные `BMI`, `HbA1c`, `Fasting Glucose` | Нужно проверить единицы измерения и схему |
| 2 | Mendeley `p5w98dvbbk` | https://data.mendeley.com/datasets/p5w98dvbbk/1 | Более широкий клинический набор: vitals + BMI + осложнения + диабет + mental health | Возможен другой target и другая логика labels |
| 3 | Kaggle high-risk pregnancy | https://www.kaggle.com/datasets/ankurray00/maternal-health-and-high-risk-pregnancy-dataset | 18 колонок, obstetric history и fetal indicators | Некоммерческая лицензия `CC BY-NC-SA 4.0` |
| 4 | MOTHER_CARE1 / MOTHER_CARE2 | https://data.mendeley.com/datasets/fvdt76zwhn/2 / https://data.mendeley.com/datasets/fcbzdg5gjm/2 | Obstetric and delivery records для high-risk pregnancy | Высокий риск leakage из post-delivery и derived flags |
| 5 | UCI Cardiotocography | https://archive.ics.uci.edu/dataset/193/cardiotocography | Fetal-health module, не текущий maternal-vitals модуль | Другая задача и другие признаки |

## Что проверяем до обучения

1. Лицензия и доступ
   - Можно ли использовать для учебного, исследовательского, коммерческого прототипа.
   - Есть ли ограничения типа `non-commercial`.
   - Есть ли embargo или закрытый доступ.

2. Grain
   - Одна строка = один пациент, один визит, одна запись мониторинга или один delivery record.
   - Есть ли повторные записи одного пациента.
   - Есть ли timestamp или visit/session id.

3. Схема
   - Какие колонки совпадают с текущими: `Age`, `SystolicBP`, `DiastolicBP`, `BS`, `BodyTemp`, `HeartRate`.
   - Какие новые поля потенциально полезны: `BMI`, `HbA1c`, `Fasting Glucose`, `Gestational Age`, `Gravida`, `Parity`, `Hemoglobin`, `Previous Complications`.
   - Какие поля нельзя использовать на этапе первичного приема: `delivery_mode`, `birth_weight`, `Apgar`, neonatal outcomes.

4. Target
   - Target multiclass: low/mid/high или binary: high-risk vs not-high-risk.
   - Кто и как выставил label: врач, правила, derived flags, post-hoc outcome.
   - Совместим ли target с текущими классами.

5. Качество данных
   - Missing values, пустые строки, `Unknown`, `Not Recorded`, `N/A`.
   - Полные дубликаты и почти-дубликаты.
   - Нереальные значения: возраст, давление, температура, сахар, пульс, BMI.
   - Распределение классов.

6. Leakage
   - Исключить признаки, которые напрямую кодируют target: `risk_flag`, `high_risk`, `hypertension_flag`, `anemia_flag`.
   - Исключить признаки, известные только после родов, если модель должна работать до родов.
   - Проверить подозрительно высокую accuracy после добавления новых колонок.

7. Сравнение с текущим UCI
   - Совпадают ли единицы: температура F или C, glucose mmol/L или mg/dL.
   - Насколько отличаются распределения vitals.
   - Не ухудшается ли качество на holdout из исходного UCI.

## Решение после аудита

| Статус | Когда ставим |
| --- | --- |
| `include_now` | Схема понятна, лицензия подходит, target совместим, leakage нет |
| `include_with_mapping` | Нужна конвертация единиц или переименование колонок |
| `separate_model` | Другая задача: fetal health, delivery outcome, binary high-risk |
| `research_only` | Лицензия или качество не подходят для основного продукта |
| `reject` | Закрытый доступ, embargo, leakage нельзя убрать, непонятный target |

## Команда для первичного аудита

```powershell
python src\audit_dataset.py --input "path\to\dataset.csv" --name dataset_name
```

Отчет сохранится в `reports/dataset_audits/`.
