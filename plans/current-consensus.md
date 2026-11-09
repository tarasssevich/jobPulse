# Consensus Record: Шаг 1 — Предобработка данных и нормализация

## 4. Consensus Record: Шаг 4 — Моделирование динамики спроса и прогнозирование временных рядов

### 1. Архитектурное решение
- **Цель:** Реализовать модуль прогнозирования востребованности IT-навыков во времени (`src/demand_forecasting/`), анализирующий динамику проникновения компетенций $D_i(t) = \frac{V_i(t)}{V_{total}(t)} \times 100\%$ и строящий прогноз на горизонт 14–30 дней с доверительными интервалами (95% CI).
- **Компоненты модуля:**
  1. `src/demand_forecasting/timeseries.py`:
     - Агрегация публикаций по дням, фильтрация валидных дат, расчет суточной доли рынка $D_i(t)$.
     - Расчет 7-дневного скользящего среднего (7-Day Rolling Window) для устранения недельной сезонности (выходные дни).
  2. `src/demand_forecasting/forecaster.py`:
     - Реализация моделей прогнозирования (Holt's Linear Exponential Smoothing, ARIMA) через `statsmodels`.
     - Генерация точечного прогноза $\hat{y}_{t+h}$ и доверительных интервалов $[\hat{y}_{lower}, \hat{y}_{upper}]$.
     - Расчет метрик ретроспективного тестирования (Backtesting MAE, MAPE).
     - Классификация технологий по матрице спроса и динамики (Growth Rate $\Delta \%$, Momentum, Evergreen, Emerging).
  3. `src/demand_forecasting/visualizer.py`:
     - Создание академической иллюстрации `reports/figures/fig6_skill_demand_forecast.png` (300 DPI, 4 панели для ключевых технологий: Python, Docker, PostgreSQL, Go).
  4. `src/demand_forecasting/runner.py`:
     - Сквозной CLI-раннер: сохранение таблицы прогнозов в `data/processed/demand_forecast_summary.csv` и метрик в `data/processed/demand_forecast_metrics.json`.
  5. `tests/test_demand_forecasting.py`:
     - Модульные тесты агрегации временных рядов, моделей Холта/ARIMA, доверительных интервалов и генератора графиков.


### 1. Архитектурное решение
- **Цель:** Разработать модули эконометрического анализа зарплат, машинного обучения и генерации графиков публикации (`src/salary_model/` и `src/analysis/`).
- **Компоненты модуля:**
  1. `src/salary_model/econometrics.py`:
     - Расчет эконометрической надбавки за навык (Skill Premium): абсолютная разница в BYN, относительная разница (%), непараметрический тест Манна-Уитни ($U$-тест, $p$-value).
     - Фильтрация статистически значимых навыков ($p < 0.05$, минимальный порог встречаемости $N \ge 15$).
  2. `src/salary_model/models.py`:
     - Пайплайн обучения моделей предсказания зарплат на основе признаков (опыт, страна, топ-50 бинарных индикаторов навыков):
       - Baseline (Dummy / Median);
       - Ridge Regressor (с L2-регуляризацией);
       - Random Forest Regressor.
     - Оценка качества: 5-Fold Cross-Validation и отложенная выборка (Test 20%), расчет $R^2$, MAE, RMSE.
     - Извлечение важности признаков (MDI Feature Importance).
  3. `src/analysis/visualizer.py`:
     - Создание 5 академических графиков высокой четкости (300 DPI) в строгом стиле для статьи (без неоновых градиентов):
       - `fig1_salary_distribution_by_grade.png` (Boxplot по грейдам);
       - `fig2_top_skills_demand.png` (Топ-20 навыков по доле рынка);
       - `fig3_skill_premium.png` (Топ-15 навыков по зарплатной премии);
       - `fig4_skill_cooccurrence_matrix.png` (Матрица совместной встречаемости технологий);
       - `fig5_feature_importance_model.png` (Важность признаков Random Forest).
     - Сохранение графиков в `reports/figures/`.
  4. `src/salary_model/runner.py`:
     - CLI-оркестратор: запуск расчетов, сохранение метрик в `data/processed/metrics.json` и экспорт графиков.
  5. `tests/test_salary_model.py`:
     - Модульные тесты эконометрических функций, обучения моделей, метрик и генератора графиков.


### 1. Архитектурное решение
- **Цель:** Реализовать модуль экстракции компетенций (`src/skills/`), извлекающий канонические навыки из списка `key_skills` и очищенного текста `description_clean` для всех вакансий.
- **Компоненты модуля:**
  1. `src/skills/taxonomy.py`:
     - Таксономия по 7 каноническим категориям: `languages`, `frameworks`, `databases`, `devops_cloud`, `architecture_api`, `testing_qa`, `data_ai`.
     - Словарь синонимов и алиасов (более 150 IT-технологий) с канонизацией (например, `k8s` -> `Kubernetes`, `postgres` -> `PostgreSQL`, `fastapi` -> `FastAPI`).
     - Поддержка языков с неалфавитными символами (`C++`, `C#`, `.NET`, `Node.js`, `CI/CD`).
  2. `src/skills/extractor.py`:
     - Класс `SkillExtractor`:
       - Компиляция безопасных регулярных выражений с границами слов (`\b`, с учетом `+`, `#`, `.`, `/`).
       - Извлечение из `key_skills` (быстрый маппинг).
       - Извлечение из `description_clean` (глубокий поиск).
       - Объединение без дубликатов с сохранением категориальной принадлежности.
  3. `src/skills/enricher.py`:
     - Функция обогащения аналитического датасета `enrich_vacancies_dataset()`.
     - Добавление колонок: `extracted_skills`, `skills_by_category`, `skills_count`.
     - Генерация матрицы бинарных признаков (One-Hot / Multi-Hot) для топ-50 ключевых технологий (необходимо для моделей прогнозирования зарплат на Шаге 3).
     - Сохранение в `data/processed/vacancies_enriched.parquet` и `data/processed/vacancies_enriched.csv`.
  4. `tests/test_skills.py`:
     - Полный набор модульных тестов: алиасы, краевые случаи с `C++`, `C#`, `.NET`, защита от ложных срабатываний (`go`, `r`, `c`), категоризация, сохранение структуры.

- **Цель:** Реализовать модуль предобработки сырых данных вакансий (`src/preprocessing/`), извлекающий из `data/staging.db` (2 221 вакансия) нормализованный аналитический датасет согласно `docs/DATA_SPECIFICATION.md`.
- **Компоненты модуля:**
  1. `src/preprocessing/salary_normalizer.py`:
     - Конвертация валют в BYN (BYN=1.0, RUR/RUB=0.034, USD=3.30, EUR=3.55, KZT=0.0068).
     - Учет `gross == True` (очистка от 13% подоходного налога net = gross * 0.87 для сопоставимости чистых доходов, либо стандартная нормализация).
     - Расчет `salary_mid_byn`: среднее между `from` и `to`, либо `from`, либо `to` при наличии только одной границы.
  2. `src/preprocessing/text_cleaner.py`:
     - Очистка HTML-разметки (regex + html unescape), удаление спецсимволов, нормализация пробелов.
  3. `src/preprocessing/experience_mapper.py`:
     - Стандартизация опыта: `noExperience` -> `Junior (0)`, `between1And3` -> `Middle (1-3)`, `between3And6` -> `Middle+ (3-6)`, `moreThan6` -> `Senior (6+)`.
  4. `src/preprocessing/pipeline.py`:
     - Класс `DataCleaningPipeline` со сквозным методом `process_all()`, возвращающий pandas DataFrame и сохраняющий очищенные данные в `data/processed/vacancies_clean.parquet` и `data/processed/vacancies_clean.csv`.
- **Контроль качества:**
  - Модульные тесты `tests/test_preprocessing.py` с покрытием всех граничных условий.
  - Проверка линтером `ruff check`.
  - Запуск код-ревью через специализированного субагента `code-reviewer`.
