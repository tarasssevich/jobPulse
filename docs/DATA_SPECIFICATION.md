# Спецификация данных (Data Specification)

Документ формализует правила отбора вакансий, структуру слоев данных и контракт полей для проекта **JobPulse**.

---

## 1. Географический охват

Первичный фокус исследования — рынок труда стран СНГ:
- **Беларусь** (`area=16` в номенклатуре HH);
- **Россия** (`area=113`);
- **Казахстан** (`area=40`).

Другие страны СНГ (Узбекистан, Армения, Грузия, Азербайджан) могут быть включены при наличии репрезентативного объема данных без нарушения баланса выборки.

---

## 2. Критерии принадлежности к IT-сфере

В выборку включаются все профессиональные роли, непосредственно связанные с разработкой, эксплуатацией, анализом данных и управлением IT-продуктами:

### Включаемые направления:
1. **Software Engineering**: Backend, Frontend, Fullstack, Mobile (iOS/Android), Desktop, Embedded, GameDev, System.
2. **Data & AI**: Data Analyst, Data Scientist, ML Engineer, Data Engineer, BI Analyst, Computer Vision / NLP.
3. **Infrastructure & Ops**: DevOps, SRE, Cloud Engineer, System Administrator, Database Administrator (DBA).
4. **Quality Assurance**: Manual QA, Automation QA, Performance / Security Testing.
5. **Security**: Information Security Specialist, SOC Analyst, AppSec Engineer, Pentester.
6. **Product & Tech Management**: Product Manager, Project Manager (IT), Technical Product Owner, Scrum Master.

### Правила исключения (Noise Filtering):
- Операторы ввода данных, специалисты 1-й линии технической поддержки без инженерных навыков;
- Преподаватели информатики начального/школьного уровня;
- Менеджеры по холодным продажам IT-услуг;
- Нетехнический копирайтинг и контент-менеджмент.

---

## 3. Трехслойная архитектура данных (Data Layers)

```
[Raw Layer (Неизменяемый)]
   │  • data/raw/*.jsonl (строгий сырой ответ API)
   │  • data/staging.db (SQLite: дедупликация по vacancy_id)
   ▼
[Clean Layer (Очищенный)]
   │  • PostgreSQL / Паркет
   │  • Нормализованные валюты (BYN), очищенный текст, стандартизованный опыт
   ▼
[Analytical Layer (Аналитический витринный)]
   │  • Связанная модель вакансий и canonical skills (Many-to-Many)
   │  • Готовые матрицы признаков для регрессионных моделей и прогнозирования
```

---

## 4. Контракт основных полей вакансии (Analytical Schema)

| Поле | Тип | Описание | Пример |
|---|---|---|---|
| `vacancy_id` | String | Уникальный ID вакансии в источнике | `"10482910"` |
| `source` | String | Источник данных | `"hh.ru"` |
| `title` | String | Название вакансии | `"Senior Python Developer"` |
| `company_name` | String | Название компании-работодателя | `"Yandex"` |
| `country` | String | Страна локации | `"Беларусь"` |
| `city` | String | Город локации | `"Минск"` |
| `salary_from` | Float (Nullable) | Нижняя граница исходной зарплаты | `5000.0` |
| `salary_to` | Float (Nullable) | Верхняя граница исходной зарплаты | `7000.0` |
| `salary_currency` | String (Nullable) | Исходная валюта | `"BYN"`, `"RUR"`, `"USD"` |
| `salary_gross` | Boolean (Nullable) | Указана ли зарплата до вычета налогов | `true` |
| `salary_mid_byn` | Float (Nullable) | Нормализованный midpoint в BYN | `6000.0` |
| `experience_raw` | String | Исходное требование к опыту | `"От 3 до 6 лет"` |
| `experience_level` | String | Нормализованный уровень | `"Middle+"` / `"Senior"` |
| `employment_type` | String | Тип занятости | `"full"`, `"part"`, `"project"` |
| `work_format` | String | Формат работы | `"remote"`, `"office"`, `"hybrid"` |
| `description_clean` | String | Очищенный текст описания | `"Разработка бэкенда на FastAPI..."` |
| `published_at` | Timestamp | Дата и время публикации | `2026-10-25T14:30:00` |
| `url` | String | Ссылка на оригинальную вакансию | `https://hh.ru/vacancy/...` |
| `skills` | List[String] | Список извлеченных canonical навыков | `["Python", "FastAPI", "Docker"]` |
