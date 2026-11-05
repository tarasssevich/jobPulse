"""Hard Skills Taxonomy for JobPulse IT Job Market Research.

Defines canonical IT skill categories, standardized canonical skills, and alias dictionaries
covering 180+ IT technologies across languages, frameworks, databases, DevOps, architecture,
testing, and data/AI domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class SkillCategory(StrEnum):
    """Canonical categories for hard skills classification."""

    LANGUAGES = "languages"
    FRAMEWORKS = "frameworks"
    DATABASES = "databases"
    DEVOPS_CLOUD = "devops_cloud"
    ARCHITECTURE_API = "architecture_api"
    TESTING_QA = "testing_qa"
    DATA_AI = "data_ai"


# Canonical category constants
LANGUAGES: Final[SkillCategory] = SkillCategory.LANGUAGES
FRAMEWORKS: Final[SkillCategory] = SkillCategory.FRAMEWORKS
DATABASES: Final[SkillCategory] = SkillCategory.DATABASES
DEVOPS_CLOUD: Final[SkillCategory] = SkillCategory.DEVOPS_CLOUD
ARCHITECTURE_API: Final[SkillCategory] = SkillCategory.ARCHITECTURE_API
TESTING_QA: Final[SkillCategory] = SkillCategory.TESTING_QA
DATA_AI: Final[SkillCategory] = SkillCategory.DATA_AI


@dataclass(frozen=True, slots=True)
class SkillDefinition:
    """Immutable definition of a canonical skill entity."""

    canonical_name: str
    category: SkillCategory
    aliases: tuple[str, ...]


# Canonical Skill Definitions covering 180+ IT technologies
_DEFINITIONS: tuple[SkillDefinition, ...] = (
    # =========================================================================
    # 1. LANGUAGES
    # =========================================================================
    SkillDefinition("Python", LANGUAGES, ("python", "python3", "python2", "python 3", "питон")),
    SkillDefinition("Java", LANGUAGES, ("java", "core java", "java core", "джава")),
    SkillDefinition("JavaScript", LANGUAGES, ("javascript", "js", "ecmascript", "es6", "es6+", "джаваскрипт")),
    SkillDefinition("TypeScript", LANGUAGES, ("typescript", "ts", "тайпскрипт")),
    SkillDefinition("C++", LANGUAGES, ("c++", "cpp", "visual c++", "c/c++")),
    SkillDefinition("C#", LANGUAGES, ("c#", "csharp", "c-sharp", "c sharp", "си шарп")),
    SkillDefinition("C", LANGUAGES, ("c language", "c lang", "язык c")),
    SkillDefinition("Go", LANGUAGES, ("golang", "go", "голанг")),
    SkillDefinition("Rust", LANGUAGES, ("rust", "rustlang", "раст")),
    SkillDefinition("Kotlin", LANGUAGES, ("kotlin", "котлин")),
    SkillDefinition("Swift", LANGUAGES, ("swift", "свифт")),
    SkillDefinition("PHP", LANGUAGES, ("php", "php7", "php8", "пхп")),
    SkillDefinition("Ruby", LANGUAGES, ("ruby", "руби")),
    SkillDefinition("Scala", LANGUAGES, ("scala", "скала")),
    SkillDefinition("R", LANGUAGES, ("r", "rlang", "r language", "r-project")),
    SkillDefinition("Dart", LANGUAGES, ("dart", "дарт")),
    SkillDefinition("SQL", LANGUAGES, ("sql", "t-sql", "tsql", "pl/sql", "plsql", "ansi sql", "скл")),
    SkillDefinition("Bash", LANGUAGES, ("bash", "sh", "shell", "zsh", "баш")),
    SkillDefinition("PowerShell", LANGUAGES, ("powershell", "pwsh")),
    SkillDefinition("HTML", LANGUAGES, ("html", "html5")),
    SkillDefinition("CSS", LANGUAGES, ("css", "css3")),
    SkillDefinition("Lua", LANGUAGES, ("lua", "луа")),
    SkillDefinition("Perl", LANGUAGES, ("perl",)),
    SkillDefinition("Groovy", LANGUAGES, ("groovy",)),
    SkillDefinition("Elixir", LANGUAGES, ("elixir",)),
    SkillDefinition("Haskell", LANGUAGES, ("haskell",)),
    SkillDefinition("Erlang", LANGUAGES, ("erlang",)),
    SkillDefinition("1C", LANGUAGES, ("1c", "1с", "1c enterprise", "1с:предприятие", "1с предприятие", "1c:enterprise", "1с программирование")),
    SkillDefinition("Solidity", LANGUAGES, ("solidity",)),
    SkillDefinition("VBA", LANGUAGES, ("vba", "visual basic")),
    SkillDefinition("Assembly", LANGUAGES, ("assembly", "assembler", "asm")),

    # =========================================================================
    # 2. FRAMEWORKS
    # =========================================================================
    SkillDefinition("React", FRAMEWORKS, ("react", "react.js", "reactjs", "реакт")),
    SkillDefinition("Angular", FRAMEWORKS, ("angular", "angular.js", "angularjs", "angular 2+", "ангуляр")),
    SkillDefinition("Vue.js", FRAMEWORKS, ("vue", "vue.js", "vuejs", "vue 3", "vue3", "vue 2", "vue2", "вью")),
    SkillDefinition("Next.js", FRAMEWORKS, ("next.js", "nextjs", "next js")),
    SkillDefinition("Nuxt.js", FRAMEWORKS, ("nuxt", "nuxt.js", "nuxtjs", "nuxt js")),
    SkillDefinition("Svelte", FRAMEWORKS, ("svelte", "sveltekit")),
    SkillDefinition("Django", FRAMEWORKS, ("django", "джанго", "drf", "django rest framework")),
    SkillDefinition("FastAPI", FRAMEWORKS, ("fastapi", "fast api", "фастапи")),
    SkillDefinition("Flask", FRAMEWORKS, ("flask", "фласк")),
    SkillDefinition("Spring", FRAMEWORKS, ("spring", "spring boot", "spring framework", "spring cloud", "spring data", "spring security", "спринг", "спринг бут")),
    SkillDefinition(".NET", FRAMEWORKS, (".net", "dotnet", ".net core", "dot net", "asp.net", "asp.net core", "entity framework", "ef core", "дотнет")),
    SkillDefinition("Node.js", FRAMEWORKS, ("node.js", "nodejs", "node js", "нода")),
    SkillDefinition("Express.js", FRAMEWORKS, ("express", "express.js", "expressjs")),
    SkillDefinition("NestJS", FRAMEWORKS, ("nestjs", "nest.js", "nest js")),
    SkillDefinition("Ruby on Rails", FRAMEWORKS, ("ruby on rails", "rails", "ror")),
    SkillDefinition("Laravel", FRAMEWORKS, ("laravel", "ларавел")),
    SkillDefinition("Symfony", FRAMEWORKS, ("symfony", "симфони")),
    SkillDefinition("Ktor", FRAMEWORKS, ("ktor",)),
    SkillDefinition("Quarkus", FRAMEWORKS, ("quarkus",)),
    SkillDefinition("Micronaut", FRAMEWORKS, ("micronaut",)),
    SkillDefinition("Fastify", FRAMEWORKS, ("fastify",)),
    SkillDefinition("Gin", FRAMEWORKS, ("gin", "gin-gonic")),
    SkillDefinition("Fiber", FRAMEWORKS, ("fiber", "gofiber")),
    SkillDefinition("Actix", FRAMEWORKS, ("actix", "actix-web")),
    SkillDefinition("Flutter", FRAMEWORKS, ("flutter", "флаттер")),
    SkillDefinition("React Native", FRAMEWORKS, ("react native", "react-native")),
    SkillDefinition("SwiftUI", FRAMEWORKS, ("swiftui",)),
    SkillDefinition("Jetpack Compose", FRAMEWORKS, ("jetpack compose", "compose")),
    SkillDefinition("Electron", FRAMEWORKS, ("electron", "electron.js")),
    SkillDefinition("Blazor", FRAMEWORKS, ("blazor",)),
    SkillDefinition("Unity", FRAMEWORKS, ("unity", "unity3d", "юнити")),
    SkillDefinition("Unreal Engine", FRAMEWORKS, ("unreal engine", "ue4", "ue5")),
    SkillDefinition("Celery", FRAMEWORKS, ("celery", "селери")),
    SkillDefinition("Redux", FRAMEWORKS, ("redux", "redux toolkit", "rtk")),
    SkillDefinition("Tailwind CSS", FRAMEWORKS, ("tailwind", "tailwindcss", "tailwind css")),
    SkillDefinition("Bootstrap", FRAMEWORKS, ("bootstrap", "бутстрап")),
    SkillDefinition("AIOHTTP", FRAMEWORKS, ("aiohttp",)),
    SkillDefinition("PyQt", FRAMEWORKS, ("pyqt", "pyqt5", "pyqt6", "pyside")),

    # =========================================================================
    # 3. DATABASES
    # =========================================================================
    SkillDefinition("PostgreSQL", DATABASES, ("postgresql", "postgres", "pgsql", "постгрес", "постгре")),
    SkillDefinition("MySQL", DATABASES, ("mysql", "майскл")),
    SkillDefinition("SQLite", DATABASES, ("sqlite", "sqlite3")),
    SkillDefinition("Microsoft SQL Server", DATABASES, ("mssql", "ms sql", "sql server", "microsoft sql server")),
    SkillDefinition("Oracle DB", DATABASES, ("oracle", "oracle db", "oracle database")),
    SkillDefinition("Redis", DATABASES, ("redis", "редис")),
    SkillDefinition("MongoDB", DATABASES, ("mongodb", "mongo", "монго")),
    SkillDefinition("ClickHouse", DATABASES, ("clickhouse", "кликхаус")),
    SkillDefinition("Elasticsearch", DATABASES, ("elasticsearch", "elastic search", "эластик")),
    SkillDefinition("OpenSearch", DATABASES, ("opensearch",)),
    SkillDefinition("Cassandra", DATABASES, ("cassandra", "apache cassandra")),
    SkillDefinition("ScyllaDB", DATABASES, ("scylla", "scylladb")),
    SkillDefinition("DynamoDB", DATABASES, ("dynamodb", "aws dynamodb")),
    SkillDefinition("MariaDB", DATABASES, ("mariadb",)),
    SkillDefinition("Neo4j", DATABASES, ("neo4j",)),
    SkillDefinition("CouchDB", DATABASES, ("couchdb",)),
    SkillDefinition("Memcached", DATABASES, ("memcached",)),
    SkillDefinition("TimescaleDB", DATABASES, ("timescaledb", "timescale")),
    SkillDefinition("ArangoDB", DATABASES, ("arangodb",)),
    SkillDefinition("Supabase", DATABASES, ("supabase",)),
    SkillDefinition("Firebase", DATABASES, ("firebase", "firestore")),
    SkillDefinition("Snowflake", DATABASES, ("snowflake",)),

    # =========================================================================
    # 4. DEVOPS_CLOUD
    # =========================================================================
    SkillDefinition("Docker", DEVOPS_CLOUD, ("docker", "docker compose", "docker-compose", "докер")),
    SkillDefinition("Kubernetes", DEVOPS_CLOUD, ("kubernetes", "k8s", "кубернетес", "кубер")),
    SkillDefinition("CI/CD", DEVOPS_CLOUD, ("ci/cd", "cicd", "continuous integration", "continuous deployment")),
    SkillDefinition("Git", DEVOPS_CLOUD, ("git", "github", "gitlab", "bitbucket", "гит")),
    SkillDefinition("Linux", DEVOPS_CLOUD, ("linux", "ubuntu", "debian", "centos", "rhel", "redhat", "alpine", "линукс")),
    SkillDefinition("Ansible", DEVOPS_CLOUD, ("ansible", "ансибл")),
    SkillDefinition("Terraform", DEVOPS_CLOUD, ("terraform", "терраформ")),
    SkillDefinition("Helm", DEVOPS_CLOUD, ("helm",)),
    SkillDefinition("Jenkins", DEVOPS_CLOUD, ("jenkins", "дженкинс")),
    SkillDefinition("GitLab CI", DEVOPS_CLOUD, ("gitlab ci", "gitlab-ci", "gitlab ci/cd")),
    SkillDefinition("GitHub Actions", DEVOPS_CLOUD, ("github actions", "gh actions")),
    SkillDefinition("AWS", DEVOPS_CLOUD, ("aws", "amazon web services", "amazon aws")),
    SkillDefinition("Azure", DEVOPS_CLOUD, ("azure", "microsoft azure", "ажур")),
    SkillDefinition("GCP", DEVOPS_CLOUD, ("gcp", "google cloud", "google cloud platform")),
    SkillDefinition("Nginx", DEVOPS_CLOUD, ("nginx", "энджинкс", "нжинкс")),
    SkillDefinition("Apache HTTP Server", DEVOPS_CLOUD, ("apache", "httpd", "apache http")),
    SkillDefinition("Traefik", DEVOPS_CLOUD, ("traefik",)),
    SkillDefinition("Prometheus", DEVOPS_CLOUD, ("prometheus", "прометеус")),
    SkillDefinition("Grafana", DEVOPS_CLOUD, ("grafana", "графана")),
    SkillDefinition("ELK Stack", DEVOPS_CLOUD, ("elk", "elk stack", "logstash", "kibana")),
    SkillDefinition("Vault", DEVOPS_CLOUD, ("vault", "hashicorp vault")),
    SkillDefinition("ArgoCD", DEVOPS_CLOUD, ("argocd", "argo cd")),
    SkillDefinition("OpenShift", DEVOPS_CLOUD, ("openshift",)),
    SkillDefinition("Pulumi", DEVOPS_CLOUD, ("pulumi",)),
    SkillDefinition("Vagrant", DEVOPS_CLOUD, ("vagrant",)),
    SkillDefinition("Datadog", DEVOPS_CLOUD, ("datadog",)),
    SkillDefinition("VictoriaMetrics", DEVOPS_CLOUD, ("victoriametrics",)),
    SkillDefinition("Graylog", DEVOPS_CLOUD, ("graylog",)),
    SkillDefinition("Sentry", DEVOPS_CLOUD, ("sentry",)),
    SkillDefinition("OpenTelemetry", DEVOPS_CLOUD, ("opentelemetry", "otel")),

    # =========================================================================
    # 5. ARCHITECTURE_API
    # =========================================================================
    SkillDefinition("REST API", ARCHITECTURE_API, ("rest api", "rest", "restful", "rest-api", "рест", "рест апи")),
    SkillDefinition("GraphQL", ARCHITECTURE_API, ("graphql", "графкл")),
    SkillDefinition("gRPC", ARCHITECTURE_API, ("grpc",)),
    SkillDefinition("Microservices", ARCHITECTURE_API, ("microservices", "microservice", "микросервисы", "микросервисная архитектура", "microservice architecture")),
    SkillDefinition("SOLID", ARCHITECTURE_API, ("solid", "solid principles", "принципы solid")),
    SkillDefinition("Clean Architecture", ARCHITECTURE_API, ("clean architecture", "чистая архитектура")),
    SkillDefinition("Design Patterns", ARCHITECTURE_API, ("design patterns", "design pattern", "паттерны проектирования", "паттерн проектирования", "gof patterns")),
    SkillDefinition("DDD", ARCHITECTURE_API, ("ddd", "domain-driven design", "domain driven design")),
    SkillDefinition("Event-Driven Architecture", ARCHITECTURE_API, ("event-driven", "eda", "event-driven architecture", "событийно-ориентированная архитектура")),
    SkillDefinition("Kafka", ARCHITECTURE_API, ("kafka", "apache kafka", "кафка")),
    SkillDefinition("RabbitMQ", ARCHITECTURE_API, ("rabbitmq", "rabbit mq", "раббит")),
    SkillDefinition("WebSocket", ARCHITECTURE_API, ("websocket", "websockets", "вебсокеты")),
    SkillDefinition("OAuth", ARCHITECTURE_API, ("oauth", "oauth2", "oauth 2.0")),
    SkillDefinition("JWT", ARCHITECTURE_API, ("jwt", "json web token", "json web tokens")),
    SkillDefinition("OpenAPI / Swagger", ARCHITECTURE_API, ("swagger", "openapi", "openapi 3.0", "сваггер")),
    SkillDefinition("SOAP", ARCHITECTURE_API, ("soap",)),
    SkillDefinition("ETL", ARCHITECTURE_API, ("etl", "elt")),
    SkillDefinition("AsyncIO", ARCHITECTURE_API, ("asyncio", "асинхронность")),

    # =========================================================================
    # 6. TESTING_QA
    # =========================================================================
    SkillDefinition("Pytest", TESTING_QA, ("pytest", "пайтест")),
    SkillDefinition("JUnit", TESTING_QA, ("junit", "junit5", "junit 5")),
    SkillDefinition("Selenium", TESTING_QA, ("selenium", "selenium webdriver", "селениум")),
    SkillDefinition("Postman", TESTING_QA, ("postman", "постман")),
    SkillDefinition("Cypress", TESTING_QA, ("cypress",)),
    SkillDefinition("Playwright", TESTING_QA, ("playwright",)),
    SkillDefinition("TestCafe", TESTING_QA, ("testcafe",)),
    SkillDefinition("Appium", TESTING_QA, ("appium",)),
    SkillDefinition("JMeter", TESTING_QA, ("jmeter", "apache jmeter")),
    SkillDefinition("Locust", TESTING_QA, ("locust",)),
    SkillDefinition("Mockito", TESTING_QA, ("mockito",)),
    SkillDefinition("Allure", TESTING_QA, ("allure", "allure report", "аллюр")),
    SkillDefinition("Unit Testing", TESTING_QA, ("unit testing", "юнит-тестирование", "юнит тесты", "юнит тест", "unit tests", "unit test", "модульное тестирование")),
    SkillDefinition("Integration Testing", TESTING_QA, ("integration testing", "интеграционное тестирование", "интеграционные тесты", "integration tests", "integration test")),
    SkillDefinition("E2E Testing", TESTING_QA, ("e2e", "end-to-end", "e2e testing", "e2e tests")),
    SkillDefinition("TestRail", TESTING_QA, ("testrail",)),
    SkillDefinition("QA Automation", TESTING_QA, ("qa automation", "автоматизация тестирования", "автотесты", "test automation")),
    SkillDefinition("K6", TESTING_QA, ("k6",)),

    # =========================================================================
    # 7. DATA_AI
    # =========================================================================
    SkillDefinition("Pandas", DATA_AI, ("pandas", "пандас")),
    SkillDefinition("NumPy", DATA_AI, ("numpy", "нампи")),
    SkillDefinition("Scikit-Learn", DATA_AI, ("scikit-learn", "sklearn", "сайкит")),
    SkillDefinition("PyTorch", DATA_AI, ("pytorch", "torch", "пайторч")),
    SkillDefinition("TensorFlow", DATA_AI, ("tensorflow", "tf", "тензорфлоу")),
    SkillDefinition("Keras", DATA_AI, ("keras",)),
    SkillDefinition("Airflow", DATA_AI, ("airflow", "apache airflow", "эйрфлоу")),
    SkillDefinition("Spark", DATA_AI, ("spark", "apache spark", "pyspark", "спарк")),
    SkillDefinition("Hadoop", DATA_AI, ("hadoop", "apache hadoop", "хадуп")),
    SkillDefinition("LLM", DATA_AI, ("llm", "large language models", "большие языковые модели", "chatgpt", "gpt-4", "gpt")),
    SkillDefinition("LangChain", DATA_AI, ("langchain",)),
    SkillDefinition("LlamaIndex", DATA_AI, ("llamaindex",)),
    SkillDefinition("Hugging Face", DATA_AI, ("hugging face", "huggingface", "transformers")),
    SkillDefinition("Computer Vision", DATA_AI, ("computer vision", "opencv", "компьютерное зрение")),
    SkillDefinition("NLP", DATA_AI, ("nlp", "natural language processing", "обработка естественного языка")),
    SkillDefinition("SciPy", DATA_AI, ("scipy",)),
    SkillDefinition("Matplotlib", DATA_AI, ("matplotlib",)),
    SkillDefinition("Seaborn", DATA_AI, ("seaborn",)),
    SkillDefinition("Plotly", DATA_AI, ("plotly",)),
    SkillDefinition("dbt", DATA_AI, ("dbt",)),
    SkillDefinition("Polars", DATA_AI, ("polars",)),
    SkillDefinition("XGBoost", DATA_AI, ("xgboost",)),
    SkillDefinition("LightGBM", DATA_AI, ("lightgbm",)),
    SkillDefinition("CatBoost", DATA_AI, ("catboost", "катбуст")),
    SkillDefinition("MLflow", DATA_AI, ("mlflow",)),
    SkillDefinition("DVC", DATA_AI, ("dvc",)),
    SkillDefinition("Tableau", DATA_AI, ("tableau", "табло")),
    SkillDefinition("Power BI", DATA_AI, ("power bi", "powerbi", "пауэр биай")),
    SkillDefinition("Apache Flink", DATA_AI, ("flink", "apache flink")),
    SkillDefinition("Vector DB", DATA_AI, ("vector db", "chromadb", "qdrant", "pinecone", "weaviate", "milvus")),
)


def _build_taxonomy() -> tuple[
    dict[str, SkillDefinition],
    dict[str, str],
    dict[str, SkillCategory],
    dict[SkillCategory, tuple[str, ...]],
]:
    """Construct verified mappings from canonical definitions with collision checks."""
    taxonomy_map: dict[str, SkillDefinition] = {}
    alias_map: dict[str, str] = {}
    category_map: dict[str, SkillCategory] = {}
    by_category_temp: dict[SkillCategory, list[str]] = {cat: [] for cat in SkillCategory}

    for item in _DEFINITIONS:
        canonical = item.canonical_name
        if canonical in taxonomy_map:
            raise ValueError(f"Duplicate canonical skill defined: {canonical}")

        taxonomy_map[canonical] = item
        category_map[canonical] = item.category
        by_category_temp[item.category].append(canonical)

        # Include lowercase canonical name itself as alias
        all_aliases = {canonical.lower()}
        for alias in item.aliases:
            normalized_alias = alias.strip().lower()
            if not normalized_alias:
                continue
            all_aliases.add(normalized_alias)

        for alias in all_aliases:
            if alias in alias_map and alias_map[alias] != canonical:
                raise ValueError(
                    f"Alias collision detected: '{alias}' is mapped to both "
                    f"'{alias_map[alias]}' and '{canonical}'"
                )
            alias_map[alias] = canonical

    by_category_map: dict[SkillCategory, tuple[str, ...]] = {
        cat: tuple(sorted(skills)) for cat, skills in by_category_temp.items()
    }

    return taxonomy_map, alias_map, category_map, by_category_map


(
    _SKILL_TAXONOMY_DICT,
    _ALIAS_TO_CANONICAL_DICT,
    _SKILL_TO_CATEGORY_DICT,
    _SKILLS_BY_CATEGORY_DICT,
) = _build_taxonomy()

# Public immutable mappings
SKILL_TAXONOMY: Final[MappingProxyType[str, SkillDefinition]] = MappingProxyType(
    _SKILL_TAXONOMY_DICT
)
ALIAS_TO_CANONICAL: Final[MappingProxyType[str, str]] = MappingProxyType(_ALIAS_TO_CANONICAL_DICT)
SKILL_TO_CATEGORY: Final[MappingProxyType[str, SkillCategory]] = MappingProxyType(
    _SKILL_TO_CATEGORY_DICT
)
SKILLS_BY_CATEGORY: Final[MappingProxyType[SkillCategory, tuple[str, ...]]] = MappingProxyType(
    _SKILLS_BY_CATEGORY_DICT
)
CANONICAL_SKILLS: Final[frozenset[str]] = frozenset(_SKILL_TAXONOMY_DICT.keys())


def get_canonical_skill(raw_skill: str | None) -> str | None:
    """Resolve raw skill string or alias to canonical skill name.

    Args:
        raw_skill: Raw skill name or alias in any case.

    Returns:
        Canonical skill name, or None if no match is found.
    """
    if raw_skill is None or not isinstance(raw_skill, str):
        return None
    cleaned = raw_skill.strip().lower()
    return ALIAS_TO_CANONICAL.get(cleaned)


def get_skill_category(skill_name: str | None) -> SkillCategory | None:
    """Retrieve canonical category for a given skill or alias.

    Args:
        skill_name: Canonical skill name or known alias.

    Returns:
        SkillCategory if found, otherwise None.
    """
    canonical = get_canonical_skill(skill_name) if skill_name not in SKILL_TO_CATEGORY else skill_name
    if canonical is None:
        return None
    return SKILL_TO_CATEGORY.get(canonical)
