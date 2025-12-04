# Command Center — локальный AI командный центр

`command_center` — это локальный хаб для управления AI‑агентами, пайплайнами и базой знаний.  
Проект разворачивается в Docker и использует OpenAI API, PostgreSQL (с pgvector), Redis и Celery.

## Основные возможности

- **Проекты** — логические рабочие пространства.
- **База знаний** — загрузка документов (PDF/MD/TXT), разбиение на фрагменты, эмбеддинги и семантический поиск.
- **MCP‑серверы и инструменты** — регистрация MCP‑серверов и вызов инструментов через Django API.
- **Агенты** — OpenAI‑агенты с привязкой к MCP‑инструментам, режимами работы (`auto/required`) и выбором модели из LLM Registry.
- **Делегирование** — агенты могут вызывать друг друга через generated tools (`delegate_to_agent_*`).
- **Личный ассистент** — диалоги, долговременная память, поиск по MemoryEvent и `tool_traces` для каждого ответа.
- **Многоуровневая память** — AGENT MEMORY (AgentMemory) + AGENT GRAPH MEMORY (KnowledgeNode/Edge) + AGENT RAG DOCUMENTS (KnowledgeEmbedding) автоматически добавляются в системный контекст каждого запроса.
- **Пайплайны** — последовательности шагов (агенты + инструменты), выполняемые через Celery.
- **UI командного центра** — проекты, агенты, чат, память, MCP и пайплайны + новый блок управления доступами к инструментам и быстрые действия для Filesystem MCP (архивы, JSON/YAML и т.д.).
- **LLM Registry** — централизованный реестр доступных LLM‑моделей, синхронизируемый с OpenAI.
- **История диалогов и вложения** — чат в стиле ChatGPT с хранением переписки, предпросмотром tool traces и поддержкой изображений/документов.
- **Project docs** — при создании проекта автоматически создаётся `docs/<slug>/README.md`, и уже существующие проекты можно покрыть командой `generate_project_docs`.
- **Knowledge Extractor** — скрытый агент и Celery‑таски, автоматически извлекающие факты/сущности из семантических диффов и «информативных» ответов для пополнения Graph Memory.

## Память и знания агента

Каждый ответ агента автоматически строится на основе трёх слоёв знаний:

1. **AGENT MEMORY** — релевантные записи из `AgentMemory` (факты/предпочтения).
2. **AGENT GRAPH MEMORY** — ближайшие узлы графа (`KnowledgeNode/Edge`) + их связи.
3. **AGENT RAG DOCUMENTS** — top‑K фрагментов внешней документации (`KnowledgeEmbedding`).

Эти блоки попадают в системный контекст перед историей диалога, а их содержимое отображается в `tool_traces` как `memory_recall`, `graph_recall`, `rag_recall`.

## Документация проектов

- `PROJECT_DOCS_ROOT` (`/docs/`) хранит каталоги каждого проекта.
- После сохранения `Project` сигнал создаёт папку `docs/<slug>/` и записывает `README.md` с названием, описанием и ссылкой на структуру.
- Чтобы создать документацию для существующих проектов, выполните:

  ```bash
  docker-compose exec web python manage.py generate_project_docs
  ```

## Диалоговый интерфейс и вложения

- Каждый проект × агент теперь имеет **отдельный Conversation**. Повторное открытие агента подхватывает последнюю беседу, а кнопка «Новый диалог» создаёт свежую сессию.
- `POST /api/projects/<id>/assistant/chat/` принимает `conversation_id`, а также мульти‑часть с полем `attachments` — туда можно загрузить изображения (PNG/JPEG) и произвольные файлы. Файлы сохраняются в `media/chat_attachments/<date>/` и показываются с превью прямо в чате.
- `GET/POST /api/projects/<id>/assistant/conversation/` — API для получения или сброса текущей беседы (используется UI).
- Tool traces, память и RAG‑вставки выводятся компактными «плашками» под ответом. По клику открывается модальное окно с полным JSON.
- Левая колонка содержит плоский список агентов + блок LLM Registry (синхронизация делается из UI). Правая панель показывает MCP‑серверы и пайплайны проекта, а сам чат визуально и по UX ориентирован на ChatGPT.

## Отдельная страница «Память»

- В верхнем меню появились вкладки **«Чат»** и **«Память»**. Страница памяти показывает:
  - журнал AgentMemory (с кнопкой обновления);
  - Graph Memory с overview (qty узлов/связей, топ‑концепты, свежие отношения) и переключателем pinned;
  - RAG панель — статус последнего синка, ошибки, changelog файлов, быстрый вызов RAG Librarian;
  - «Память проекта» — последние MemoryEvent.
- Данные подгружаются через новые REST‑эндпоинты (`/graph/overview/`, `/rag/changelog/` и т.д.) и могут обновляться без перезагрузки страницы.

## Knowledge Extractor и Graph Memory

- Миграцией `0030_create_knowledge_extractor` фиксируется агент `knowledge-extractor`. Он не отображается в UI, но вызывается фоном:
  - после любого Semantic Diff RAG (`KnowledgeChangeLog`);
  - после ответов агентов, удовлетворяющих `is_knowledge_rich_text`;
  - по ручным запросам пользователя («обнови граф», «извлеки знания...»).
- Сервис `core/services/knowledge_extraction.py` формирует JSON (`nodes` + `edges`) и передаёт в `graph_ingest`, который:
  - дедуплицирует узлы по `label`/`type`;
  - наращивает `usage_count` и `last_used_at`;
  - создаёт связи только при отсутствии дублей;
  - может прикреплять `object_type/object_id` и embedding.
- `GET /api/projects/<slug>/graph/overview/` — агрегированная статистика, которая питает UI и внешние интеграции.

## RAG мониторинг, diff и changelog

- `Project` хранит `rag_last_full_sync_at`, `rag_last_error_at`, `rag_error_count`. Фоновая задача `sync_all_projects_rag` (Celery beat, раз в сутки) автоматически ставит на переиндексацию «протухшие» проекты.
- `KnowledgeSourceVersion` и `KnowledgeChangeLog` фиксируют каждый синк: хэш, `text_head`, полный снапшот (для текстов ≤ 512КБ) и unified diff. Из diff извлекаются «новые/удалённые факты», которыми можно сразу обновлять Graph Memory.
- UI показывает changelog таблицу, баннер при ошибках и позволяет одним кликом «попросить RAG‑агента» пересобрать индекс.
- REST‑слой (`/api/projects/<slug>/rag/sources/`, `/rag/ingest/`, `/rag/changelog/`) используется как UI, так и RAG Librarian/другие агенты, обеспечивая единый контракт.

## Стек

- Backend: Django 5 + Django REST Framework
- База: PostgreSQL + pgvector
- Очередь/кэш: Redis
- Фоновая обработка: Celery
- LLM: OpenAI API
- Фронтенд: Django Templates + немного JS
- Контейнеры: Docker + docker‑compose

## Быстрый старт (через Docker)

1. Скопируйте `.env.example` в `.env` и заполните переменные:

   - `OPENAI_API_KEY` — ключ OpenAI
   - `POSTGRES_*`, `REDIS_*` — при необходимости
   - опционально `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS`

2. Соберите и запустите сервисы:

   ```bash
   docker-compose up --build
   ```

3. Примените миграции и создайте суперпользователя:

   ```bash
   docker-compose exec web python manage.py migrate
   docker-compose exec web python manage.py createsuperuser
   ```

4. Откройте:

   - Админку: `http://localhost:8000/admin/`
   - Командный центр (UI): `http://localhost:8000/`

## Основные API

- Healthcheck: `GET /api/health`
- Проекты и база знаний:
  - `POST /api/projects/`
  - `POST /api/projects/{id}/knowledge-bases/`
  - `POST /api/projects/{id}/documents/`
  - `GET  /api/projects/{id}/search/?q=...`
- MCP:
  - `GET/POST /api/mcp/servers/`
  - `GET      /api/mcp/servers/{id}/tools/`
  - `POST     /api/mcp/servers/{id}/sync-tools/`
  - `POST     /api/mcp/tools/{tool_id}/invoke/`
- Агенты:
  - `GET/POST   /api/agents/`
  - `GET/PATCH  /api/agents/{id}/`
  - `POST       /api/agents/{id}/invoke/`
  - `GET/POST   /api/agents/{id}/mcp-access/`
- Ассистент проекта:
  - `POST /api/projects/{project_id}/assistant/chat/`
  - `GET  /api/conversations/{id}/`
- Пайплайны:
  - `GET/POST   /api/pipelines/`
  - `GET/PATCH  /api/pipelines/{id}/`
- `POST       /api/pipelines/{id}/run/`
- `GET        /api/tasks/{id}/`
- Память и диалоги:
  - `GET/POST /api/projects/{project_id}/assistant/conversation/`
  - `GET      /api/conversations/{id}/`
- Graph Memory:
  - `GET /api/projects/{slug}/graph/overview/`
- RAG:
  - `GET  /api/projects/{slug}/rag/sources/`
  - `POST /api/projects/{slug}/rag/ingest/`
  - `GET  /api/projects/{slug}/rag/changelog/`
- LLM Registry:
- `GET  /api/models/registry/` — текущее состояние реестра моделей
- `POST /api/models/registry/sync/` — ручной запуск синхронизации (только для админов)

## Быстрые действия Filesystem MCP

В веб-интерфейсе (панель справа) есть отдельный блок для Filesystem MCP:

- переключатели доступа к MCP‑инструментам по каждому серверу;
- формы для «быстрых действий» (`fs_zip`, `fs_unzip`, `fs_read_json`, `fs_write_json`, …);
- журнал выполненных команд (ответы MCP попадают в лог, ошибки подсвечены красным);
- панель автоматически скрывается, если у выбранного агента нет доступа к Filesystem MCP.

## Smoke‑тесты Filesystem MCP

Для проверки MCP‑инструментов без UI предусмотрен минимальный smoke‑набор `tests/test_smoke.py` в проекте `filesystem-mcp`. Он проверяет initialize → tools.list → запись/чтение файла → zip/unzip → JSON read/write.

```bash
cd filesystem-mcp
MCP_BASE_URL=http://localhost:8020/mcp python3 -m unittest tests.test_smoke
```

Перед запуском убедитесь, что контейнер Filesystem MCP поднят (`docker-compose up -d` внутри каталога `filesystem-mcp`).

## Резервное копирование

В репозитории есть скрипт `scripts/backup_command_center.sh`, который сохраняет:

- дамп PostgreSQL (`postgres.sql`, через `pg_dump` внутри контейнера `postgres`);
- ключевые конфигурации (.env, docker-compose, requirements) и пользовательские данные (`docs/`, `media/`, `command_center/config/`);
- служебный файл `backup_info.txt` с метаданными бэкапа.

Запуск из корня проекта:

```bash
./scripts/backup_command_center.sh
```

По умолчанию архивы попадают в `backups/command_center_YYYYmmdd_HHMMSS.tar.gz`. Можно переопределить каталог и env-файл:

```bash
BACKUP_DIR=/mnt/backups ENV_FILE=/path/to/.env ./scripts/backup_command_center.sh
```

Восстановление:

1. Распаковать архив (`tar -xzf command_center_*.tar.gz`).
2. Вернуть нужные файлы/директории из каталога `files/` (например `.env`, `docs`, `media`).
3. Загрузить БД:
   ```bash
  docker compose exec -T postgres bash -c "PGPASSWORD='$POSTGRES_PASSWORD' psql -U '$POSTGRES_USER' '$POSTGRES_DB'" < postgres.sql
  ```

Перед запуском убедитесь, что переменные `POSTGRES_*` совпадают с теми, что использовались при создании бэкапа.

## Экспорт / импорт конфигурации

- `docker-compose exec web python manage.py export_cc_config export.json` — сохраняет проекты, MCP‑серверы, агентов и доступы (включая привязанные инструменты) в один JSON.
- `docker-compose exec web python manage.py import_cc_config export.json` — восстанавливает те же сущности, синхронизирует инструменты серверов и заново генерирует `docs/<slug>/README.md`.
- Команды полезны для миграции окружений и репликации настроек.

### Агент “RAG Librarian”

- При миграции `0026_create_rag_librarian` автоматически создаётся глобальный агент **RAG Librarian** (`slug=rag-librarian`), который отвечает за сканирование `/docs/<project_slug>/` и переиндексацию RAG.
- Все primary-агенты автоматически получают его в делегаты (сигнал `post_save` следит за новыми агентами); при запросах, связанных с документацией, основной ассистент обязан вызывать инструмент `delegate_to_agent_<id>` и передавать задачу библиотекарю.
- Если нужен кастомный вариант для конкретного проекта — можно клонировать агента в админке и указать `project`, система всё равно подключит его как делегата и добавит инструкцию в системный промпт.

## LLM Registry

Реестр моделей хранится в `command_center/config/models_registry.json` и:

- обновляется через management‑команду:

  ```bash
  docker-compose exec web python manage.py update_llm_registry
  ```

- синхронизируется по расписанию (Celery beat, 1‑е число месяца в 03:00);
- может быть обновлён вручную из UI кнопкой “Sync from OpenAI”.

Агенты используют:

- `Agent.model_name` — явный выбор модели (может быть пустым);
- `Agent.resolved_model_name` — фактическая модель (явная или `chat.primary` из реестра).

## Структура проекта

- `command_center/` — Django‑проект, настройки, Celery, LLM‑реестр, сервисы.
- `core/` — основное приложение (модели, API, задачи, админка).
- `templates/` — шаблоны Django (UI командного центра).
- `docker/`, `Dockerfile`, `docker-compose.yml` — контейнеризация.
- `COMMAND_CENTER_PLAN.md` — детализация спринтов и архитектуры (игнорируется git по умолчанию).
- `CHECKLISTS.md` — чек‑листы по спринтам (также игнорируется).

## Разработка без Docker (опционально)

1. Установить зависимости:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Настроить `.env` и PostgreSQL, затем:

   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

## Лицензия

Проект создаётся как внутренний инструмент; лицензию можно добавить позже при необходимости.
