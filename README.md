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
- **Пайплайны** — последовательности шагов (агенты + инструменты), выполняемые через Celery.
- **UI командного центра** — проекты, агенты, чат, память, MCP и пайплайны + новый блок управления доступами к инструментам и быстрые действия для Filesystem MCP (архивы, JSON/YAML и т.д.).
- **LLM Registry** — централизованный реестр доступных LLM‑моделей, синхронизируемый с OpenAI.

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
- LLM Registry:
- `GET  /api/models/registry/` — текущее состояние реестра моделей
- `POST /api/models/registry/sync/` — ручной запуск синхронизации (только для админов)

## Быстрые действия Filesystem MCP

В веб‑интерфейсе (панель справа) появился отдельный блок для Filesystem MCP:

- переключатели доступа к MCP‑инструментам по каждому серверу;
- формы для «быстрых действий»: `fs_zip`, `fs_unzip`, `fs_read_json`, `fs_write_json` (список расширяется по мере добавления инструментов);
- журнал выполненных команд (все ответы MCP попадают в лог, ошибки подсвечены красным);
- панель автоматически скрывается, если у выбранного агента нет доступа к Filesystem MCP.

## Smoke‑тесты Filesystem MCP

Для проверки MCP‑инструментов без UI используется минимальный smoke‑набор `tests/test_smoke.py` в проекте `filesystem-mcp`. Он проверяет initialize → tools.list → запись/чтение файла → zip/unzip → JSON read/write.

```bash
cd /Users/paskalex/Work/mcp/filesystem-mcp
MCP_BASE_URL=http://localhost:8020/mcp python3 -m unittest tests.test_smoke
```

Перед запуском убедитесь, что контейнер Filesystem MCP поднят (`docker-compose up -d` в `/Users/paskalex/Work/mcp/filesystem-mcp`).

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
