# Changelog

Все значимые изменения в этом проекте будут документироваться в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

## [0.4.0] — 2025-12-02

### Добавлено

- **Новый чат-интерфейс**: макет в стиле ChatGPT (одна левая колонка с агентами и LLM Registry, центр — чат, справа — MCP/пайплайны). Tool traces и память отображаются компактными плашками с модалкой по клику.
- **Хранение переписок**: Conversation/Message перепривязаны к проекту+агенту, UI подхватывает последнюю сессию, можно сбросить кнопкой «Новый диалог». Добавлены REST‑эндпоинты `/assistant/conversation/`.
- **Вложения**: `MessageAttachment`, multipart‑отправка файлов и изображений, предпросмотр в чате и хранение в `media/chat_attachments/%Y/%m/%d/`.
- **Knowledge Extractor**: миграция `0030_create_knowledge_extractor`, Celery‑таска `run_knowledge_extractor_for_text`, сервисы `knowledge_extraction` и `graph_ingest`, unit‑тесты.
- **Graph overview API**: `GET /api/projects/<slug>/graph/overview/` агрегирует количество узлов/связей, топ‑концепты и timestamp для UI/интеграций.
- **RAG версии и changelog**: модели `KnowledgeSourceVersion` + `KnowledgeChangeLog`, unified diff, semantic summary (добавленные/удалённые факты), вывод в UI, REST `/rag/changelog/`.
- **RAG мониторинг**: поля `rag_last_full_sync_at`, `rag_last_error_at`, `rag_error_count` в `Project`, Celery‑таска `sync_all_projects_rag`, баннеры/кнопки в UI, автозапуск RAG Librarian.
- **Страница «Память»**: отдельный шаблон `templates/core/memory.html` с AgentMemory, Graph Memory (включая overview), RAG Librarian панелью и журналом MemoryEvent.
- **Команды экспорта/импорта**: `export_cc_config` и `import_cc_config` для проектов, агентов, MCP‑серверов и доступов.
- **Доп. REST и тесты**: `/api/projects/<slug>/rag/sources/`, `/rag/ingest/`, `/rag/changelog/` + тестовые наборы `core.tests_api_rag`, `core.tests_rag_sync`, `core.tests_graph_*`, `core.tests_agents_rag`, `core.tests_rag_versions`.

### Изменено

- AssistantChatView теперь понимает `conversation_id`, multipart‑формы с `attachments`, и автоматически переключается на Responses API для кодовых моделей (через `requires_responses_api`).
- Улучшен LLM Registry (`gpt‑5.1‑codex` и другие code‑модели попадают и в chat‑категорию, lightweight‑модели определяются по имени, UI запускает синк из левой панели).
- RAG Librarian использует только REST (`/rag/sources/`, `/rag/ingest/`, `/rag/changelog/`), выдаёт статусы, может инициировать `reindex_all` и подсвечивает проблемные документы.
- Knowledge Extractor запускается автоматически после semantic diff и «knowledge-rich» ответов, обновляя Graph Memory без участия пользователя.
- MCP/UI: перечень серверов отображается аккордеоном, Filesystem MCP быстрые действия убраны из правого нижнего блока, количество панелей снижено до одной колонки слева, а память/документация перенесены в новую вкладку.
- README дополнен разделами про Project docs, вложения, Memory view, Knowledge Extractor, RAG changelog и экспорт/импорт конфигураций.

## [0.3.0] — 2025-11-29

### Добавлено

- **Graph Memory v1**: модели `KnowledgeNode`/`KnowledgeEdge`, Graph Extractor на базе LLM, автоматическое извлечение графа из MemoryEvent и UI-доступ к узлам/связям (с фильтром pinned).
- **RAG слой**: сервис `build_rag_memory_block` ищет top-K `KnowledgeEmbedding`, формирует блок `AGENT RAG DOCUMENTS` и добавляет `rag_recall` в `tool_traces`.
- **Tool traces** теперь фиксируют три типа (`memory_recall`, `graph_recall`, `rag_recall`) и отображаются в Assistant Chat и делегировании.
- **Автотесты**: unit-наборы для Graph Extractor, Graph Memory retrieval и Graph Cleanup (`python manage.py test core.tests.Graph*`).
- **Project docs**: `PROJECT_DOCS_ROOT` (`/docs/`), `Project.slug`/`docs_path`, сигнал создает `docs/<slug>/README.md` при создании проекта и management-команда `generate_project_docs` покрывает существующие проекты.

### Изменено

- `build_agent_llm_messages` возвращает пять артефактов (messages + memory + graph + rag) и строит контекст в порядке SYSTEM → MEMORY → GRAPH → RAG → HISTORY → USER.
- AssistantChatView и делегирование используют `_prepend_recall_traces`, чтобы всегда добавлять summary по памяти/графу/RAG в `tool_traces`; метаданные ответов содержат `graph_nodes` и `rag_documents`.
- README обновлён: описаны Graph/RAG слои, убраны локальные пути, уточнены инструкции по запуску и Filesystem MCP actions.
- `cleanup_graph_memory` стал usage-осмысленным: защищённые узлы (`is_pinned`) не удаляются, висячие и неиспользованные чистятся первыми, заведены `usage_count` и `last_used_at`, лимиты управляются через настройки.

## [0.2.0] — 2025-11-27

### Добавлено

- Режимы использования инструментов для агентов (`tool_mode=auto|required`) с миграциями, настройками в админке и проверками в `AssistantChatView`.
- Делегирование задач между агентами: генерация `delegate_to_agent_*` tools, повторная инициализация MCP-сессий и лог `tool_traces` в сообщениях.
- REST API `GET/POST /api/agents/<id>/mcp-access/` и новый блок в UI для управления доступами агента к MCP-инструментам.
- Панель «Filesystem MCP — быстрые действия» (архивирование, JSON/YAML) и лог действий прямо в веб-интерфейсе.
- Smoke‑тесты Filesystem MCP (`tests/test_smoke.py`) — запускают initialize→tools.list и проверяют запись/чтение файлов, zip/unzip, JSON I/O.

### Изменено

- Assistant Chat теперь принудительно использует инструменты, хранит `tool_traces` и выводит в UI историю всех вызовов.
- UI подчёркивает новые состояния агентов, отображает tool traces и автоматически скрывает панели без доступа к соответствующим MCP-серверам.

## [0.1.0] — Initial commit

### Добавлено

- Базовая инфраструктура Django + DRF + Docker + Celery + Redis + PostgreSQL (pgvector).
- Модели и API для:
  - проектов и базы знаний (KnowledgeBase, KnowledgeDocument, KnowledgeChunk, KnowledgeEmbedding);
  - MCP‑серверов и инструментов (MCPServer, MCPTool);
  - агентов и привязок к MCP (Agent, AgentServerBinding);
  - диалогов и памяти (Conversation, Message, MemoryEvent, MemoryEmbedding);
  - пайплайнов агентов (Pipeline, PipelineStep, Task).
- Семантический поиск по базе знаний.
- Личный ассистент проекта с долговременной памятью.
- Исполнение пайплайнов через Celery.
- LLM Registry:
  - `models_registry.json` + команда `update_llm_registry`;
  - сервис `sync_models_registry()` и периодический синк через Celery beat;
  - API `/api/models/registry/` и `/api/models/registry/sync/`.
- Минимальный веб‑интерфейс:
  - селектор проекта;
  - список агентов;
  - чат с выбранным агентом;
  - панель MCP‑серверов, инструментов, пайплайнов и памяти;
  - блок LLM Registry с текущими моделями и кнопкой “Sync from OpenAI”.
