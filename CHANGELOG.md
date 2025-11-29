# Changelog

Все значимые изменения в этом проекте будут документироваться в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

## [0.3.0] — 2025-11-29

### Добавлено

- **Graph Memory v1**: модели `KnowledgeNode`/`KnowledgeEdge`, Graph Extractor на базе LLM, автоматическое извлечение графа из MemoryEvent и UI-доступ к узлам/связям (с фильтром pinned).
- **RAG слой**: сервис `build_rag_memory_block` ищет top-K `KnowledgeEmbedding`, формирует блок `AGENT RAG DOCUMENTS` и добавляет `rag_recall` в `tool_traces`.
- **Tool traces** теперь фиксируют три типа (`memory_recall`, `graph_recall`, `rag_recall`) и отображаются в Assistant Chat и делегировании.
- **Автотесты**: unit-наборы для Graph Extractor, Graph Memory retrieval и Graph Cleanup (`python manage.py test core.tests.Graph*`).

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
