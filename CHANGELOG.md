# Changelog

Все значимые изменения в этом проекте будут документироваться в этом файле.

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

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

