from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


class RAGAgentError(RuntimeError):
    pass


def _base_url() -> str:
    return getattr(settings, "RAG_API_BASE_URL", os.getenv("RAG_API_BASE_URL", "http://localhost:8000")).rstrip("/")


def _request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{_base_url()}{path}"
    try:
        response = requests.request(method, url, timeout=10, **kwargs)
    except requests.RequestException as exc:  # noqa: PERF203
        logger.error("RAG API request failed: %s %s (%s)", method, url, exc)
        raise RAGAgentError(f"Unable to reach RAG API: {exc}") from exc
    if response.status_code >= 400:
        logger.warning("RAG API returned %s %s -> %s %s", method, url, response.status_code, response.text)
        raise RAGAgentError(f"RAG API error ({response.status_code}): {response.text}")
    return response


def _parse_paginated(data: Any) -> dict:
    if isinstance(data, list):
        return {"count": len(data), "results": data, "project_meta": None}
    if isinstance(data, dict):
        results = data.get("results")
        if results is None:
            return {"count": len(data), "results": data, "project_meta": data.get("project_meta")}
        return {
            "count": data.get("count", len(results)),
            "results": results,
            "project_meta": data.get("project_meta"),
        }
    return {"count": 0, "results": [], "project_meta": None}


def get_sources(project_slug: str, status: str | None = None) -> dict:
    params = {}
    if status:
        params["status"] = status
    response = _request("GET", f"/api/projects/{project_slug}/rag/sources/", params=params)
    return _parse_paginated(response.json())


def reindex_all(project_slug: str) -> str:
    response = _request(
        "POST",
        f"/api/projects/{project_slug}/rag/ingest/",
        json={"mode": "all"},
    )
    data = response.json()
    task_id = data.get("task_id")
    return f"Индексация запущена (task_id={task_id})." if task_id else "Индексация поставлена в очередь."


def reindex_single(project_slug: str, source_id: str) -> str:
    response = _request(
        "POST",
        f"/api/projects/{project_slug}/rag/ingest/",
        json={"mode": "single", "source_id": source_id},
    )
    data = response.json()
    task_id = data.get("task_id")
    return f"Переиндексация файла запущена (task_id={task_id})." if task_id else "Переиндексация файла поставлена в очередь."


def summarize_sources(project_slug: str) -> str:
    payload = get_sources(project_slug)
    results: List[Dict[str, Any]] = payload["results"]
    if not results:
        return f"В проекте {project_slug} нет зарегистрированных источников."
    meta = payload.get("project_meta") or {}

    counters: Dict[str, int] = {}
    for item in results:
        counters[item.get("status", "unknown")] = counters.get(item.get("status", "unknown"), 0) + 1

    lines = [
        f"📚 Документация проекта {project_slug}",
        f"Файлов: {payload['count']}",
        f"Последний полный синк: {meta.get('rag_last_full_sync_at') or '—'}",
        f"Ошибок: {meta.get('rag_error_count', 0)}",
        "Статусы:",
    ]
    for status, value in counters.items():
        lines.append(f" • {status}: {value}")

    preview = results[:5]
    lines.append("")
    lines.append("Примеры файлов:")
    for idx, item in enumerate(preview, start=1):
        lines.append(f"{idx}) {item.get('relative_path') or item.get('path')} — {item.get('status')}")

    if payload["count"] > len(preview):
        lines.append("…")
    return "\n".join(lines)


def get_error_sources(project_slug: str) -> list[dict]:
    payload = get_sources(project_slug, status="error")
    return payload["results"]


def find_source_by_hint(project_slug: str, hint: str) -> dict | None:
    payload = get_sources(project_slug)
    results = payload["results"]
    match_uuid = UUID_RE.search(hint or "")
    if match_uuid:
        uuid_value = match_uuid.group(0)
        for item in results:
            if str(item.get("id")) == uuid_value:
                return item
    hint_lower = (hint or "").lower()
    for item in results:
        path = (item.get("relative_path") or item.get("path") or "").lower()
        if path and path in hint_lower:
            return item
    return None


def get_changelog(project_slug: str, limit: int = 5) -> list[dict]:
    response = _request(
        "GET",
        f"/api/projects/{project_slug}/rag/changelog/",
        params={"page_size": limit},
    )
    data = response.json()
    if isinstance(data, list):
        return data
    return data.get("results", [])
