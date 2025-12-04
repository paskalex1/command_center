from __future__ import annotations

import difflib
import logging
from typing import List, Tuple

from django.conf import settings
from django.utils import timezone

from core.constants import RAG_LIBRARIAN_SLUG
from core.graph_extractor import extract_graph_from_text
from core.models import (
    Agent,
    KnowledgeChangeLog,
    KnowledgeSource,
    KnowledgeSourceVersion,
)

logger = logging.getLogger(__name__)

TEXT_MIME_PREFIXES = ("text/", "application/json", "application/xml")
DEFAULT_SNAPSHOT_LIMIT = 512 * 1024  # 512 KB


def _can_snapshot(mime_type: str, text_bytes: int) -> bool:
    if not mime_type:
        return True
    if not mime_type.startswith(TEXT_MIME_PREFIXES):
        return False
    limit = getattr(settings, "RAG_SNAPSHOT_MAX_BYTES", DEFAULT_SNAPSHOT_LIMIT)
    return text_bytes <= limit


def _safe_head(text: str, limit: int = 512) -> str:
    if not text:
        return ""
    return text[:limit]


def _get_version_text(version: KnowledgeSourceVersion | None, fallback: str = "") -> str:
    if version is None:
        return fallback
    if version.full_text:
        return version.full_text
    return fallback


def create_version_snapshot(
    source: KnowledgeSource,
    text: str,
) -> KnowledgeSourceVersion:
    encoded = text.encode("utf-8", errors="ignore")
    snapshot_allowed = _can_snapshot(source.mime_type or "", len(encoded))
    version = KnowledgeSourceVersion.objects.create(
        source=source,
        project=source.project,
        content_hash=source.content_hash,
        mime_type=source.mime_type or "",
        size_bytes=len(encoded),
        text_head=_safe_head(text),
        full_text=text if snapshot_allowed else "",
    )
    return version


def _lines(text: str) -> List[str]:
    return (text or "").splitlines()


def compute_diff(previous: str, current: str) -> str:
    diff = difflib.unified_diff(
        _lines(previous),
        _lines(current),
        fromfile="previous",
        tofile="current",
        lineterm="",
    )
    return "\n".join(diff)


def analyze_semantic_changes(diff_text: str) -> Tuple[str, List[str], List[str]]:
    added: List[str] = []
    removed: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            value = line[1:].strip()
            if value:
                added.append(value)
        elif line.startswith("-"):
            value = line[1:].strip()
            if value:
                removed.append(value)
    summary_parts = []
    if added:
        summary_parts.append(f"Добавлено строк: {len(added)}")
    if removed:
        summary_parts.append(f"Удалено строк: {len(removed)}")
    summary = "; ".join(summary_parts) if summary_parts else "Изменения не детализированы."
    return summary, added[:10], removed[:10]


def update_graph_with_facts(project_slug: str, facts: List[str]) -> None:
    if not facts:
        return
    agent = Agent.objects.filter(slug=RAG_LIBRARIAN_SLUG).first()
    if not agent:
        return
    text = "\n".join(facts)
    extract_graph_from_text(agent, f"Новые факты для проекта {project_slug}:\n{text}")


def record_changelog_entry(
    *,
    source: KnowledgeSource,
    previous_version: KnowledgeSourceVersion | None,
    new_version: KnowledgeSourceVersion,
    current_text: str,
) -> KnowledgeChangeLog:
    previous_text = _get_version_text(previous_version)
    diff_text = compute_diff(previous_text, current_text)
    summary, new_facts, removed_facts = analyze_semantic_changes(diff_text)

    if previous_version is None:
        change_type = KnowledgeChangeLog.TYPE_ADDED
    elif not current_text.strip():
        change_type = KnowledgeChangeLog.TYPE_REMOVED
    else:
        change_type = KnowledgeChangeLog.TYPE_MODIFIED

    entry = KnowledgeChangeLog.objects.create(
        project=source.project,
        source=source,
        previous_version=previous_version,
        version=new_version,
        change_type=change_type,
        diff_text=diff_text,
        semantic_summary=summary,
        new_facts=new_facts,
        removed_facts=removed_facts,
    )

    if new_facts:
        update_graph_with_facts(source.project.slug, new_facts)
    return entry
