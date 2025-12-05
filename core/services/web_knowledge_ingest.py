from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from django.utils import timezone
from django.utils.text import slugify

from core.models import Project


def persist_web_knowledge_documents(
    project: Project,
    payload: Dict[str, Any],
    query: str | None = None,
    subdir: str = "web_knowledge",
) -> List[Dict[str, Any]]:
    """
    Сохраняет документы, возвращённые Web Knowledge MCP, в docs/<project>/subdir.

    Возвращает список метаданных по сохранённым файлам вида:
    [{"title": ..., "source_url": ..., "relative_path": "...", "size_bytes": 1234}, ...]
    """

    documents = payload.get("documents") or []
    if not documents:
        return []

    root_dir = Path(project.docs_path)
    target_dir = root_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    collected_at = timezone.now()
    query_text = query or payload.get("query")

    for index, document in enumerate(documents, start=1):
        content = (document.get("markdown") or document.get("plain_text") or "").strip()
        if not content:
            continue

        source_url = document.get("source_url") or ""
        title = (
            document.get("title")
            or document.get("metadata", {}).get("title")
            or f"Web Document {index}"
        )
        slug = slugify(title) or f"doc-{index}"
        hash_suffix = (document.get("hash") or "")[:8]

        timestamp = collected_at.strftime("%Y%m%d_%H%M%S")
        filename_parts = [timestamp, slug[:50]]
        if hash_suffix:
            filename_parts.append(hash_suffix)
        filename = "_".join(filter(None, filename_parts)) + ".md"
        target_path = target_dir / filename
        counter = 1
        while target_path.exists():
            target_path = target_dir / f"{target_path.stem}_{counter}{target_path.suffix}"
            counter += 1

        header_lines = [
            f"# {title}",
            "",
        ]
        meta_pairs = [
            ("Source", source_url),
            ("Query", query_text),
            ("Collected", collected_at.strftime("%Y-%m-%d %H:%M")),
            ("Language", document.get("language")),
            ("Tags", ", ".join(document.get("tags") or [])),
        ]
        for label, value in meta_pairs:
            if value:
                header_lines.append(f"- **{label}:** {value}")

        extra_meta = document.get("metadata") or {}
        if extra_meta:
            header_lines.append("- **Metadata:**")
            for key, value in sorted(extra_meta.items()):
                serialized = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                header_lines.append(f"  - {key}: {serialized}")

        header_lines.append("")
        header_lines.append(content)
        header_lines.append("")

        target_path.write_text("\n".join(header_lines), encoding="utf-8")

        saved.append(
            {
                "title": title,
                "source_url": source_url,
                "relative_path": target_path.relative_to(root_dir).as_posix(),
                "size_bytes": target_path.stat().st_size,
            }
        )

    return saved
