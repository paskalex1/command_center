from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Iterable, List, Tuple

from core.models import KnowledgeSource, Project

ALLOWED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".html"}


def scan_project_docs(project: Project) -> List[KnowledgeSource]:
    """
    Сканирует docs/<project.slug>/ и регистрирует файлы как KnowledgeSource.
    """

    root = Path(project.docs_path)
    if not root.exists():
        return []

    sources: List[KnowledgeSource] = []
    for file_path, rel_path in _iter_project_files(root):
        content = file_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        mime_type = _guess_mime_type(file_path)
        rel_str = rel_path.as_posix()

        source, created = KnowledgeSource.objects.get_or_create(
            project=project,
            path=rel_str,
            defaults={
                "filename": file_path.name,
                "content_hash": content_hash,
                "mime_type": mime_type,
                "status": KnowledgeSource.STATUS_NEW,
            },
        )

        updates: dict[str, object] = {}
        if created:
            source.last_error = ""
        else:
            if source.content_hash != content_hash:
                updates["content_hash"] = content_hash
                updates["status"] = KnowledgeSource.STATUS_NEW
                updates["last_error"] = ""
            if source.filename != file_path.name:
                updates["filename"] = file_path.name
            if source.mime_type != mime_type:
                updates["mime_type"] = mime_type

        if updates:
            for field, value in updates.items():
                setattr(source, field, value)
            source.save(update_fields=list(updates.keys()))

        sources.append(source)

    return sources


def _iter_project_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(root)
        if _is_hidden(rel_path):
            continue
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        yield file_path, rel_path


def _is_hidden(rel_path: Path) -> bool:
    return any(part.startswith(".") for part in rel_path.parts)


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"
