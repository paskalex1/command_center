from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

from core.embeddings import embed_text
from core.models import KnowledgeChunk, KnowledgeEmbedding, KnowledgeSource, KnowledgeSourceVersion
from .rag_status import refresh_project_rag_stats
from .rag_versions import create_version_snapshot, record_changelog_entry
from .knowledge_extraction import schedule_knowledge_extractor

logger = logging.getLogger(__name__)


def extract_text_from_source(source: KnowledgeSource) -> str:
    """
    Извлекает текст из файла в docs/<slug>.
    Поддерживает .md/.markdown/.txt/.pdf (остальные пробует прочитать как текст).
    """

    file_path = Path(source.project.docs_path) / source.path
    if not file_path.exists():
        raise FileNotFoundError(f"KnowledgeSource file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf_text(file_path)
    if suffix == ".html":
        return _extract_html_text(file_path)

    # fallback — читаем как текст
    return file_path.read_text(encoding="utf-8", errors="ignore")


def split_text_into_chunks(
    text: str,
    max_tokens: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Делит текст на перекрывающиеся чанки.

    Т.к. точного токенизатора под рукой нет, считаем 1 токен ~= 4 символа.
    """

    if not text:
        return []

    max_chars = max(max_tokens * 4, 200)
    overlap_chars = max(overlap * 4, 0)

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap_chars)

    return chunks


def index_source(source: KnowledgeSource) -> int:
    """
    Извлекает текст, нарезает на чанки и создаёт эмбеддинги.
    Возвращает количество созданных чанков.
    """

    try:
        text = extract_text_from_source(source)
        chunks = split_text_into_chunks(text)
    except Exception as exc:
        source.status = KnowledgeSource.STATUS_ERROR
        source.last_error = str(exc)
        source.save(update_fields=["status", "last_error"])
        refresh_project_rag_stats(source.project)
        raise

    previous_version = (
        KnowledgeSourceVersion.objects.filter(source=source).order_by("-created_at").first()
    )
    new_version = create_version_snapshot(source, text)
    changelog_entry = record_changelog_entry(
        source=source,
        previous_version=previous_version,
        new_version=new_version,
        current_text=text,
    )
    summary_parts: List[str] = []
    if changelog_entry.semantic_summary:
        summary_parts.append(changelog_entry.semantic_summary)
    if changelog_entry.new_facts:
        facts = "\n".join(f"- {fact}" for fact in changelog_entry.new_facts if fact)
        if facts:
            summary_parts.append("Новые факты:\n" + facts)
    if summary_parts:
        schedule_knowledge_extractor(
            project_id=source.project_id,
            text="\n\n".join(summary_parts),
            source=f"rag_diff:{source.path}",
        )

    KnowledgeChunk.objects.filter(source=source).delete()

    created = 0
    try:
        for idx, chunk_text in enumerate(chunks):
            chunk = KnowledgeChunk.objects.create(
                document=None,
                source=source,
                project=source.project,
                text=chunk_text,
                chunk_index=idx,
                meta={"path": source.path, "filename": source.filename},
            )
            embedding = embed_text(chunk_text)
            KnowledgeEmbedding.objects.create(
                chunk=chunk,
                source=source,
                embedding=embedding,
            )
            created += 1
    except Exception as exc:  # noqa: BLE001
        KnowledgeChunk.objects.filter(source=source).delete()
        source.status = KnowledgeSource.STATUS_ERROR
        source.last_error = str(exc)
        source.save(update_fields=["status", "last_error"])
        refresh_project_rag_stats(source.project)
        raise

    source.status = KnowledgeSource.STATUS_PROCESSED
    source.last_error = ""
    source.save(update_fields=["status", "last_error"])
    refresh_project_rag_stats(source.project)
    return created


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page_num, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read PDF page %s (%s): %s", path, page_num, exc)
    return "\n".join(parts).strip()


def _extract_html_text(path: Path) -> str:
    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: List[str] = []

        def handle_data(self, data: str) -> None:  # noqa: D401
            self.parts.append(data)

    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(part.strip() for part in parser.parts if part.strip())
