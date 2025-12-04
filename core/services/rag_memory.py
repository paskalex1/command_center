import logging
from pathlib import Path
from typing import List, Tuple

from django.db.models import QuerySet
from pgvector.django import CosineDistance

from ..embeddings import embed_text
from ..models import Agent, KnowledgeEmbedding

logger = logging.getLogger(__name__)


def _short_file_name(file_field) -> str:
    if not file_field:
        return ""
    try:
        name = Path(file_field.name or "").name
        return name or ""
    except Exception:  # pragma: no cover
        return ""


def build_rag_memory_block(
    agent: Agent,
    query_text: str,
    max_docs: int = 5,
) -> Tuple[str, List[dict]]:
    """
    Возвращает текстовый блок AGENT RAG DOCUMENTS и метаданные о найденных фрагментах.
    """

    query = (query_text or "").strip()
    if not query:
        return "", []

    try:
        query_embedding = embed_text(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to embed query for RAG memory: %s", exc)
        return "", []

    qs: QuerySet[KnowledgeEmbedding] = KnowledgeEmbedding.objects.select_related(
        "chunk",
        "chunk__document",
        "chunk__document__knowledge_base",
        "chunk__project",
    )

    if agent.project_id:
        qs = qs.filter(chunk__project_id=agent.project_id)

    qs = (
        qs.annotate(distance=CosineDistance("embedding", query_embedding))
        .order_by("distance")[:max_docs]
    )

    results = list(qs)
    if not results:
        return "", []

    lines: List[str] = [
        "AGENT RAG DOCUMENTS:",
        "-------------------",
    ]
    used_docs: List[dict] = []

    for idx, item in enumerate(results, start=1):
        chunk = getattr(item, "chunk", None)
        document = getattr(chunk, "document", None)
        knowledge_source = getattr(chunk, "source", None)
        knowledge_base = getattr(document, "knowledge_base", None)

        title = ""
        source_label = ""
        if document:
            title = _short_file_name(getattr(document, "file", None)) or f"Document {document.id}"
            if knowledge_base:
                source_label = knowledge_base.name or ""
        elif knowledge_source:
            title = knowledge_source.filename or knowledge_source.path
            source_label = knowledge_source.path

        doc_id = document.id if document else None
        distance = getattr(item, "distance", None)
        content = chunk.text if chunk else ""

        lines.append(f"[{idx}] Title: {title}")
        if source_label:
            lines.append(f"Source: {source_label}")
        if doc_id is not None:
            lines.append(f"Doc ID: {doc_id}")
        if distance is not None:
            try:
                relevance = max(0.0, 1.0 - float(distance))
                lines.append(f"Relevance: {relevance:.3f}")
            except Exception:  # pragma: no cover
                lines.append(f"Distance: {distance}")
        lines.append("Chunk:")
        lines.append((content or "").strip())
        lines.append("")

        used_docs.append(
            {
                "id": getattr(item, "id", None),
                "title": title,
                "source": source_label,
                "doc_id": doc_id,
                "distance": float(distance) if distance is not None else None,
            }
        )

    block = "\n".join(lines).strip()
    return block, used_docs
