from __future__ import annotations

from ..constants import RAG_LIBRARIAN_SLUG
from ..models import Agent


def get_rag_delegate(agent: Agent) -> Agent | None:
    """
    Возвращает делегата RAG Librarian для данного агента, если он есть.
    """

    try:
        delegates = agent.delegates.all()
    except Exception:  # pragma: no cover
        return None

    for delegate in delegates:
        slug = (delegate.slug or "").lower()
        name = (delegate.name or "").lower()
        if RAG_LIBRARIAN_SLUG in slug or name == "rag librarian":
            return delegate
    return None
