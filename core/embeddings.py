import logging
import os
from functools import lru_cache
from typing import List

from openai import OpenAI

from command_center.llm_registry import get_embedding_default


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set, cannot create embeddings")
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def embed_text(text: str) -> List[float]:
    """Return embedding vector for arbitrary text using registry default model."""

    if not text:
        return []

    model_name = get_embedding_default() or "text-embedding-3-small"
    client = _get_client()
    response = client.embeddings.create(
        model=model_name,
        input=text,
    )
    return response.data[0].embedding

