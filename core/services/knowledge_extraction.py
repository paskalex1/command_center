from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

from celery import current_app

from core.constants import KNOWLEDGE_EXTRACTOR_SLUG
from core.models import Agent, Project

try:  # noqa: SIM105
    from openai import OpenAI, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are the Knowledge Extraction Agent.\n"
    "Your task is to extract structured knowledge from text produced within the project:\n"
    "— concepts\n"
    "— entities (Product, Property, CRM models, API endpoints, tables, fields, modules)\n"
    "— relationships (belongs_to, part_of, depends_on, describes, interacts_with)\n"
    "— process steps\n"
    "— key facts and definitions\n\n"
    "You DO NOT generate general text. You only return structured JSON with nodes and edges.\n\n"
    "JSON format:\n"
    "{\n"
    '  \"nodes\": [\n'
    '      {\"type\": \"entity|concept|process|component\", \"text\": \"...\", \"description\": \"...\"},\n'
    "      ...\n"
    "  ],\n"
    '  \"edges\": [\n'
    '      {\"source\": \"...\", \"relation\": \"belongs_to|part_of|connected_to\", \"target\": \"...\", \"description\": \"...\"},\n'
    "      ...\n"
    "  ]\n"
    "}\n\n"
    "Keep nodes short (1–5 words). Keep relations explicit. Only include high-quality knowledge, skip noise."
)

KNOWLEDGE_REQUEST_KEYWORDS = (
    "извлеки",
    "извлечь",
    "добавь в граф",
    "обнови граф",
    "добавь знания",
    "extract knowledge",
    "update graph",
)

MIN_KNOWLEDGE_LENGTH = 300
EXTRACTION_TEXT_LIMIT = 6000


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; skip knowledge extraction")
        return None
    if OpenAI is None:  # pragma: no cover - handled in tests via patching
        logger.warning("OpenAI SDK is not available; skip knowledge extraction")
        return None
    return OpenAI(api_key=api_key)


def _trim_text(text: str) -> str:
    text = (text or "").strip()
    if len(text) > EXTRACTION_TEXT_LIMIT:
        return text[:EXTRACTION_TEXT_LIMIT]
    return text


def is_manual_extraction_request(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    return any(keyword in lowered for keyword in KNOWLEDGE_REQUEST_KEYWORDS)


def is_knowledge_rich_text(text: str) -> bool:
    text = (text or "").strip()
    if len(text) >= MIN_KNOWLEDGE_LENGTH:
        return True
    bullet_chars = sum(text.count(char) for char in ("-", "—", "•", "*"))
    colon_count = text.count(":")
    newline_count = text.count("\n")
    return (bullet_chars + colon_count) >= 4 or newline_count >= 8


def schedule_knowledge_extractor(
    *,
    project_id: int,
    text: str,
    agent_id: Optional[int] = None,
    source: str | None = None,
) -> bool:
    trimmed = _trim_text(text)
    if not trimmed:
        return False
    payload = {
        "project_id": project_id,
        "text": trimmed,
        "target_agent_id": agent_id,
        "source": source or "",
    }
    current_app.send_task("core.tasks.run_knowledge_extractor_for_text", kwargs=payload)
    return True


def maybe_schedule_knowledge_extractor_for_agent(
    *,
    project: Project,
    agent: Optional[Agent],
    text: str,
    source: str,
) -> bool:
    if agent is None:
        return False
    if agent.slug == KNOWLEDGE_EXTRACTOR_SLUG:
        return False
    if not is_knowledge_rich_text(text):
        return False
    return schedule_knowledge_extractor(
        project_id=project.id,
        text=text,
        agent_id=agent.id,
        source=source,
    )


def call_knowledge_extractor(agent: Agent, project: Project, text: str, source: str | None = None) -> dict:
    """
    Calls OpenAI using the Knowledge Extractor prompt and returns parsed JSON.
    """

    client = _get_openai_client()
    if client is None:
        return {"nodes": [], "edges": []}

    user_sections: List[str] = []
    if project:
        user_sections.append(f"Project: {project.name} (slug={project.slug})")
    if source:
        user_sections.append(f"Source: {source}")
    user_sections.append(_trim_text(text))
    user_content = "\n\n".join(section for section in user_sections if section)

    system_prompt = agent.system_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        completion = client.chat.completions.create(
            model=agent.resolved_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            timeout=45,
        )
        content = completion.choices[0].message.content or "{}"
        payload = json.loads(content)
    except (OpenAIError, json.JSONDecodeError) as exc:
        logger.warning("Knowledge extractor failed: %s", exc)
        return {"nodes": [], "edges": []}

    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(edges, list):
        edges = []
    return {"nodes": nodes, "edges": edges}
