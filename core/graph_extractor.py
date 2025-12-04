import json
import logging
import os
from typing import Dict, List, Optional, Tuple

from django.db import transaction

try:  # noqa: SIM105
    from openai import OpenAI, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    OpenAIError = Exception  # type: ignore

from .embeddings import embed_text
from .models import Agent, KnowledgeEdge, KnowledgeNode

logger = logging.getLogger(__name__)


GRAPH_EXTRACTOR_PROMPT = (
    "Ты — Graph Memory Extractor в системе Sochi.Rent Command Center.\n"
    "Твоя задача:\n"
    "- читать текст (факт, событие, описание архитектуры, договорённость);\n"
    "- выделять из него важные сущности (узлы графа);\n"
    "- устанавливать связи между этими сущностями.\n\n"
    "Сущности (nodes) — сервисы, MCP-серверы, бизнес-объекты, модели данных,\n"
    "инфраструктура, процессы (например: Command Center, filesystem MCP, Property, Booking, Make-сценарий).\n\n"
    "Связи (edges) — relation типа: uses, depends_on, part_of, deployed_on, manages, connects_to, related_to.\n\n"
    "Формат ответа — СТРОГО JSON:\n"
    "{\n"
    '  "nodes": [\n'
    '    {"label": "...", "type": "...", "description": "...", "object_type": "...", "object_id": "..."},\n'
    "    ...\n"
    "  ],\n"
    '  "edges": [\n'
    '    {"source_label": "...", "target_label": "...", "relation": "...", "description": "...", "weight": 1.0},\n'
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Если нечего извлекать — верни {\"nodes\": [], \"edges\": []}. Никакого текста вне JSON."
)


def _get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; graph extraction skipped")
        return None
    if OpenAI is None:
        logger.warning("OpenAI SDK is not available; graph extraction skipped")
        return None
    return OpenAI(api_key=api_key)


def _select_graph_model(agent: Agent) -> str:
    env_model = os.getenv("GRAPH_MEMORY_MODEL")
    if env_model:
        return env_model
    resolved = getattr(agent, "resolved_model_name", None)
    if resolved:
        return resolved
    return os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")


def _safe_trim(value: str, max_length: int) -> str:
    if value and len(value) > max_length:
        return value[:max_length]
    return value


def _normalize_label(label: str) -> str:
    return (label or "").strip().lower()


def _create_or_update_node(
    *,
    agent: Agent,
    label: str,
    node_type: str,
    description: str,
    object_type: str,
    object_id: str,
) -> Tuple[KnowledgeNode, bool]:
    defaults = {
        "description": description,
        "object_type": object_type,
        "object_id": object_id,
    }
    node, created = KnowledgeNode.objects.get_or_create(
        agent=agent,
        label=_safe_trim(label, 255),
        type=_safe_trim(node_type, 100),
        defaults=defaults,
    )

    updated_fields: List[str] = []
    if not created:
        if description and (not node.description or len(description) > len(node.description)):
            node.description = description
            updated_fields.append("description")
        if object_type and object_type != node.object_type:
            node.object_type = object_type
            updated_fields.append("object_type")
        if object_id and object_id != node.object_id:
            node.object_id = object_id
            updated_fields.append("object_id")
        if updated_fields:
            node.save(update_fields=updated_fields)
    elif not node.embedding:
        text_for_embedding = description or label
        if text_for_embedding:
            try:
                node.embedding = embed_text(text_for_embedding)
                node.save(update_fields=["embedding"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to embed KnowledgeNode %s: %s", node.id, exc)

    return node, created


def _create_edge_if_needed(
    *,
    agent: Agent,
    source: KnowledgeNode,
    target: KnowledgeNode,
    relation: str,
    description: str,
    weight: float,
) -> bool:
    exists = KnowledgeEdge.objects.filter(
        agent=agent,
        source=source,
        target=target,
        relation=relation,
    ).exists()
    if exists:
        return False

    KnowledgeEdge.objects.create(
        agent=agent,
        source=source,
        target=target,
        relation=relation,
        description=description,
        weight=weight or 1.0,
    )
    return True


def extract_graph_from_text(agent: Agent, text: str) -> Dict[str, int]:
    """
    Вызывает LLM, создаёт/обновляет KnowledgeNode и KnowledgeEdge.
    Возвращает статистику вида {"created_nodes": N, "updated_nodes": M, "created_edges": K}.
    """

    stats = {
        "created_nodes": 0,
        "updated_nodes": 0,
        "created_edges": 0,
    }

    text = (text or "").strip()
    if not text:
        return stats

    client = _get_openai_client()
    if client is None:
        return stats

    model_name = _select_graph_model(agent)

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": GRAPH_EXTRACTOR_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            timeout=45,
        )
        content = completion.choices[0].message.content or "{}"
        payload = json.loads(content)
    except (OpenAIError, json.JSONDecodeError) as exc:
        logger.warning("Graph extractor failed for agent %s: %s", agent.id, exc)
        return stats

    nodes_data = payload.get("nodes") or []
    edges_data = payload.get("edges") or []

    if not isinstance(nodes_data, list):
        nodes_data = []
    if not isinstance(edges_data, list):
        edges_data = []

    label_cache: Dict[str, KnowledgeNode] = {}

    try:
        with transaction.atomic():
            for node_entry in nodes_data:
                label = (node_entry or {}).get("label", "").strip()
                if not label:
                    continue

                node_type = (node_entry or {}).get("type", "").strip()
                description = (node_entry or {}).get("description", "").strip()
                object_type = (node_entry or {}).get("object_type", "").strip()
                object_id = (node_entry or {}).get("object_id", "").strip()

                node, created = _create_or_update_node(
                    agent=agent,
                    label=label,
                    node_type=node_type,
                    description=description,
                    object_type=object_type,
                    object_id=object_id,
                )
                key = (_normalize_label(label), _normalize_label(node_type))
                label_cache[key] = node
                if created:
                    stats["created_nodes"] += 1
                else:
                    stats["updated_nodes"] += 1

            def resolve_node(label: str, node_type: str) -> Optional[KnowledgeNode]:
                label_key = _normalize_label(label)
                type_key = _normalize_label(node_type)
                cached = label_cache.get((label_key, type_key)) or label_cache.get((label_key, ""))
                if cached:
                    return cached
                qs = KnowledgeNode.objects.filter(agent=agent, label__iexact=label.strip())
                if node_type:
                    qs = qs.filter(type__iexact=node_type.strip())
                node = qs.first()
                if node:
                    label_cache[(label_key, type_key)] = node
                return node

            for edge_entry in edges_data:
                source_label = (edge_entry or {}).get("source_label", "").strip()
                target_label = (edge_entry or {}).get("target_label", "").strip()
                relation = (edge_entry or {}).get("relation", "").strip()
                if not (source_label and target_label and relation):
                    continue

                source_type = (edge_entry or {}).get("source_type", "").strip()
                target_type = (edge_entry or {}).get("target_type", "").strip()

                source_node = resolve_node(source_label, source_type)
                target_node = resolve_node(target_label, target_type)
                if not (source_node and target_node):
                    continue

                description = (edge_entry or {}).get("description", "").strip()
                weight = edge_entry.get("weight") or 1.0
                created_edge = _create_edge_if_needed(
                    agent=agent,
                    source=source_node,
                    target=target_node,
                    relation=relation,
                    description=description,
                    weight=weight,
                )
                if created_edge:
                    stats["created_edges"] += 1

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to persist graph extraction for agent %s: %s", agent.id, exc)
        return stats

    return stats
