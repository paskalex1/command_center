from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

from django.db import transaction
from django.db.models import F
from django.utils import timezone

from core.embeddings import embed_text
from core.models import Agent, KnowledgeEdge, KnowledgeNode

logger = logging.getLogger(__name__)

PIN_USAGE_THRESHOLD = 5


def _normalize(value: str, limit: int) -> str:
    value = (value or "").strip()
    if len(value) > limit:
        return value[:limit]
    return value


def _get_or_create_node(agent: Agent, label: str, node_type: str, description: str) -> Tuple[KnowledgeNode, bool]:
    label_norm = _normalize(label, 255)
    type_norm = _normalize(node_type or "entity", 100)
    defaults = {
        "description": _normalize(description, 2000),
    }
    node, created = KnowledgeNode.objects.get_or_create(
        agent=agent,
        label=label_norm,
        type=type_norm,
        defaults=defaults,
    )
    if not created and description and len(description) > len(node.description or ""):
        node.description = defaults["description"]
        node.save(update_fields=["description", "updated_at"])
    elif created and description:
        try:
            node.embedding = embed_text(description)
            node.save(update_fields=["description", "embedding"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to embed node %s: %s", node.id, exc)
    return node, created


def _create_edge(agent: Agent, source: KnowledgeNode, target: KnowledgeNode, relation: str, description: str) -> bool:
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
        relation=_normalize(relation, 100),
        description=_normalize(description, 1000),
        weight=1.0,
    )
    return True


def ingest_extracted_knowledge(agent: Agent, nodes: Iterable[dict], edges: Iterable[dict]) -> Dict[str, int]:
    """
    Persists nodes and edges returned by the Knowledge Extractor.
    """

    stats = {
        "created_nodes": 0,
        "updated_nodes": 0,
        "created_edges": 0,
    }
    if agent is None:
        return stats

    nodes = list(nodes or [])
    edges = list(edges or [])
    if not nodes and not edges:
        return stats

    node_cache: Dict[tuple[str, str], KnowledgeNode] = {}

    with transaction.atomic():
        for entry in nodes:
            label = (entry or {}).get("text") or (entry or {}).get("label") or ""
            if not label.strip():
                continue
            description = (entry or {}).get("description") or ""
            node_type = (entry or {}).get("type") or "entity"
            node, created = _get_or_create_node(agent, label, node_type, description)
            cache_key = (label.strip().lower(), (node.type or "").lower())
            node_cache[cache_key] = node

            next_usage = (node.usage_count or 0) + 1
            update_kwargs = {
                "usage_count": F("usage_count") + 1,
                "updated_at": timezone.now(),
            }
            if not node.is_pinned and next_usage >= PIN_USAGE_THRESHOLD:
                update_kwargs["is_pinned"] = True
            KnowledgeNode.objects.filter(pk=node.pk).update(**update_kwargs)
            if created:
                stats["created_nodes"] += 1
            else:
                stats["updated_nodes"] += 1

        def resolve_node(label: str, node_type: str) -> KnowledgeNode | None:
            key = (label.strip().lower(), (node_type or "entity").strip().lower())
            cached = node_cache.get(key)
            if cached:
                return cached
            qs = KnowledgeNode.objects.filter(agent=agent, label__iexact=label.strip())
            if node_type:
                qs = qs.filter(type__iexact=node_type.strip())
            node = qs.first()
            if node:
                node_cache[key] = node
            return node

        for entry in edges:
            source_label = (entry or {}).get("source") or ""
            target_label = (entry or {}).get("target") or ""
            relation = (entry or {}).get("relation") or ""
            if not (source_label.strip() and target_label.strip() and relation.strip()):
                continue
            source_type = (entry or {}).get("source_type") or ""
            target_type = (entry or {}).get("target_type") or ""
            source_node = resolve_node(source_label, source_type)
            target_node = resolve_node(target_label, target_type)
            if not (source_node and target_node):
                continue
            description = (entry or {}).get("description") or ""
            created_edge = _create_edge(agent, source_node, target_node, relation, description)
            if created_edge:
                stats["created_edges"] += 1

    return stats
