import logging
from typing import List, Tuple

from django.db.models import F, QuerySet
from django.utils import timezone
from pgvector.django import CosineDistance

from ..embeddings import embed_text
from ..models import Agent, KnowledgeEdge, KnowledgeNode

logger = logging.getLogger(__name__)


def _format_node_line(node: KnowledgeNode) -> str:
    type_part = node.type or "entity"
    description = node.description.strip() if node.description else ""
    if description:
        return f"- [type={type_part}] {node.label} — {description}"
    return f"- [type={type_part}] {node.label}"


def _format_edge_line(edge: KnowledgeEdge) -> str:
    description = edge.description.strip() if edge.description else ""
    relation = edge.relation or "related_to"
    base = f"- {edge.source.label} --{relation}--> {edge.target.label}"
    if description:
        return f"{base} ({description})"
    return base


def build_graph_memory_block(agent: Agent, query_text: str, max_nodes: int = 5) -> Tuple[str, List[dict]]:
    """
    Возвращает текстовый блок AGENT GRAPH MEMORY на основе ближайших Graph-узлов.
    """

    query = (query_text or "").strip()
    if not query:
        return "", []

    try:
        query_embedding = embed_text(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to embed query for graph memory: %s", exc)
        return "", []

    nodes_qs: QuerySet[KnowledgeNode] = (
        KnowledgeNode.objects.filter(agent=agent, embedding__isnull=False)
        .annotate(distance=CosineDistance("embedding", query_embedding))
        .order_by("distance")[:max_nodes]
    )

    nodes = list(nodes_qs)
    if not nodes:
        return "", []

    now = timezone.now()
    KnowledgeNode.objects.filter(id__in=[node.id for node in nodes]).update(
        usage_count=F("usage_count") + 1,
        last_used_at=now,
    )

    node_lines: List[str] = []
    relation_lines: List[str] = []
    nodes_info: List[dict] = []

    for node in nodes:
        node_lines.append(_format_node_line(node))
        nodes_info.append(
            {
                "id": str(node.id),
                "label": node.label,
                "type": node.type,
                "description": node.description,
                "distance": float(getattr(node, "distance", 0.0))
                if hasattr(node, "distance")
                else None,
            }
        )

        outgoing = list(
            KnowledgeEdge.objects.filter(agent=agent, source=node)
            .select_related("source", "target")
            .order_by("-created_at")[:5]
        )
        incoming = list(
            KnowledgeEdge.objects.filter(agent=agent, target=node)
            .select_related("source", "target")
            .order_by("-created_at")[:5]
        )

        for edge in outgoing + incoming:
            relation_lines.append(_format_edge_line(edge))

    if not relation_lines:
        relation_section = ""
    else:
        relation_section = "Relations:\n" + "\n".join(relation_lines)

    block_parts = [
        "AGENT GRAPH MEMORY:",
        "Nodes:",
        "\n".join(node_lines),
    ]
    if relation_section:
        block_parts.append(relation_section)

    return "\n".join(block_parts), nodes_info
