from django.shortcuts import get_object_or_404
from django.db.models import Count
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from core.models import KnowledgeEdge, KnowledgeNode, Project


class ProjectGraphOverviewView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, slug: str):
        project = get_object_or_404(Project, slug=slug)
        node_qs = KnowledgeNode.objects.filter(agent__project=project)
        edge_qs = KnowledgeEdge.objects.filter(agent__project=project)

        nodes_count = node_qs.count()
        edges_count = edge_qs.count()

        top_nodes_qs = node_qs.order_by("-usage_count", "-updated_at")[:5]
        top_nodes = [
            {
                "label": node.label,
                "type": node.type,
                "usage_count": node.usage_count,
            }
            for node in top_nodes_qs
        ]

        top_relations = list(
            edge_qs.values("relation")
            .annotate(count=Count("id"))
            .order_by("-count")[:5]
        )

        latest_updated = (
            node_qs.order_by("-updated_at").values_list("updated_at", flat=True).first()
        )

        return Response(
            {
                "project": {
                    "id": project.id,
                    "slug": project.slug,
                },
                "nodes": nodes_count,
                "edges": edges_count,
                "top_nodes": top_nodes,
                "top_relations": top_relations,
                "updated_at": latest_updated.isoformat() if latest_updated else None,
            }
        )
