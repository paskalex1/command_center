from __future__ import annotations

import uuid

from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.pagination import PageNumberPagination

from core.models import KnowledgeSource, KnowledgeChangeLog, Project
from core.tasks import ingest_project_docs, ingest_knowledge_source
from ..serializers.rag import RAGSourceSerializer, RAGChangeSerializer


class ProjectSlugMixin:
    permission_classes = [IsAuthenticated]

    def get_project(self) -> Project:
        if not hasattr(self, "_project"):
            self._project = get_object_or_404(Project, slug=self.kwargs["slug"])
        return self._project


class RAGSourcePagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = "page_size"
    max_page_size = 100


class ProjectRAGSourceListView(ProjectSlugMixin, generics.ListAPIView):
    serializer_class = RAGSourceSerializer
    pagination_class = RAGSourcePagination

    def get_queryset(self):
        project = self.get_project()
        queryset = KnowledgeSource.objects.filter(project=project).order_by("path")
        status_param = self.request.query_params.get("status")
        if status_param:
            queryset = queryset.filter(status=status_param)
        return queryset

    def list(self, request, *args, **kwargs):
        response = super().list(request, *args, **kwargs)
        project = self.get_project()
        data = response.data
        if isinstance(data, list):
            payload = {"count": len(data), "results": data}
        else:
            payload = dict(data)
        payload["project_meta"] = {
            "slug": project.slug,
            "rag_last_full_sync_at": project.rag_last_full_sync_at.isoformat() if project.rag_last_full_sync_at else None,
            "rag_last_error_at": project.rag_last_error_at.isoformat() if project.rag_last_error_at else None,
            "rag_error_count": project.rag_error_count,
        }
        return Response(payload)


class ProjectRAGIngestView(ProjectSlugMixin, APIView):
    def post(self, request, slug: str):
        project = self.get_project()
        mode = (request.data or {}).get("mode", "all")

        if mode == "single":
            source_id = request.data.get("source_id")
            if not source_id:
                return Response(
                    {"detail": "source_id is required for single mode."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                source_uuid = uuid.UUID(str(source_id))
            except (TypeError, ValueError):
                return Response(
                    {"detail": "source_id must be a valid UUID."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            source = get_object_or_404(
                KnowledgeSource, id=source_uuid, project=project
            )
            async_result = ingest_knowledge_source.delay(str(source.id))
        elif mode == "all":
            async_result = ingest_project_docs.delay(project.id)
        else:
            return Response(
                {"detail": "mode must be either 'all' or 'single'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                "status": "queued",
                "task_id": async_result.id if async_result else None,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class ProjectRAGChangeListView(ProjectSlugMixin, generics.ListAPIView):
    serializer_class = RAGChangeSerializer
    pagination_class = RAGSourcePagination

    def get_queryset(self):
        project = self.get_project()
        return (
            KnowledgeChangeLog.objects.filter(project=project)
            .select_related("source")
            .order_by("-created_at")
        )
