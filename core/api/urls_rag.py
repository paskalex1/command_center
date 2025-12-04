from django.urls import path

from .views.rag import (
    ProjectRAGChangeListView,
    ProjectRAGIngestView,
    ProjectRAGSourceListView,
)

urlpatterns = [
    path(
        "<slug:slug>/rag/sources/",
        ProjectRAGSourceListView.as_view(),
        name="project-rag-sources",
    ),
    path(
        "<slug:slug>/rag/ingest/",
        ProjectRAGIngestView.as_view(),
        name="project-rag-ingest",
    ),
    path(
        "<slug:slug>/rag/changelog/",
        ProjectRAGChangeListView.as_view(),
        name="project-rag-changelog",
    ),
]
