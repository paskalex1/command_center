from django.urls import include, path

urlpatterns = [
    path("projects/", include(("core.api.urls_rag", "core.api.rag"))),
    path("projects/", include(("core.api.urls_graph", "core.api.graph"))),
]
