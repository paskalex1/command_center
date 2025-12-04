from django.urls import path

from .views.graph import ProjectGraphOverviewView

urlpatterns = [
    path(
        "<slug:slug>/graph/overview/",
        ProjectGraphOverviewView.as_view(),
        name="project-graph-overview",
    ),
]
