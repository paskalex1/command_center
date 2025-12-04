from __future__ import annotations

from django.utils import timezone

from core.models import Project


def refresh_project_rag_stats(project: Project) -> None:
    """
    Обновляет счетчики ошибок и последние отметки синка для проекта.
    """

    sources = project.knowledge_sources.all()
    error_count = sources.filter(status="error").count()
    updates = {"rag_error_count": error_count}
    if error_count > 0:
        updates["rag_last_error_at"] = timezone.now()
    elif project.rag_error_count != 0:
        updates["rag_last_error_at"] = project.rag_last_error_at

    Project.objects.filter(pk=project.pk).update(**updates)
