from django.core.management.base import BaseCommand

from core.models import Project
from core.signals import ensure_project_docs_infrastructure


class Command(BaseCommand):
    help = "Создаёт файлы документации для всех существующих проектов"

    def handle(self, *args, **options):
        projects = Project.objects.all()
        if not projects:
            self.stdout.write("Проекты отсутствуют, нечего генерировать.")
            return

        for project in projects:
            readme = ensure_project_docs_infrastructure(project)
            self.stdout.write(
                f"README для проекта {project.slug or project.name} -> {readme}"
            )
