from pathlib import Path

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db.models import Q

from core.constants import RAG_LIBRARIAN_SLUG
from core.models import Agent, MCPServer, Project
from core.services.mcp_tools import sync_tools_for_server


def ensure_project_docs_infrastructure(project: Project) -> Path:
    """
    Создаёт папку docs/<slug>/ и README.md для проекта.
    Возвращает путь к сформированному README.
    """
    docs_dir = project.docs_path
    docs_dir.mkdir(parents=True, exist_ok=True)

    readme = docs_dir / "README.md"
    desc = (project.description or "").strip()
    relative_path = Path("docs") / project.slug

    lines = [
        f"# {project.name}",
        "",
        f"Базовая документация проекта **{project.name}**.",
    ]

    if desc:
        lines.extend(
            [
                "",
                "## Описание проекта",
                "",
                desc,
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "Этот файл создан автоматически Command Center при создании проекта.",
            f"Структура проекта доступна в `{relative_path}`.",
        ]
    )

    readme.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return readme


@receiver(post_save, sender=Project)
def create_project_docs_infrastructure(sender, instance: Project, created: bool, **kwargs):
    if not created:
        return
    ensure_project_docs_infrastructure(instance)


@receiver(post_save, sender=MCPServer)
def mcp_server_post_save(sender, instance: MCPServer, **kwargs):
    if not instance.is_active or not instance.base_url:
        return
    if instance.transport != MCPServer.TRANSPORT_HTTP:
        return
    sync_tools_for_server(instance)


@receiver(post_save, sender=Agent)
def ensure_rag_delegate(sender, instance: Agent, **kwargs):
    if not instance.is_primary or not instance.is_active:
        return
    # prevent recursion when saving rag agent itself
    slug = (instance.slug or "").lower()
    name = (instance.name or "").lower()
    if RAG_LIBRARIAN_SLUG in slug or name == "rag librarian":
        return

    rag_filters = Q(slug__icontains=RAG_LIBRARIAN_SLUG) | Q(name__iexact="RAG Librarian")
    rag_agent = (
        Agent.objects.filter(is_active=True, project=instance.project)
        .filter(rag_filters)
        .first()
    )
    if not rag_agent:
        rag_agent = (
            Agent.objects.filter(is_active=True, project__isnull=True)
            .filter(rag_filters)
            .first()
        )
    if not rag_agent:
        return
    instance.delegates.add(rag_agent)
