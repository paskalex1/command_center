import json
from pathlib import Path

from django.core.management.base import BaseCommand

from core.models import (
    Agent,
    AgentServerBinding,
    MCPServer,
    Project,
)


class Command(BaseCommand):
    help = "Экспортирует проекты, агенты и MCP-серверы в JSON-конфигурацию."

    def add_arguments(self, parser):
        parser.add_argument(
            "path", help="Путь, по которому будет сохранён экспортированный файл."
        )

    def handle(self, *args, **options):
        path = Path(options["path"]).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "projects": [],
            "mcp_servers": [],
            "agents": [],
            "agent_accesses": [],
        }

        for project in Project.objects.all().order_by("slug", "id"):
            data["projects"].append(
                {
                    "slug": project.slug,
                    "name": project.name,
                    "description": project.description or "",
                }
            )

        for server in MCPServer.objects.all().order_by("slug", "name"):
            data["mcp_servers"].append(
                {
                    "slug": server.slug,
                    "name": server.name,
                    "description": server.description or "",
                    "base_url": server.base_url or "",
                    "command": server.command or "",
                    "command_args": server.command_args or [],
                    "transport": server.transport,
                    "is_active": server.is_active,
                }
            )

        for agent in Agent.objects.all().order_by("slug", "name"):
            data["agents"].append(
                {
                    "slug": agent.slug,
                    "name": agent.name,
                    "system_prompt": agent.system_prompt or "",
                    "model_name": agent.model_name or "",
                    "temperature": agent.temperature,
                    "max_tokens": agent.max_tokens,
                    "tool_mode": agent.tool_mode,
                    "is_primary": agent.is_primary,
                    "is_active": agent.is_active,
                    "project_slug": agent.project.slug if agent.project else None,
                }
            )

        bindings = (
            AgentServerBinding.objects.prefetch_related("allowed_tools")
            .select_related("agent", "server")
            .all()
        )

        for binding in bindings:
            data["agent_accesses"].append(
                {
                    "agent_slug": binding.agent.slug,
                    "server_slug": binding.server.slug,
                    "allowed_tools": [
                        tool.name for tool in binding.allowed_tools.all()
                    ],
                }
            )

        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)

        self.stdout.write(f"Конфигурация экспортирована в {path}")
