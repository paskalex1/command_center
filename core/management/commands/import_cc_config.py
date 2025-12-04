import json
from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError

from core.models import (
    Agent,
    AgentServerBinding,
    MCPServer,
    MCPTool,
    Project,
)
from core.services.mcp_tools import sync_tools_for_server


class Command(BaseCommand):
    help = "Импортирует проекты, агентов и MCP-серверы из JSON-конфигурации."

    def add_arguments(self, parser):
        parser.add_argument(
            "path", help="Путь к файлу экспорта (json), полученному с export_cc_config."
        )

    def handle(self, *args, **options):
        path = Path(options["path"]).expanduser()
        if not path.exists():
            raise CommandError(f"Файл {path} не найден.")

        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        projects_data = payload.get("projects") or []
        servers_data = payload.get("mcp_servers") or []
        agents_data = payload.get("agents") or []
        accesses_data = payload.get("agent_accesses") or []

        projects_by_slug = {}
        for item in projects_data:
            slug = item.get("slug")
            if not slug:
                continue
            project, _ = Project.objects.update_or_create(
                slug=slug,
                defaults={
                    "name": item.get("name") or slug,
                    "description": item.get("description") or "",
                },
            )
            projects_by_slug[slug] = project
            self.stdout.write(f"Проект {project.slug} ({project.name}) обновлён.")

        servers = []
        for item in servers_data:
            slug = item.get("slug")
            if not slug:
                continue
            server, _ = MCPServer.objects.update_or_create(
                slug=slug,
                defaults={
                    "name": item.get("name") or slug,
                    "description": item.get("description") or "",
                    "base_url": item.get("base_url") or "",
                    "command": item.get("command") or "",
                    "command_args": item.get("command_args") or [],
                    "transport": item.get("transport") or MCPServer.TRANSPORT_HTTP,
                    "is_active": bool(item.get("is_active")),
                },
            )
            servers.append(server)
            self.stdout.write(f"MCP-сервер {server.slug} обновлён.")

        for server in servers:
            if server.is_active and server.transport == MCPServer.TRANSPORT_HTTP:
                stats = sync_tools_for_server(server)
                self.stdout.write(
                    f"Сервер {server.name} синхронизирован: {stats.get('created',0)} created, "
                    f"{stats.get('updated',0)} updated, {stats.get('disabled',0)} disabled."
                )

        agents = {}
        for item in agents_data:
            slug = item.get("slug")
            if not slug:
                continue
            project_slug = item.get("project_slug")
            project = projects_by_slug.get(project_slug)
            if project_slug and not project:
                raise CommandError(f"Проект с slug={project_slug} отсутствует.")

            temperature = item.get("temperature")
            defaults = {
                "project": project,
                "name": item.get("name") or slug,
                "system_prompt": item.get("system_prompt") or "",
                "model_name": item.get("model_name") or "",
                "temperature": temperature if temperature is not None else 0.2,
                "max_tokens": item.get("max_tokens"),
                "tool_mode": item.get("tool_mode") or Agent.ToolMode.AUTO,
                "is_primary": bool(item.get("is_primary")),
                "is_active": bool(item.get("is_active")),
            }

            agent, _ = Agent.objects.update_or_create(slug=slug, defaults=defaults)
            agents[slug] = agent
            self.stdout.write(f"Агент {agent.slug} обновлён.")

        for item in accesses_data:
            agent_slug = item.get("agent_slug")
            server_slug = item.get("server_slug")
            if not agent_slug or not server_slug:
                continue
            agent = agents.get(agent_slug) or Agent.objects.filter(slug=agent_slug).first()
            server = next((s for s in servers if s.slug == server_slug), None)
            if server is None:
                server = MCPServer.objects.filter(slug=server_slug).first()
            if not agent or not server:
                self.stdout.write(
                    f"Пропущен доступ {agent_slug} -> {server_slug} (отсутствуют сущности)."
                )
                continue

            binding, _ = AgentServerBinding.objects.get_or_create(
                agent=agent,
                server=server,
            )

            allowed_tool_names = item.get("allowed_tools") or []
            tools = MCPTool.objects.filter(server=server, name__in=allowed_tool_names)
            binding.allowed_tools.set(tools)
            self.stdout.write(
                f"Доступ агента {agent.slug} к серверу {server.slug} обновлён ({tools.count()} инструментов)."
            )

        call_command("generate_project_docs")
        self.stdout.write("Генерация README в /docs/ завершена.")
