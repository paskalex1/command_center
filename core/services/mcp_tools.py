import logging
from typing import Dict, List

from django.utils import timezone

from core.mcp_client import MCPClientError, list_tools_for_server
from core.models import MCPServer, MCPTool

logger = logging.getLogger(__name__)


def sync_tools_for_server(server: MCPServer) -> Dict[str, object]:
    """
    Стандартная синхронизация tools/list для MCP-сервера.
    Возвращает словарь со статистикой и не поднимает исключения наружу.
    """
    stats: Dict[str, object] = {
        "created": 0,
        "updated": 0,
        "disabled": 0,
        "errors": [],
    }

    now = timezone.now()

    try:
        tools_data = list_tools_for_server(server)
    except MCPClientError as exc:
        error_message = str(exc)
        logger.error(
            "Failed to sync MCP tools for server %s: %s", server.name, error_message
        )
        stats["errors"] = [error_message]
        server.last_error = error_message
        server.last_synced_at = now
        type(server).objects.filter(pk=server.pk).update(
            last_error=error_message,
            last_synced_at=now,
        )
        return stats

    seen_names: List[str] = []
    created = 0
    updated = 0

    for tool in tools_data:
        name = tool.get("name")
        if not name:
            continue

        seen_names.append(name)
        defaults = {
            "description": tool.get("description", "") or "",
            "input_schema": tool.get("input_schema") or {},
            "output_schema": tool.get("output_schema") or {},
            "is_active": True,
        }

        _, created_flag = MCPTool.objects.update_or_create(
            server=server,
            name=name,
            defaults=defaults,
        )
        if created_flag:
            created += 1
        else:
            updated += 1

    disabled = MCPTool.objects.filter(server=server).exclude(
        name__in=[name for name in seen_names if name]
    ).update(is_active=False)

    stats["created"] = created
    stats["updated"] = updated
    stats["disabled"] = disabled

    server.last_error = ""
    server.last_synced_at = now
    type(server).objects.filter(pk=server.pk).update(
        last_error="",
        last_synced_at=now,
    )
    return stats
