from django.db import migrations


FILESYSTEM_SERVER = {
    "slug": "filesystem-mcp",
    "name": "Filesystem MCP",
    "description": "Filesystem bridge based on cyanheads/filesystem-mcp server",
    "base_url": "http://filesystem-mcp:8020/mcp",
}

WEB_KNOWLEDGE_SERVER = {
    "slug": "web-knowledge-mcp",
    "name": "Web Knowledge MCP",
    "description": "search_and_collect_knowledge (DuckDuckGo + fetch + normalize)",
    "base_url": "http://web-knowledge-mcp:8000/mcp",
}


def _ensure_server(apps, server_data):
    MCPServer = apps.get_model("core", "MCPServer")
    Agent = apps.get_model("core", "Agent")
    AgentServerBinding = apps.get_model("core", "AgentServerBinding")

    defaults = {
        "slug": server_data["slug"],
        "description": server_data["description"],
        "base_url": server_data["base_url"],
        "transport": "http",
        "is_active": True,
    }

    server, created = MCPServer.objects.get_or_create(
        name=server_data["name"],
        defaults=defaults,
    )

    if created and not server.slug:
        server.slug = server_data["slug"]
        server.save(update_fields=["slug"])

    updated_fields: list[str] = []
    for field in ("description", "base_url", "transport", "is_active"):
        desired = defaults[field]
        if getattr(server, field) != desired:
            setattr(server, field, desired)
            updated_fields.append(field)
    if updated_fields:
        server.save(update_fields=updated_fields)

    active_agents = Agent.objects.filter(is_active=True).only("id")
    for agent in active_agents:
        AgentServerBinding.objects.get_or_create(agent=agent, server=server)


def seed_servers(apps, schema_editor):
    _ensure_server(apps, FILESYSTEM_SERVER)
    _ensure_server(apps, WEB_KNOWLEDGE_SERVER)


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0031_messageattachment"),
    ]

    operations = [
        migrations.RunPython(seed_servers, migrations.RunPython.noop),
    ]
