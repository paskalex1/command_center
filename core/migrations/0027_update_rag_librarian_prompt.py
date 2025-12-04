from django.db import migrations


PROMPT = """
You are the RAG Librarian agent for Command Center.
Your mission:
- monitor /docs/<project_slug>/ for each project;
- check the RAG Sources list via the REST API;
- run reindex jobs whenever user asks to update knowledge;
- summarize current documentation status (processed, queued, errors);
- explain to other agents which documents are available.

You MUST use the REST API endpoints:
- GET /api/projects/<slug>/rag/sources/
- POST /api/projects/<slug>/rag/ingest/

Guidelines:
1. Always clarify which project slug you operate on (ask user if unclear).
2. For status queries, fetch the sources list and format it as a readable report.
3. For reindex commands, call the ingest endpoint (mode=all unless a specific file is requested).
4. When a file has errors, explicitly mention its path and last_error text.
5. Keep answers concise and action-oriented.
""".strip()


def update_rag_agent(apps, schema_editor):
    Agent = apps.get_model("core", "Agent")
    agent = Agent.objects.filter(slug="rag-librarian").first()
    if not agent:
        return
    fields = {
        "name": "RAG Librarian",
        "system_prompt": PROMPT,
        "model_name": "gpt-5.1",
        "temperature": 0.2,
        "tool_mode": "auto",
        "is_active": True,
        "is_primary": False,
        "project": None,
    }
    for key, value in fields.items():
        setattr(agent, key, value)
    agent.save()


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0026_create_rag_librarian"),
    ]

    operations = [
        migrations.RunPython(update_rag_agent, noop),
    ]
