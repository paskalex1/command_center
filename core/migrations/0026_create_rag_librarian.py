from django.db import migrations


def create_rag_librarian(apps, schema_editor):
    Agent = apps.get_model("core", "Agent")

    system_prompt = (
        "Ты RAG Librarian. Твоя ответственность — отслеживать файлы проекта в каталоге docs/, "
        "инициировать сканирование и переиндексацию, объяснять пользователям статус синхронизации "
        "и помогать обновлять документацию. Если тебе делегируют задачу, уточни проект и slug, "
        "запусти необходимые команды (ingest_project_docs) и сообщи результат."
    )

    rag_agent, created = Agent.objects.get_or_create(
        slug="rag-librarian",
        defaults={
            "name": "RAG Librarian",
            "project": None,
            "system_prompt": system_prompt,
            "model_name": "",
            "temperature": 0.2,
            "tool_mode": "auto",
            "is_primary": False,
            "is_active": True,
        },
    )
    if not created:
        # ensure prompt is up to date
        updated = False
        if rag_agent.system_prompt != system_prompt:
            rag_agent.system_prompt = system_prompt
            updated = True
        if not rag_agent.is_active:
            rag_agent.is_active = True
            updated = True
        if updated:
            rag_agent.save(update_fields=["system_prompt", "is_active"])

    # ensure every primary agent can delegate to RAG Librarian
    primary_agents = Agent.objects.filter(is_primary=True, is_active=True)
    for agent in primary_agents:
        agent.delegates.add(rag_agent)


def remove_rag_librarian(apps, schema_editor):
    Agent = apps.get_model("core", "Agent")
    Agent.objects.filter(slug="rag-librarian").delete()


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0025_knowledgechunk_source_knowledgeembedding_source_and_more"),
    ]

    operations = [
        migrations.RunPython(create_rag_librarian, remove_rag_librarian),
    ]
