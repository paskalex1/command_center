from django.db import migrations


SYSTEM_PROMPT = (
    "You are the Knowledge Extraction Agent.\n"
    "Your task is to extract structured knowledge from text produced within the project:\n"
    "— concepts\n"
    "— entities (Product, Property, CRM models, API endpoints, tables, fields, modules)\n"
    "— relationships (belongs_to, part_of, depends_on, describes, interacts_with)\n"
    "— process steps\n"
    "— key facts and definitions\n\n"
    "You DO NOT generate general text. You only return structured JSON with nodes and edges.\n\n"
    "JSON format:\n"
    "{\n"
    '  \"nodes\": [\n'
    '      {\"type\": \"entity|concept|process|component\", \"text\": \"...\"},\n'
    "      ...\n"
    "  ],\n"
    '  \"edges\": [\n'
    '      {\"source\": \"...\", \"relation\": \"belongs_to|part_of|connected_to\", \"target\": \"...\"},\n'
    "      ...\n"
    "  ]\n"
    "}\n\n"
    "Keep nodes short (1–5 words). Keep relations explicit. Only include high-quality knowledge, skip noise."
)


def create_knowledge_extractor(apps, schema_editor):
    Agent = apps.get_model("core", "Agent")
    Agent.objects.update_or_create(
        slug="knowledge-extractor",
        defaults={
            "name": "Knowledge Extractor",
            "project": None,
            "system_prompt": SYSTEM_PROMPT,
            "model_name": "",
            "temperature": 0.0,
            "tool_mode": "auto",
            "is_primary": False,
            "is_active": True,
        },
    )


def remove_knowledge_extractor(apps, schema_editor):
    Agent = apps.get_model("core", "Agent")
    Agent.objects.filter(slug="knowledge-extractor").delete()


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0029_knowledgesourceversion_knowledgechangelog"),
    ]

    operations = [
        migrations.RunPython(create_knowledge_extractor, remove_knowledge_extractor),
    ]
