from django.db import migrations, models
import django.db.models.deletion
import uuid
import pgvector.django


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0015_agentmemory_content_hash"),
    ]

    operations = [
        migrations.CreateModel(
            name="KnowledgeNode",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("label", models.CharField(max_length=255, verbose_name="Метка")),
                ("type", models.CharField(blank=True, default="", max_length=100, verbose_name="Тип сущности")),
                ("description", models.TextField(blank=True, default="", verbose_name="Описание")),
                ("embedding", pgvector.django.VectorField(blank=True, dimensions=1536, null=True, verbose_name="Эмбеддинг")),
                ("object_type", models.CharField(blank=True, default="", max_length=100, verbose_name="Тип объекта")),
                ("object_id", models.CharField(blank=True, default="", max_length=100, verbose_name="ID объекта")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="Создано")),
                ("updated_at", models.DateTimeField(auto_now=True, verbose_name="Обновлено")),
                (
                    "agent",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="knowledge_nodes",
                        to="core.agent",
                        verbose_name="Агент",
                    ),
                ),
            ],
            options={
                "verbose_name": "Графовый узел",
                "verbose_name_plural": "Графовые узлы",
                "ordering": ["label"],
                "unique_together": {("agent", "label", "type")},
            },
        ),
        migrations.CreateModel(
            name="KnowledgeEdge",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("relation", models.CharField(max_length=100, verbose_name="Связь")),
                ("description", models.TextField(blank=True, default="", verbose_name="Описание")),
                ("weight", models.FloatField(default=1.0, verbose_name="Вес")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="Создано")),
                (
                    "agent",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="knowledge_edges",
                        to="core.agent",
                        verbose_name="Агент",
                    ),
                ),
                (
                    "source",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="outgoing_edges",
                        to="core.knowledgenode",
                        verbose_name="Источник",
                    ),
                ),
                (
                    "target",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="incoming_edges",
                        to="core.knowledgenode",
                        verbose_name="Цель",
                    ),
                ),
            ],
            options={
                "verbose_name": "Графовая связь",
                "verbose_name_plural": "Графовые связи",
                "ordering": ["-created_at"],
            },
        ),
    ]
