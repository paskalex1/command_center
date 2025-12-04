from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0030_create_knowledge_extractor"),
    ]

    operations = [
        migrations.CreateModel(
            name="MessageAttachment",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("file", models.FileField(upload_to="chat_attachments/%Y/%m/%d/", verbose_name="Файл")),
                ("original_name", models.CharField(blank=True, max_length=255, verbose_name="Имя файла")),
                ("mime_type", models.CharField(blank=True, max_length=100, verbose_name="MIME-тип")),
                ("size", models.PositiveIntegerField(default=0, verbose_name="Размер, байт")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="Загружено")),
                (
                    "message",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="attachments",
                        to="core.message",
                        verbose_name="Сообщение",
                    ),
                ),
            ],
            options={
                "verbose_name": "Вложение сообщения",
                "verbose_name_plural": "Вложения сообщений",
                "ordering": ["created_at"],
            },
        ),
    ]
