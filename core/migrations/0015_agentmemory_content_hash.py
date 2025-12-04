from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0014_remove_memoryevent_source_message_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="agentmemory",
            name="content_hash",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                max_length=64,
                verbose_name="Хэш содержимого",
            ),
        ),
    ]
