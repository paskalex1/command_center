from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0018_remove_memoryevent_graph_processed_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="memoryevent",
            name="graph_processed",
            field=models.BooleanField(default=False, verbose_name="Граф обработан"),
        ),
        migrations.AddField(
            model_name="memoryevent",
            name="updated_at",
            field=models.DateTimeField(auto_now=True, verbose_name="Обновлено"),
        ),
        migrations.RemoveField(
            model_name="project",
            name="graph_processed",
        ),
        migrations.RemoveField(
            model_name="project",
            name="updated_at",
        ),
    ]
