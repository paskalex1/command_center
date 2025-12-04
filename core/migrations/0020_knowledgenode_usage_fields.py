from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0019_restore_memoryevent_graph_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="knowledgenode",
            name="is_pinned",
            field=models.BooleanField(
                default=False,
                help_text="Не удалять в автоматической очистке.",
                verbose_name="Защищённый узел",
            ),
        ),
        migrations.AddField(
            model_name="knowledgenode",
            name="last_used_at",
            field=models.DateTimeField(
                blank=True,
                help_text="Когда узел последний раз попадал в контекст.",
                null=True,
                verbose_name="Последнее использование",
            ),
        ),
        migrations.AddField(
            model_name="knowledgenode",
            name="usage_count",
            field=models.IntegerField(
                default=0,
                help_text="Сколько раз узел попадал в graph-recall.",
                verbose_name="Использований",
            ),
        ),
    ]
