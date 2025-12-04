from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"
    verbose_name = "Командный центр"

    def ready(self):
        # Импорт сигналов гарантирует создание docs/<slug> при старте.
        import core.signals  # noqa: F401
