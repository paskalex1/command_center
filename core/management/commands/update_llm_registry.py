from django.core.management.base import BaseCommand

from command_center.services.llm_models import sync_models_registry


class Command(BaseCommand):
    help = "Обновляет реестр LLM-моделей (models_registry.json) на основе /v1/models"

    def handle(self, *args, **options):
        self.stdout.write("Запрашиваю список моделей из OpenAI...")
        try:
            registry = sync_models_registry()
        except Exception as exc:  # noqa: BLE001
            self.stderr.write(self.style.ERROR(f"Не удалось обновить реестр моделей: {exc}"))
            return

        chat = registry.get("chat", {})
        embedding = registry.get("embedding", {})

        self.stdout.write(self.style.SUCCESS("Реестр моделей обновлён."))
        self.stdout.write("")
        self.stdout.write(f"chat.primary: {chat.get('primary')}")
        self.stdout.write(f"embedding.default: {embedding.get('default')}")
