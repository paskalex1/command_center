from celery import shared_task

from command_center.services.llm_models import sync_models_registry


@shared_task
def sync_llm_registry_task() -> None:
    """
    Celery-задача для синхронизации реестра LLM-моделей.
    """
    sync_models_registry()

