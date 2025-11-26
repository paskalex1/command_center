import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "command_center.settings")

app = Celery("command_center")

app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

# Явно импортируем задачи проекта, чтобы они были зарегистрированы
import command_center.tasks  # noqa: F401,E402

