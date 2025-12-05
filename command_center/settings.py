import os
from pathlib import Path

from celery.schedules import crontab
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DOCS_ROOT = BASE_DIR / "docs"
PROJECT_DOCS_ROOT.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "insecure-default-key")

DEBUG = os.getenv("DJANGO_DEBUG", "0") == "1"

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "pgvector.django",
    "core.apps.CoreConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "command_center.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "command_center.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("POSTGRES_DB", "command_center"),
        "USER": os.getenv("POSTGRES_USER", "command_center"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
        "HOST": os.getenv("POSTGRES_HOST", "postgres"),
        "PORT": int(os.getenv("POSTGRES_PORT", "5432")),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

LANGUAGE_CODE = "ru"

TIME_ZONE = "Europe/Moscow"
USE_I18N = True
USE_L10N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

LOGIN_URL = "/admin/login/"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
}

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
CELERY_RESULT_BACKEND = CELERY_BROKER_URL

CELERY_BEAT_SCHEDULE = {
    "sync-llm-registry-monthly": {
        "task": "command_center.tasks.sync_llm_registry_task",
        "schedule": crontab(day_of_month=1, hour=3, minute=0),
    },
    "cleanup-agent-memory-daily": {
        "task": "core.tasks.cleanup_agent_memory",
        "schedule": crontab(hour=3, minute=0),
    },
    "cleanup-graph-memory-daily": {
        "task": "core.tasks.cleanup_graph_memory",
        "schedule": crontab(hour=4, minute=0),
    },
    "sync-rag-projects-daily": {
        "task": "core.tasks.sync_all_projects_rag",
        "schedule": crontab(hour=5, minute=0),
    },
}

# MCP HTTP settings
MCP_HTTP_ORIGIN = "http://command-center"

AGENT_MEMORY_RETENTION_LOW_DAYS = int(os.getenv("AGENT_MEMORY_RETENTION_LOW_DAYS", "60"))
AGENT_MEMORY_RETENTION_NORMAL_DAYS = int(os.getenv("AGENT_MEMORY_RETENTION_NORMAL_DAYS", "180"))
MAX_AGENT_MEMORY_PER_AGENT = int(os.getenv("MAX_AGENT_MEMORY_PER_AGENT", "2000"))
MAX_GRAPH_NODES_PER_AGENT = int(os.getenv("MAX_GRAPH_NODES_PER_AGENT", "2000"))
MAX_GRAPH_EDGES_PER_AGENT = int(os.getenv("MAX_GRAPH_EDGES_PER_AGENT", "5000"))

RAG_AUTO_SYNC_DAYS = int(os.getenv("RAG_AUTO_SYNC_DAYS", "3"))

AGENT_TOOL_MAX_ITERATIONS = max(0, int(os.getenv("AGENT_TOOL_MAX_ITERATIONS", "100")))
