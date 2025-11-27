from django.db import models
from pgvector.django import VectorField

from command_center.llm_registry import (
    get_all_known_model_names,
    get_chat_primary,
    is_model_deprecated,
)

import logging
from urllib.parse import urlparse
from django.core.exceptions import ValidationError


logger = logging.getLogger(__name__)


def validate_mcp_base_url(value: str):
    """
    Более мягкая валидация базового URL MCP:
    - допускает docker-хосты (filesystem-mcp, my-service и т.п.);
    - требует только http/https и наличие host:port.
    """
    if not value:
        return

    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            "URL MCP сервера должен начинаться с http:// или https://"
        )

    if not parsed.netloc:
        raise ValidationError(
            "URL MCP сервера должен содержать хост (например: filesystem-mcp:8015)"
        )


class Project(models.Model):
    name = models.CharField("Название", max_length=255)
    description = models.TextField("Описание", blank=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Проект"
        verbose_name_plural = "Проекты"

    def __str__(self) -> str:
        return self.name


class KnowledgeBase(models.Model):
    project = models.ForeignKey(
        Project,
        verbose_name="Проект",
        on_delete=models.CASCADE,
        related_name="knowledge_bases",
    )
    name = models.CharField("Название", max_length=255)
    description = models.TextField("Описание", blank=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "База знаний"
        verbose_name_plural = "Базы знаний"

    def __str__(self) -> str:
        return f"{self.project_id}: {self.name}"


def kb_document_path(instance: "KnowledgeDocument", filename: str) -> str:
    project_id = instance.knowledge_base.project_id
    kb_id = instance.knowledge_base_id
    return f"knowledge/{project_id}/{kb_id}/{filename}"


class KnowledgeDocument(models.Model):
    STATUS_NEW = "new"
    STATUS_PROCESSING = "processing"
    STATUS_READY = "ready"
    STATUS_ERROR = "error"

    STATUS_CHOICES = [
        (STATUS_NEW, "Новый"),
        (STATUS_PROCESSING, "Обрабатывается"),
        (STATUS_READY, "Готов"),
        (STATUS_ERROR, "Ошибка"),
    ]

    knowledge_base = models.ForeignKey(
        KnowledgeBase,
        verbose_name="База знаний",
        on_delete=models.CASCADE,
        related_name="documents",
    )
    file = models.FileField("Файл", upload_to=kb_document_path)
    mime_type = models.CharField("MIME-тип", max_length=255)
    status = models.CharField(
        "Статус",
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_NEW,
    )
    meta = models.JSONField("Метаданные", default=dict, blank=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Документ"
        verbose_name_plural = "Документы"

    def __str__(self) -> str:
        return f"Document {self.id} ({self.mime_type})"


class KnowledgeChunk(models.Model):
    document = models.ForeignKey(
        KnowledgeDocument,
        verbose_name="Документ",
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    project = models.ForeignKey(
        Project,
        verbose_name="Проект",
        on_delete=models.CASCADE,
        related_name="chunks",
    )
    text = models.TextField("Текст")
    chunk_index = models.PositiveIntegerField("Индекс фрагмента")
    meta = models.JSONField("Метаданные", default=dict, blank=True)

    class Meta:
        verbose_name = "Фрагмент"
        verbose_name_plural = "Фрагменты"
        ordering = ["chunk_index"]

    def __str__(self) -> str:
        return f"Chunk {self.chunk_index} of document {self.document_id}"


class KnowledgeEmbedding(models.Model):
    chunk = models.ForeignKey(
        KnowledgeChunk,
        verbose_name="Фрагмент",
        on_delete=models.CASCADE,
        related_name="embeddings",
    )
    embedding = VectorField("Вектор", dimensions=1536)
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Эмбеддинг (фрагмент)"
        verbose_name_plural = "Эмбеддинги (фрагменты)"
        constraints = [
            models.UniqueConstraint(
                fields=["chunk"], name="unique_embedding_per_chunk"
            )
        ]

    def __str__(self) -> str:
        return f"Embedding for chunk {self.chunk_id}"


class MCPServer(models.Model):
    TRANSPORT_STDIO = "stdio"
    TRANSPORT_HTTP = "http"

    TRANSPORT_CHOICES = [
        (TRANSPORT_STDIO, "STDIO"),
        (TRANSPORT_HTTP, "HTTP"),
    ]

    name = models.CharField("Название", max_length=100, unique=True)
    description = models.TextField("Описание", blank=True)
    base_url = models.CharField(
        "Базовый URL MCP сервера",
        max_length=255,
        blank=True,
        null=True,
        help_text="Например: http://filesystem-mcp:8015/mcp (для HTTP-транспорта)",
        validators=[validate_mcp_base_url],
    )

    command = models.CharField(
        "Команда запуска",
        max_length=255,
        blank=True,
        null=True,
        help_text=(
            "Например: python -m core.dummy_mcp_server (для STDIO-транспорта). "
            "Для HTTP можно оставить пустым."
        ),
    )
    command_args = models.JSONField(
        "Аргументы команды",
        blank=True,
        null=True,
        help_text=(
            "Список аргументов для команды (используется только для STDIO-транспорта)."
        ),
    )

    transport = models.CharField(
        "Транспорт",
        max_length=20,
        choices=TRANSPORT_CHOICES,
        default=TRANSPORT_STDIO,
    )

    is_active = models.BooleanField("Активен", default=True)

    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "MCP-сервер"
        verbose_name_plural = "MCP-серверы"

    def __str__(self) -> str:
        return self.name


class MCPTool(models.Model):
    server = models.ForeignKey(
        MCPServer,
        verbose_name="MCP сервер",
        on_delete=models.CASCADE,
        related_name="tools",
    )
    name = models.CharField("Название", max_length=100)
    description = models.TextField("Описание", blank=True)
    input_schema = models.JSONField("Схема входа", default=dict, blank=True)
    output_schema = models.JSONField("Схема выхода", default=dict, blank=True)
    is_active = models.BooleanField("Активен", default=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "Инструмент MCP"
        verbose_name_plural = "Инструменты MCP"
        unique_together = ("server", "name")

    def __str__(self) -> str:
        return f"{self.server.name}:{self.name}"


class Agent(models.Model):
    class ToolMode(models.TextChoices):
        AUTO = "auto", "Авто (инструменты по желанию)"
        REQUIRED = "required", "Только через инструменты"

    project = models.ForeignKey(
        Project,
        verbose_name="Проект",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="agents",
        help_text="null = глобальный агент",
    )
    name = models.CharField("Имя агента", max_length=100)
    system_prompt = models.TextField("Системный промпт", blank=True)
    model_name = models.CharField(
        "Модель",
        max_length=100,
        blank=True,
        default="",
        help_text="LLM модель, например: gpt-5.1. Пусто = использовать модель по умолчанию из реестра.",
    )
    temperature = models.FloatField(
        "Температура",
        default=0.2,
        help_text="Креативность модели (0.0–1.0)",
    )
    max_tokens = models.IntegerField(
        "Максимум токенов",
        null=True,
        blank=True,
        help_text="Ограничение длины ответа модели; null = по умолчанию модели",
    )
    delegates = models.ManyToManyField(
        "self",
        symmetrical=False,
        blank=True,
        related_name="delegated_by",
        verbose_name="Агенты, которым можно делегировать задачи",
    )
    tool_mode = models.CharField(
        "Режим инструментов",
        max_length=16,
        choices=ToolMode.choices,
        default=ToolMode.AUTO,
        help_text=(
            "Как агент использует MCP-инструменты и делегатов: 'Авто' — может игнорировать; "
            "'Только через инструменты' — обязан вызывать хотя бы один инструмент."
        ),
    )
    is_primary = models.BooleanField("Агент по умолчанию", default=False)
    is_active = models.BooleanField("Активен", default=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "Агент"
        verbose_name_plural = "Агенты"
        unique_together = ("project", "name")

    def __str__(self) -> str:
        return self.name

    @property
    def resolved_model_name(self) -> str:
        """
        Фактическое имя модели:
        - если model_name не задано — берём primary из реестра;
        - иначе — возвращаем model_name как есть.
        """
        if self.model_name:
            return self.model_name
        primary = get_chat_primary()
        if primary:
            return primary
        # Фолбэк на случай пустого реестра
        return "gpt-4.1-mini"

    def save(self, *args, **kwargs):
        if self.model_name:
            if is_model_deprecated(self.model_name):
                logger.warning(
                    "Agent %s uses deprecated LLM model '%s' (marked deprecated in registry)",
                    self.pk or "<new>",
                    self.model_name,
                )
            known = set(get_all_known_model_names())
            if known and self.model_name not in known:
                logger.warning(
                    "Agent %s uses unknown LLM model '%s' (not present in registry)",
                    self.pk or "<new>",
                    self.model_name,
                )
        super().save(*args, **kwargs)

    @property
    def warning(self) -> str | None:
        """
        Короткое предупреждение для UI (например, при использовании deprecated модели).
        """
        if self.model_name and is_model_deprecated(self.model_name):
            return "Выбрана устаревшая модель. Рекомендуется сменить."
        return None


class AgentServerBinding(models.Model):
    agent = models.ForeignKey(
        Agent,
        verbose_name="Агент",
        on_delete=models.CASCADE,
        related_name="server_bindings",
    )
    server = models.ForeignKey(
        MCPServer,
        verbose_name="MCP сервер",
        on_delete=models.CASCADE,
        related_name="agent_bindings",
    )
    allowed_tools = models.ManyToManyField(
        MCPTool,
        verbose_name="Разрешённые инструменты",
        blank=True,
        related_name="agent_accesses",
        help_text="Инструменты MCP, которые может вызывать агент",
    )

    class Meta:
        verbose_name = "Доступ агента к MCP"
        verbose_name_plural = "Доступы агентов к MCP"
        unique_together = ("agent", "server")

    def __str__(self) -> str:
        return f"{self.agent.name} -> {self.server.name}"


class Conversation(models.Model):
    project = models.ForeignKey(
        Project,
        verbose_name="Проект",
        on_delete=models.CASCADE,
        related_name="conversations",
    )
    agent = models.ForeignKey(
        Agent,
        verbose_name="Агент",
        on_delete=models.CASCADE,
        related_name="conversations",
    )
    title = models.CharField("Название", max_length=255, blank=True)
    is_archived = models.BooleanField("В архиве", default=False)
    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "Диалог"
        verbose_name_plural = "Диалоги"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.title or f"Conversation {self.id}"


class Message(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"

    ROLE_CHOICES = [
        (ROLE_USER, "Пользователь"),
        (ROLE_ASSISTANT, "Ассистент"),
        (ROLE_SYSTEM, "Система"),
    ]

    conversation = models.ForeignKey(
        Conversation,
        verbose_name="Диалог",
        on_delete=models.CASCADE,
        related_name="messages",
    )
    sender = models.CharField("Отправитель", max_length=20, choices=ROLE_CHOICES)
    content = models.TextField("Текст")
    meta = models.JSONField("Метаданные", default=dict, blank=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Сообщение"
        verbose_name_plural = "Сообщения"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.sender}: {self.content[:50]}"


class MemoryEvent(models.Model):
    TYPE_FACT = "fact"
    TYPE_PREF = "preference"
    TYPE_TASK = "task"

    TYPE_CHOICES = [
        (TYPE_FACT, "Факт"),
        (TYPE_PREF, "Предпочтение"),
        (TYPE_TASK, "Задача"),
    ]

    agent = models.ForeignKey(
        Agent,
        verbose_name="Агент",
        on_delete=models.CASCADE,
        related_name="memory_events",
    )
    project = models.ForeignKey(
        Project,
        verbose_name="Проект",
        on_delete=models.CASCADE,
        related_name="memory_events",
    )
    source_message = models.ForeignKey(
        Message,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="memory_events",
    )
    type = models.CharField("Тип", max_length=50, choices=TYPE_CHOICES)
    content = models.TextField(
        "Содержимое",
        help_text="Нормализованный факт/предпочтение/задача в человеческом виде",
    )
    importance = models.FloatField(
        "Важность",
        help_text="Важность памяти в диапазоне 0.0–1.0",
    )
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Память"
        verbose_name_plural = "Память"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.type}: {self.content[:50]}"


class MemoryEmbedding(models.Model):
    event = models.ForeignKey(
        MemoryEvent,
        verbose_name="Событие памяти",
        on_delete=models.CASCADE,
        related_name="embeddings",
    )
    embedding = VectorField("Вектор", dimensions=1536)
    created_at = models.DateTimeField("Создано", auto_now_add=True)

    class Meta:
        verbose_name = "Эмбеддинг (память)"
        verbose_name_plural = "Эмбеддинги (память)"
        constraints = [
            models.UniqueConstraint(
                fields=["event"],
                name="unique_embedding_per_memory_event",
            )
        ]

    def __str__(self) -> str:
        return f"Embedding for memory event {self.event_id}"


class Pipeline(models.Model):
    project = models.ForeignKey(
        Project,
        verbose_name="Проект",
        on_delete=models.CASCADE,
        related_name="pipelines",
    )
    owner_agent = models.ForeignKey(
        Agent,
        verbose_name="Владелец (агент)",
        on_delete=models.CASCADE,
        related_name="owned_pipelines",
    )
    name = models.CharField("Название", max_length=255)
    description = models.TextField("Описание", blank=True)
    is_active = models.BooleanField("Активен", default=True)
    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "Пайплайн"
        verbose_name_plural = "Пайплайны"
        unique_together = ("project", "name")
        ordering = ["name"]

    def __str__(self) -> str:
        return f"{self.project_id}:{self.name}"


class PipelineStep(models.Model):
    TYPE_AGENT = "agent"
    TYPE_TOOL = "tool"

    TYPE_CHOICES = [
        (TYPE_AGENT, "Агент"),
        (TYPE_TOOL, "Инструмент"),
    ]

    pipeline = models.ForeignKey(
        Pipeline,
        verbose_name="Пайплайн",
        on_delete=models.CASCADE,
        related_name="steps",
    )
    order = models.PositiveIntegerField("Порядок")
    type = models.CharField("Тип", max_length=20, choices=TYPE_CHOICES)
    agent = models.ForeignKey(
        Agent,
        verbose_name="Агент",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="pipeline_steps",
    )
    tool = models.ForeignKey(
        MCPTool,
        verbose_name="Инструмент",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="pipeline_steps",
    )
    config = models.JSONField("Конфигурация", default=dict, blank=True)

    class Meta:
        verbose_name = "Шаг пайплайна"
        verbose_name_plural = "Шаги пайплайнов"
        ordering = ["order"]
        unique_together = ("pipeline", "order")

    def __str__(self) -> str:
        return f"{self.pipeline_id}#{self.order} ({self.type})"


class Task(models.Model):
    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_ERROR = "error"

    STATUS_CHOICES = [
        (STATUS_QUEUED, "В очереди"),
        (STATUS_RUNNING, "Выполняется"),
        (STATUS_DONE, "Готово"),
        (STATUS_ERROR, "Ошибка"),
    ]

    pipeline = models.ForeignKey(
        Pipeline,
        verbose_name="Пайплайн",
        on_delete=models.CASCADE,
        related_name="tasks",
    )
    created_by_agent = models.ForeignKey(
        Agent,
        verbose_name="Создано агентом",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_tasks",
    )
    input_payload = models.JSONField("Входные данные", default=dict)
    result_payload = models.JSONField("Результат", default=dict, blank=True)
    status = models.CharField(
        "Статус",
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_QUEUED,
    )
    current_step_index = models.PositiveIntegerField(
        "Текущий шаг",
        null=True,
        blank=True,
        help_text="Индекс (0-based) текущего шага в отсортированном списке steps",
    )
    error_message = models.TextField(
        "Сообщение об ошибке",
        blank=True,
        help_text="Описание ошибки, если статус = error",
    )
    created_at = models.DateTimeField("Создано", auto_now_add=True)
    updated_at = models.DateTimeField("Обновлено", auto_now=True)

    class Meta:
        verbose_name = "Задача"
        verbose_name_plural = "Задачи"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Task {self.id} for pipeline {self.pipeline_id}"
