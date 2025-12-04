from django import forms
from django.contrib import admin
from django.contrib.admin.widgets import FilteredSelectMultiple

admin.site.site_header = "Command Center — администрирование"
admin.site.site_title = "Админка Command Center"
admin.site.index_title = "Администрирование Command Center"

from command_center.llm_registry import get_chat_primary, get_all_models_by_type

from .services.mcp_tools import sync_tools_for_server

from .models import (
    Agent,
    AgentMemory,
    AgentServerBinding,
    Conversation,
    KnowledgeBase,
    KnowledgeChangeLog,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEmbedding,
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeSourceVersion,
    MCPServer,
    MCPTool,
    MemoryEmbedding,
    MemoryEvent,
    Message,
    MessageAttachment,
    Pipeline,
    PipelineStep,
    Project,
    RetrievalLog,
    Task,
)


class AgentAdminForm(forms.ModelForm):
    class Meta:
        model = Agent
        fields = "__all__"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_field = self.fields.get("model_name")
        if model_field:
            allowed_names = []
            for section in ("chat", "code"):
                for item in get_all_models_by_type(section):
                    name = item.get("name")
                    if name:
                        allowed_names.append(name)
            unique_names = sorted(set(allowed_names))
            if unique_names:
                choices = [("", "— использовать модель по умолчанию —")]
                choices.extend((name, name) for name in unique_names)
                model_field.widget = forms.Select(choices=choices)
                model_field.required = False

    def clean(self):
        cleaned = super().clean()
        model_name = cleaned.get("model_name")
        if not model_name:
            default_model = get_chat_primary()
            if not default_model:
                raise forms.ValidationError(
                    "Нужно указать модель LLM или настроить модель по умолчанию в реестре."
                )

        if self.instance.pk is None and cleaned.get("is_active") is None:
            cleaned["is_active"] = True

        return cleaned


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "slug", "created_at")
    list_display_links = ("id", "name")
    search_fields = ("name",)
    fieldsets = (
        (None, {"fields": ("name", "slug", "description")}),
    )

    def get_readonly_fields(self, request, obj=None):
        readonly = list(super().get_readonly_fields(request, obj))
        if obj:
            readonly.append("slug")
        return readonly


@admin.register(KnowledgeBase)
class KnowledgeBaseAdmin(admin.ModelAdmin):
    list_display = ("id", "project", "name", "created_at")
    list_display_links = ("id", "project", "name")
    search_fields = ("name",)
    list_filter = ("project",)


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "knowledge_base", "mime_type", "status", "created_at")
    list_filter = ("status", "knowledge_base")


@admin.register(KnowledgeChunk)
class KnowledgeChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "project", "chunk_index")
    list_display_links = ("id", "document")
    list_filter = ("project", "document")


@admin.register(KnowledgeEmbedding)
class KnowledgeEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("id", "chunk", "created_at")
    list_display_links = ("id", "chunk")


@admin.register(KnowledgeSourceVersion)
class KnowledgeSourceVersionAdmin(admin.ModelAdmin):
    list_display = ("id", "source", "content_hash", "size_bytes", "created_at")
    list_filter = ("project", "source__path")
    search_fields = ("source__path", "content_hash")
    readonly_fields = ("created_at",)


@admin.register(KnowledgeChangeLog)
class KnowledgeChangeLogAdmin(admin.ModelAdmin):
    list_display = ("id", "project", "source", "change_type", "created_at")
    list_filter = ("project", "change_type")
    search_fields = ("source__path", "semantic_summary")
    readonly_fields = ("created_at",)


@admin.register(MCPServer)
class MCPServerAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "slug",
        "name",
        "transport",
        "is_active",
        "last_synced_at",
        "last_error",
        "created_at",
    )
    list_display_links = ("id", "name")
    list_filter = ("transport", "is_active")
    search_fields = ("name",)
    readonly_fields = ("last_synced_at", "last_error")
    actions = ("sync_tools_action",)

    def sync_tools_action(self, request, queryset):
        lines = []
        for server in queryset:
            stats = sync_tools_for_server(server)
            text = (
                f"{server.name}: "
                f"{stats.get('created', 0)} created, "
                f"{stats.get('updated', 0)} updated, "
                f"{stats.get('disabled', 0)} disabled"
            )
            errors = stats.get("errors") or []
            if errors:
                text += f" | errors: {', '.join(errors)}"
            lines.append(text)
        if lines:
            self.message_user(request, " ; ".join(lines))
        else:
            self.message_user(request, "Инструменты не синхронизированы (нет изменений).")

    sync_tools_action.short_description = "Синхронизировать инструменты MCP"


@admin.register(MCPTool)
class MCPToolAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "server", "is_active", "created_at")
    list_display_links = ("id", "name")
    list_filter = ("server", "is_active")
    search_fields = ("name",)


@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    form = AgentAdminForm
    list_display = ("id", "slug", "name", "project", "model_name", "is_primary", "is_active")
    list_display_links = ("id", "name")
    list_filter = ("project", "is_primary", "is_active")
    search_fields = ("name",)
    filter_horizontal = ("delegates",)
    fieldsets = (
        (
            None,
            {
                "fields": (
                    "project",
                    "slug",
                    "name",
                    "system_prompt",
                )
            },
        ),
        (
            "LLM-настройки",
            {
                "fields": (
                    "model_name",
                    "temperature",
                    "max_tokens",
                    "tool_mode",
                )
            },
        ),
        (
            "Статус",
            {
                "fields": (
                    "is_primary",
                    "is_active",
                )
            },
        ),
        (
            "Делегаты",
            {
                "fields": ("delegates",),
            },
        ),
    )


@admin.register(AgentServerBinding)
class AgentServerBindingAdmin(admin.ModelAdmin):
    list_display = ("id", "agent", "server")
    list_display_links = ("id", "agent", "server")
    filter_horizontal = ("allowed_tools",)

    def get_form(self, request, obj=None, **kwargs):
        request._obj_ = obj
        return super().get_form(request, obj, **kwargs)

    def formfield_for_manytomany(self, db_field, request, **kwargs):
        if db_field.name == "allowed_tools":
            obj = getattr(request, "_obj_", None)
            if obj and obj.server_id:
                kwargs["queryset"] = MCPTool.objects.filter(server=obj.server)
            else:
                kwargs["queryset"] = MCPTool.objects.none()

            field = super().formfield_for_manytomany(db_field, request, **kwargs)

            def label_from_instance(tool: MCPTool) -> str:
                desc = (tool.description or "").strip()
                first_line = desc.splitlines()[0] if desc else ""
                return f"{tool.name} — {first_line}" if first_line else tool.name

            field.label_from_instance = label_from_instance
            field.widget = FilteredSelectMultiple("Разрешённые инструменты", is_stacked=False)
            return field

        return super().formfield_for_manytomany(db_field, request, **kwargs)


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "project", "agent", "is_archived", "created_at")
    list_display_links = ("id", "name")
    list_filter = ("project", "agent", "is_archived")
    search_fields = ("title",)

    @admin.display(description="Название")
    def name(self, obj):
        return obj.title


class MessageAttachmentInline(admin.TabularInline):
    model = MessageAttachment
    extra = 0
    readonly_fields = ("original_name", "mime_type", "size", "created_at", "file")
    fields = ("original_name", "mime_type", "size", "created_at", "file")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "sender", "created_at")
    list_display_links = ("id", "conversation")
    list_filter = ("sender", "conversation")
    inlines = [MessageAttachmentInline]


@admin.register(MemoryEvent)
class MemoryEventAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "agent",
        "project",
        "conversation",
        "type",
        "importance",
        "created_at",
    )
    list_display_links = ("id", "agent", "project")
    list_filter = ("type", "agent", "project")


@admin.register(MemoryEmbedding)
class MemoryEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("id", "event", "created_at")
    list_display_links = ("id", "event")


@admin.register(AgentMemory)
class AgentMemoryAdmin(admin.ModelAdmin):
    list_display = ("id", "agent", "importance", "created_at", "updated_at")
    list_display_links = ("id", "agent")
    list_filter = ("importance", "agent")
    search_fields = ("content",)


@admin.register(RetrievalLog)
class RetrievalLogAdmin(admin.ModelAdmin):
    list_display = ("id", "agent", "created_at")
    list_display_links = ("id", "agent")
    search_fields = ("query",)


@admin.register(KnowledgeNode)
class KnowledgeNodeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "agent",
        "label",
        "type",
        "object_type",
        "object_id",
        "created_at",
    )
    list_display_links = ("id", "label")
    list_filter = ("agent", "type", "object_type")
    search_fields = ("label", "description", "object_type", "object_id")


@admin.register(KnowledgeEdge)
class KnowledgeEdgeAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "agent",
        "source",
        "relation",
        "target",
        "weight",
        "created_at",
    )
    list_display_links = ("id", "relation")
    list_filter = ("agent", "relation")
    search_fields = ("source__label", "target__label", "relation", "description")


@admin.register(Pipeline)
class PipelineAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "project", "owner_agent", "is_active")
    list_display_links = ("id", "name")
    list_filter = ("project", "owner_agent", "is_active")
    search_fields = ("name",)


@admin.register(PipelineStep)
class PipelineStepAdmin(admin.ModelAdmin):
    list_display = ("id", "pipeline", "order", "type", "agent", "tool")
    list_display_links = ("id", "pipeline")


@admin.register(Task)
class TaskAdmin(admin.ModelAdmin):
    list_display = ("id", "pipeline", "status", "current_step_index", "created_at")
    list_display_links = ("id", "pipeline")
    list_filter = ("status", "pipeline")


# --- Custom ordering for left admin menu within "Командный центр" app ---
from django.contrib.admin import AdminSite


_original_get_app_list = AdminSite.get_app_list


def _core_ordered_get_app_list(self, request):
    app_list = _original_get_app_list(self, request)

    desired_order = [
        # Проекты и диалоги
        "Project",
        "Conversation",
        "Message",
        # Знания
        "KnowledgeBase",
        "KnowledgeDocument",
        "KnowledgeChunk",
        "KnowledgeEmbedding",
        # MCP и агенты
        "MCPServer",
        "MCPTool",
        "Agent",
        "AgentServerBinding",
        # Пайплайны
        "Pipeline",
        "PipelineStep",
        "Task",
        # Память
        "MemoryEvent",
        "MemoryEmbedding",
    ]
    order_map = {name: index for index, name in enumerate(desired_order)}

    for app in app_list:
        if app.get("app_label") == "core":
            app["models"].sort(
                key=lambda m: order_map.get(m.get("object_name"), 10**6)
            )
    return app_list


AdminSite.get_app_list = _core_ordered_get_app_list
