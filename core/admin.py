from django.contrib import admin
from django.contrib.admin.widgets import FilteredSelectMultiple

admin.site.site_header = "Command Center — администрирование"
admin.site.site_title = "Админка Command Center"
admin.site.index_title = "Администрирование Command Center"

from .models import (
    Agent,
    AgentServerBinding,
    Conversation,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEmbedding,
    MCPServer,
    MCPTool,
    MemoryEmbedding,
    MemoryEvent,
    Message,
    Pipeline,
    PipelineStep,
    Project,
    Task,
)


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "created_at")
    list_display_links = ("id", "name")
    search_fields = ("name",)


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


@admin.register(MCPServer)
class MCPServerAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "transport", "is_active", "created_at")
    list_display_links = ("id", "name")
    list_filter = ("transport", "is_active")
    search_fields = ("name",)


@admin.register(MCPTool)
class MCPToolAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "server", "is_active", "created_at")
    list_display_links = ("id", "name")
    list_filter = ("server", "is_active")
    search_fields = ("name",)


@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "project", "model_name", "is_primary", "is_active")
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


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "sender", "created_at")
    list_display_links = ("id", "conversation")
    list_filter = ("sender", "conversation")


@admin.register(MemoryEvent)
class MemoryEventAdmin(admin.ModelAdmin):
    list_display = ("id", "agent", "project", "type", "importance", "created_at")
    list_display_links = ("id", "agent", "project")
    list_filter = ("type", "agent", "project")


@admin.register(MemoryEmbedding)
class MemoryEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("id", "event", "created_at")
    list_display_links = ("id", "event")


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
