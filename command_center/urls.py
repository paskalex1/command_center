from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from core.views import (
    AgentDetailView,
    AgentGraphEdgesView,
    AgentGraphNodesView,
    AgentInvokeView,
    AgentListView,
    AgentMCPAccessView,
    AgentMemoryListView,
    AssistantChatView,
    AssistantConversationView,
    ConversationDetailView,
    KnowledgeBaseCreateView,
    KnowledgeDocumentCreateView,
    MCPServerDetailView,
    MCPServerListCreateView,
    MCPServerSyncToolsView,
    MCPServerToolsListView,
    MCPToolInvokeView,
    PipelineDetailView,
    PipelineListCreateView,
    PipelineRunView,
    ProjectCreateView,
    ProjectSearchView,
    TaskDetailView,
    LLMRegistryView,
    LLMRegistrySyncView,
    dashboard_view,
    memory_view,
    health_check,
)

urlpatterns = [
    path("", dashboard_view, name="ui-dashboard"),
    path("memory/", memory_view, name="ui-memory"),
    path("admin/", admin.site.urls),
    path("api/", include("core.api.urls")),
    path("api/health", health_check, name="api-health"),
    path("api/agents/", AgentListView.as_view(), name="api-agents"),
    path(
        "api/agents/<int:pk>/",
        AgentDetailView.as_view(),
        name="api-agent-detail",
    ),
    path(
        "api/agents/<int:agent_id>/invoke/",
        AgentInvokeView.as_view(),
        name="api-agent-invoke",
    ),
    path(
        "api/agents/<int:agent_id>/mcp-access/",
        AgentMCPAccessView.as_view(),
        name="api-agent-mcp-access",
    ),
    path(
        "api/agents/<int:agent_id>/memories/",
        AgentMemoryListView.as_view(),
        name="api-agent-memories",
    ),
    path(
        "api/agents/<int:agent_id>/graph/nodes/",
        AgentGraphNodesView.as_view(),
        name="api-agent-graph-nodes",
    ),
    path(
        "api/agents/<int:agent_id>/graph/edges/",
        AgentGraphEdgesView.as_view(),
        name="api-agent-graph-edges",
    ),
    path(
        "api/models/registry/",
        LLMRegistryView.as_view(),
        name="api-models-registry",
    ),
    path(
        "api/models/registry/sync/",
        LLMRegistrySyncView.as_view(),
        name="api-models-registry-sync",
    ),
    path(
        "api/conversations/<int:pk>/",
        ConversationDetailView.as_view(),
        name="api-conversation-detail",
    ),
    path("api/projects/", ProjectCreateView.as_view(), name="api-projects"),
    path(
        "api/projects/<int:project_id>/knowledge-bases/",
        KnowledgeBaseCreateView.as_view(),
        name="api-knowledge-bases",
    ),
    path(
        "api/projects/<int:project_id>/documents/",
        KnowledgeDocumentCreateView.as_view(),
        name="api-documents",
    ),
    path(
        "api/projects/<int:project_id>/search/",
        ProjectSearchView.as_view(),
        name="api-project-search",
    ),
    path(
        "api/projects/<int:project_id>/assistant/chat/",
        AssistantChatView.as_view(),
        name="api-project-assistant-chat",
    ),
    path(
        "api/projects/<int:project_id>/assistant/conversation/",
        AssistantConversationView.as_view(),
        name="api-project-assistant-conversation",
    ),
    path(
        "api/pipelines/",
        PipelineListCreateView.as_view(),
        name="api-pipelines",
    ),
    path(
        "api/pipelines/<int:pk>/",
        PipelineDetailView.as_view(),
        name="api-pipeline-detail",
    ),
    path(
        "api/pipelines/<int:pipeline_id>/run/",
        PipelineRunView.as_view(),
        name="api-pipeline-run",
    ),
    path(
        "api/tasks/<int:pk>/",
        TaskDetailView.as_view(),
        name="api-task-detail",
    ),
    path(
        "api/mcp/servers/",
        MCPServerListCreateView.as_view(),
        name="api-mcp-servers",
    ),
    path(
        "api/mcp/servers/<int:pk>/",
        MCPServerDetailView.as_view(),
        name="api-mcp-server-detail",
    ),
    path(
        "api/mcp/servers/<int:server_id>/tools/",
        MCPServerToolsListView.as_view(),
        name="api-mcp-server-tools",
    ),
    path(
        "api/mcp/servers/<int:server_id>/sync-tools/",
        MCPServerSyncToolsView.as_view(),
        name="api-mcp-server-sync-tools",
    ),
    path(
        "api/mcp/tools/<int:tool_id>/invoke/",
        MCPToolInvokeView.as_view(),
        name="api-mcp-tool-invoke",
    ),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
