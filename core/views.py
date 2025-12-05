import json
import mimetypes
import os
from typing import Any
from types import SimpleNamespace

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from django.shortcuts import render
from openai import OpenAI

try:  # безопасный импорт для разных версий SDK
    from openai import AuthenticationError, NotFoundError, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    AuthenticationError = NotFoundError = OpenAIError = Exception
from pgvector.django import CosineDistance
from rest_framework import generics, status
from rest_framework.decorators import api_view
from rest_framework.exceptions import ValidationError
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .mcp_client import MCPClientError, call_tool
from .services.mcp_tools import sync_tools_for_server
from .services import rag_agent
from .services.rag_agent import RAGAgentError
from .services.knowledge_extraction import (
    is_manual_extraction_request,
    maybe_schedule_knowledge_extractor_for_agent,
    schedule_knowledge_extractor,
)
from .constants import (
    KNOWLEDGE_EXTRACTOR_SLUG,
    RAG_LIBRARIAN_SLUG,
    WEB_KNOWLEDGE_MCP_SLUG,
    WEB_KNOWLEDGE_TOOL_NAME,
)
from command_center.llm_registry import (
    get_chat_primary,
    get_chat_recommended,
    get_embedding_default,
    load_registry,
)
from command_center.services.llm_models import requires_responses_api
from command_center.tasks import sync_llm_registry_task
from .models import (
    Agent,
    AgentMemory,
    AgentServerBinding,
    Conversation,
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeEmbedding,
    KnowledgeEdge,
    KnowledgeNode,
    MCPServer,
    MCPTool,
    MemoryEvent,
    Message,
    MessageAttachment,
    Pipeline,
    Project,
    RetrievalLog,
    Task,
)
from .serializers import (
    AgentSerializer,
    ConversationDetailSerializer,
    ConversationSerializer,
    KnowledgeBaseSerializer,
    KnowledgeDocumentSerializer,
    MCPServerSerializer,
    MCPToolSerializer,
    MessageSerializer,
    PipelineSerializer,
    ProjectSerializer,
    TaskSerializer,
)
from .embeddings import embed_text
from .memory_extractor import extract_memory_events
from .services.agent_context import build_agent_llm_messages
from .services.rag_delegate import get_rag_delegate
from .services.web_knowledge_ingest import persist_web_knowledge_documents
from .tasks import (
    generate_memory_embedding_for_event,
    process_knowledge_document,
    ingest_project_docs,
)

import logging


logger = logging.getLogger(__name__)


def _resolve_projects_and_current(request):
    projects = Project.objects.all().order_by("id")
    current_project = None

    project_param = request.GET.get("project")
    if project_param:
        try:
            current_project = projects.get(id=project_param)
        except Project.DoesNotExist:
            current_project = None

    if current_project is None and projects.exists():
        current_project = projects.first()

    return projects, current_project


def _get_agents_for_project(project):
    agents = []
    initial_agent = None

    if project is not None:
        agents = list(
            Agent.objects.available(project=project)
            .exclude(slug=KNOWLEDGE_EXTRACTOR_SLUG)
            .order_by("-is_primary", "name")
        )
        for agent in agents:
            if agent.is_primary:
                initial_agent = agent
                break
        if initial_agent is None and agents:
            initial_agent = agents[0]

    return agents, initial_agent


@api_view(["GET"])
def health_check(request):
    return Response({"status": "ok"})


class LLMRegistryView(APIView):
    """
    Read-only эндпоинт для выдачи реестра LLM-моделей.
    """

    def get(self, request):
        registry = load_registry()
        chat_section = registry.get("chat") or {}
        embedding_section = registry.get("embedding") or {}

        response_data = {
            "chat": chat_section.get("models") or [],
            "embedding": embedding_section.get("models") or [],
            "lightweight": registry.get("lightweight") or [],
            "realtime": registry.get("realtime") or [],
            "search": registry.get("search") or [],
            "deprecated": registry.get("deprecated") or [],
            "chat_primary": get_chat_primary(),
            "embedding_default": get_embedding_default(),
            "chat_recommended": get_chat_recommended(),
        }
        return Response(response_data)


class LLMRegistrySyncView(APIView):
    """
    Ручной запуск синхронизации реестра LLM-моделей из UI.
    Доступ только для админов.
    """

    permission_classes = [IsAdminUser]

    def post(self, request):
        sync_llm_registry_task.delay()
        return Response(
            {
                "status": "scheduled",
                "message": "LLM registry sync task has been scheduled.",
            }
        )


@login_required
def dashboard_view(request):
    projects, current_project = _resolve_projects_and_current(request)
    agents, initial_agent = _get_agents_for_project(current_project)

    mcp_servers = MCPServer.objects.all().order_by("name")
    mcp_tools = MCPTool.objects.select_related("server").all().order_by(
        "server__name", "name"
    )

    pipelines = []

    if current_project is not None:
        pipelines = list(
            Pipeline.objects.select_related("owner_agent")
            .filter(project=current_project)
            .order_by("name")
        )
        rag_meta = {
            "last_full_sync": current_project.rag_last_full_sync_at,
            "last_error": current_project.rag_last_error_at,
            "error_count": current_project.rag_error_count,
        }
    else:
        rag_meta = None

    # данные LLM реестра для UI
    llm_chat_primary = get_chat_primary()
    llm_embedding_default = get_embedding_default()

    rag_agent_entry = Agent.objects.filter(slug=RAG_LIBRARIAN_SLUG).first()

    context = {
        "projects": projects,
        "current_project": current_project,
        "agents": agents,
        "initial_agent": initial_agent,
        "mcp_servers": mcp_servers,
        "mcp_tools": mcp_tools,
        "pipelines": pipelines,
        "llm_chat_primary": llm_chat_primary,
        "llm_embedding_default": llm_embedding_default,
        "rag_agent_id": rag_agent_entry.id if rag_agent_entry else None,
        "current_project_rag_meta": rag_meta,
    }
    return render(request, "core/dashboard.html", context)


@login_required
def memory_view(request):
    projects, current_project = _resolve_projects_and_current(request)
    agents, initial_agent = _get_agents_for_project(current_project)

    memory_events = []
    rag_meta = None
    if current_project is not None:
        memory_events = list(
            MemoryEvent.objects.filter(project=current_project)
            .order_by("-created_at")[:50]
        )
        rag_meta = {
            "last_full_sync": current_project.rag_last_full_sync_at,
            "last_error": current_project.rag_last_error_at,
            "error_count": current_project.rag_error_count,
        }

    rag_agent_entry = Agent.objects.filter(slug=RAG_LIBRARIAN_SLUG).first()

    context = {
        "projects": projects,
        "current_project": current_project,
        "agents": agents,
        "initial_agent": initial_agent,
        "memory_events": memory_events,
        "rag_agent_id": rag_agent_entry.id if rag_agent_entry else None,
        "current_project_rag_meta": rag_meta,
    }
    return render(request, "core/memory.html", context)


class ProjectCreateView(generics.CreateAPIView):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer


class KnowledgeBaseCreateView(generics.CreateAPIView):
    serializer_class = KnowledgeBaseSerializer

    def get_queryset(self):
        return KnowledgeBase.objects.filter(project_id=self.kwargs["project_id"])

    def perform_create(self, serializer):
        project = generics.get_object_or_404(Project, pk=self.kwargs["project_id"])
        serializer.save(project=project)


class KnowledgeDocumentCreateView(generics.CreateAPIView):
    serializer_class = KnowledgeDocumentSerializer

    def get_queryset(self):
        return KnowledgeDocument.objects.filter(
            knowledge_base__project_id=self.kwargs["project_id"]
        )

    def perform_create(self, serializer):
        project_id = self.kwargs["project_id"]
        knowledge_base = serializer.validated_data["knowledge_base"]

        if knowledge_base.project_id != project_id:
            raise ValidationError(
                {"knowledge_base": "Knowledge base does not belong to this project."}
            )

        uploaded_file = self.request.FILES.get("file")
        mime_type = ""

        if uploaded_file is not None:
            mime_type = getattr(uploaded_file, "content_type", "") or ""

        if not mime_type and uploaded_file is not None:
            guessed_type, _ = mimetypes.guess_type(uploaded_file.name)
            mime_type = guessed_type or ""

        document = serializer.save(mime_type=mime_type)

        process_knowledge_document.delay(document.id)


class ProjectSearchView(APIView):
    def get(self, request, project_id: int):
        query = request.query_params.get("q")
        if not query:
            raise ValidationError({"q": "This query parameter is required."})

        top_k_param = request.query_params.get("top_k", "5")
        try:
            top_k = int(top_k_param)
        except ValueError:
            raise ValidationError({"top_k": "Must be an integer."})

        if top_k <= 0:
            raise ValidationError({"top_k": "Must be greater than 0."})

        if top_k > 50:
            top_k = 50

            query_embedding = embed_text(query)

        embeddings_qs = KnowledgeEmbedding.objects.filter(
            chunk__project_id=project_id,
        ).filter(
            Q(chunk__document__status=KnowledgeDocument.STATUS_READY)
            | Q(chunk__document__isnull=True, chunk__source__isnull=False)
        ).annotate(
            distance=CosineDistance("embedding", query_embedding)
        ).order_by("distance")[:top_k]

        results = []
        for embedding in embeddings_qs:
            chunk = embedding.chunk
            document = chunk.document
            knowledge_base = document.knowledge_base if document else None
            source = chunk.source

            results.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": document.id if document else None,
                    "knowledge_base_id": knowledge_base.id if knowledge_base else None,
                    "project_id": (
                        knowledge_base.project_id if knowledge_base else chunk.project_id
                    ),
                    "source_id": str(source.id) if source else None,
                    "source_path": source.path if source else None,
                    "text": chunk.text,
                    "score": float(getattr(embedding, "distance", 0.0)),
                    "meta": chunk.meta or {},
                }
            )

        return Response(results)


class AgentListView(generics.ListCreateAPIView):
    serializer_class = AgentSerializer

    def get_queryset(self):
        project_id = self.request.query_params.get("project")
        project = Project.objects.filter(pk=project_id).first() if project_id else None
        return Agent.objects.available(project=project).order_by("-is_primary", "name")


class AgentDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Agent.objects.all()
    serializer_class = AgentSerializer


class AgentInvokeView(APIView):
    def post(self, request, agent_id: int):
        agent = generics.get_object_or_404(Agent, pk=agent_id, is_active=True)

        message = request.data.get("message")
        if not message or not isinstance(message, str):
            raise ValidationError({"message": "This field is required and must be a string."})

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return Response(
                {
                    "agent_id": agent.id,
                    "error": {
                        "type": "openai_error",
                        "message": "OPENAI_API_KEY is not set",
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        client = OpenAI(api_key=api_key)

        system_prompt = agent.system_prompt or ""
        model_name = agent.resolved_model_name

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        request_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": agent.temperature,
            "timeout": 30,
        }
        if isinstance(agent.max_tokens, int) and agent.max_tokens > 0:
            request_kwargs["max_tokens"] = agent.max_tokens

        try:
            completion = client.chat.completions.create(**request_kwargs)
            reply = completion.choices[0].message.content or ""
        except NotFoundError as exc:
            logger.warning(
                "OpenAI model_not_found for agent %s: model_name=%s resolved=%s error=%s",
                agent.id,
                agent.model_name,
                agent.resolved_model_name,
                exc,
            )
            user_message = (
                f"Выбранная модель ('{agent.resolved_model_name}') больше не доступна в OpenAI. "
                "Зайдите в настройки агента и выберите другую модель из списка."
            )
            return Response(
                {
                    "agent_id": agent.id,
                    "error": {
                        "type": "model_not_found",
                        "message": user_message,
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except AuthenticationError as exc:
            logger.error("OpenAI authentication error for agent %s: %s", agent.id, exc)
            return Response(
                {
                    "agent_id": agent.id,
                    "error": {
                        "type": "auth_error",
                        "message": "Проблема с ключом OpenAI. Проверьте настройки сервера.",
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except OpenAIError as exc:
            logger.error("OpenAI error for agent %s: %s", agent.id, exc)
            return Response(
                {
                    "agent_id": agent.id,
                    "error": {
                        "type": "openai_error",
                        "message": "Сейчас не удалось связаться с LLM. Попробуйте ещё раз позже.",
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        available_tools = []
        bindings = (
            AgentServerBinding.objects.select_related("server")
            .prefetch_related("allowed_tools")
            .filter(agent=agent)
        )

        for binding in bindings:
            server = binding.server
            allowed_qs = binding.allowed_tools.filter(is_active=True)
            if allowed_qs.exists():
                tools_qs = allowed_qs
            else:
                tools_qs = server.tools.filter(is_active=True)

            tool_names = list(tools_qs.values_list("name", flat=True))
            available_tools.append(
                {
                    "server_id": server.id,
                    "server_name": server.name,
                    "tools": tool_names,
                }
            )

        return Response(
            {
                "agent_id": agent.id,
                "model": model_name,
                "reply": reply,
                "available_tools": available_tools,
            }
        )


class AgentMCPAccessView(APIView):
    """
    Возвращает и обновляет доступ агента к MCP-инструментам.
    """

    def _serialize_binding(self, binding: AgentServerBinding) -> Dict[str, Any]:
        server = binding.server
        tools = server.tools.filter(is_active=True).order_by("name")
        allowed_ids = list(binding.allowed_tools.values_list("id", flat=True))
        return {
            "server": {
                "id": server.id,
                "name": server.name,
            },
            "all_allowed": len(allowed_ids) == 0,
            "allowed_tool_ids": allowed_ids,
            "tools": [
                {
                    "id": tool.id,
                    "name": tool.name,
                    "description": tool.description or "",
                }
                for tool in tools
            ],
        }

    def get(self, request, agent_id: int):
        agent = generics.get_object_or_404(Agent, pk=agent_id, is_active=True)
        bindings = (
            AgentServerBinding.objects.select_related("server")
            .prefetch_related("allowed_tools", "server__tools")
            .filter(agent=agent)
        )
        data = [self._serialize_binding(binding) for binding in bindings]
        return Response({"agent_id": agent.id, "bindings": data})

    def post(self, request, agent_id: int):
        agent = generics.get_object_or_404(Agent, pk=agent_id, is_active=True)
        server_id = request.data.get("server_id")
        if server_id is None:
            raise ValidationError({"server_id": "This field is required."})

        try:
            server = MCPServer.objects.get(id=int(server_id))
        except (ValueError, MCPServer.DoesNotExist) as exc:  # noqa: PERF203
            raise ValidationError({"server_id": "Server not found."}) from exc

        binding, _ = AgentServerBinding.objects.get_or_create(agent=agent, server=server)
        allowed_tool_ids = request.data.get("allowed_tool_ids", [])
        if allowed_tool_ids in (None, []):
            binding.allowed_tools.clear()
        else:
            if not isinstance(allowed_tool_ids, list):
                raise ValidationError({"allowed_tool_ids": "Must be a list of tool IDs."})
            try:
                allowed_ids_int = [int(tool_id) for tool_id in allowed_tool_ids]
            except (TypeError, ValueError) as exc:  # noqa: PERF203
                raise ValidationError({"allowed_tool_ids": "Must contain integer IDs."}) from exc
            tools_qs = server.tools.filter(is_active=True, id__in=allowed_ids_int)
            if tools_qs.count() != len(set(allowed_ids_int)):
                raise ValidationError({"allowed_tool_ids": "Contains invalid IDs for this server."})
            binding.allowed_tools.set(tools_qs)

        binding.refresh_from_db()
        return Response(self._serialize_binding(binding))


class AgentMemoryListView(APIView):
    """
    Список последних элементов долгосрочной памяти выбранного агента.
    """

    def get(self, request, agent_id: int):
        agent = generics.get_object_or_404(Agent, pk=agent_id, is_active=True)
        limit_param = request.query_params.get("limit", "20")
        try:
            limit = int(limit_param)
        except ValueError as exc:  # noqa: PERF203
            raise ValidationError({"limit": "Must be an integer."}) from exc
        if limit <= 0:
            limit = 1
        if limit > 100:
            limit = 100

        memories_qs = AgentMemory.objects.filter(agent=agent).order_by("-updated_at")[:limit]
        data = [
            {
                "id": str(memory.id),
                "content": memory.content,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            }
            for memory in memories_qs
        ]
        return Response(
            {
                "agent_id": agent.id,
                "memories": data,
            }
        )


class AgentGraphNodesView(APIView):
    def get(self, request, agent_id: int):
        agent = generics.get_object_or_404(Agent, pk=agent_id, is_active=True)
        limit_param = request.query_params.get("limit", "100")
        pinned_only = request.query_params.get("pinned") in {"1", "true", "True"}
        try:
            limit = int(limit_param)
        except ValueError as exc:  # noqa: PERF203
            raise ValidationError({"limit": "Must be an integer."}) from exc
        limit = max(1, min(limit, 500))

        nodes_qs = KnowledgeNode.objects.filter(agent=agent)
        if pinned_only:
            nodes_qs = nodes_qs.filter(is_pinned=True)
        nodes_qs = nodes_qs.order_by("-updated_at")[:limit]
        data = [
            {
                "id": str(node.id),
                "label": node.label,
                "type": node.type,
                "description": node.description,
                "object_type": node.object_type,
                "object_id": node.object_id,
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
            }
            for node in nodes_qs
        ]
        return Response({"agent_id": agent.id, "nodes": data})


class AgentGraphEdgesView(APIView):
    def get(self, request, agent_id: int):
        agent = generics.get_object_or_404(Agent, pk=agent_id, is_active=True)
        limit_param = request.query_params.get("limit", "200")
        pinned_only = request.query_params.get("pinned") in {"1", "true", "True"}
        try:
            limit = int(limit_param)
        except ValueError as exc:  # noqa: PERF203
            raise ValidationError({"limit": "Must be an integer."}) from exc
        limit = max(1, min(limit, 1000))

        edges_qs = KnowledgeEdge.objects.filter(agent=agent)
        if pinned_only:
            edges_qs = edges_qs.filter(source__is_pinned=True, target__is_pinned=True)
        edges_qs = edges_qs.select_related("source", "target").order_by("-created_at")[:limit]
        data = [
            {
                "id": str(edge.id),
                "source_label": edge.source.label,
                "target_label": edge.target.label,
                "relation": edge.relation,
                "description": edge.description,
                "weight": edge.weight,
                "created_at": edge.created_at.isoformat(),
            }
            for edge in edges_qs
        ]
        return Response({"agent_id": agent.id, "edges": data})


class ConversationDetailView(generics.RetrieveAPIView):
    queryset = Conversation.objects.all()
    serializer_class = ConversationDetailSerializer


class AssistantChatView(APIView):
    MAX_TOOL_ITERATIONS = max(0, getattr(settings, "AGENT_TOOL_MAX_ITERATIONS", 10))
    parser_classes = [JSONParser, FormParser, MultiPartParser]

    def _save_message_attachments(self, message: Message, files: list) -> list[MessageAttachment]:
        saved: list[MessageAttachment] = []
        for uploaded in files or []:
            if not uploaded:
                continue
            mime_type = getattr(uploaded, "content_type", "") or mimetypes.guess_type(uploaded.name)[0] or ""
            attachment = MessageAttachment.objects.create(
                message=message,
                file=uploaded,
                original_name=(uploaded.name or "")[:255],
                mime_type=mime_type or "",
                size=getattr(uploaded, "size", 0) or 0,
            )
            saved.append(attachment)
        return saved

    @staticmethod
    def _build_memory_trace(retrieved_memories: list[dict]) -> dict | None:
        if not retrieved_memories:
            return None
        return {
            "tool_call_id": "memory",
            "function": "memory_recall",
            "payload": {
                "memories": retrieved_memories,
            },
        }

    @staticmethod
    def _build_graph_trace(graph_nodes: list[dict]) -> dict | None:
        if not graph_nodes:
            return None
        return {
            "tool_call_id": "graph",
            "function": "graph_recall",
            "payload": {
                "nodes": graph_nodes,
            },
        }

    @staticmethod
    def _build_rag_trace(rag_docs: list[dict]) -> dict | None:
        if not rag_docs:
            return None
        return {
            "tool_call_id": "rag",
            "function": "rag_recall",
            "payload": {
                "documents": rag_docs,
            },
        }

    @classmethod
    def _prepend_recall_traces(
        cls,
        tool_traces: list[dict],
        retrieved_memories: list[dict],
        graph_nodes: list[dict],
        rag_docs: list[dict],
    ) -> list[dict]:
        traces = list(tool_traces or [])
        recall_traces = []
        for trace in (
            cls._build_memory_trace(retrieved_memories),
            cls._build_graph_trace(graph_nodes),
            cls._build_rag_trace(rag_docs),
        ):
            if trace:
                recall_traces.append(trace)
        if recall_traces:
            traces = recall_traces + traces
        return traces

    def _build_mcp_tools(self, agent: Agent):
        tool_definitions: list[dict] = []
        lookup: dict[str, tuple[MCPServer, MCPTool]] = {}

        bindings = (
            AgentServerBinding.objects.select_related("server")
            .prefetch_related("allowed_tools")
            .filter(agent=agent, server__is_active=True)
        )

        for binding in bindings:
            server = binding.server
            allowed_qs = binding.allowed_tools.filter(is_active=True)
            if allowed_qs.exists():
                tools_qs = allowed_qs
            else:
                tools_qs = server.tools.filter(is_active=True)

            for tool in tools_qs:
                function_name = f"mcp_tool_{tool.id}"
                description = tool.description.strip() if tool.description else ""
                if description:
                    desc_text = f"[{server.name}] {description}"
                else:
                    desc_text = f"[{server.name}] MCP tool '{tool.name}'"

                parameters = tool.input_schema or {}
                if not isinstance(parameters, dict):
                    parameters = {}
                parameters = dict(parameters)
                if not parameters:
                    parameters = {"type": "object", "properties": {}}
                parameters.setdefault("type", "object")
                parameters.setdefault("properties", {})

                tool_definitions.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": desc_text,
                            "parameters": parameters,
                        },
                    }
                )
                lookup[function_name] = (server, tool)

        return tool_definitions, lookup

    def _build_delegate_tools(self, agent: Agent):
        tool_definitions: list[dict] = []
        lookup: dict[str, Agent] = {}

        delegates = agent.delegates.filter(is_active=True)
        for delegate in delegates:
            function_name = f"delegate_to_agent_{delegate.id}"
            tool_definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": (
                            f"Делегировать задачу агенту «{delegate.name}». "
                            "Используй этот инструмент для задач в зоне ответственности этого агента."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "Текст задачи, которую нужно выполнить.",
                                }
                            },
                            "required": ["task"],
                        },
                    },
                }
            )
            lookup[function_name] = delegate

        return tool_definitions, lookup

    def _should_auto_delegate(self, text: str) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        keywords = [
            "rag",
            "библиотекар",
            "docs",
            "/docs",
            "докум",
            "проиндекс",
            "индекс",
            "обнови зн",
            "knowledge base",
            "change",
            "измен",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _detect_rag_command(self, text: str) -> tuple[str, str | None]:
        lowered = (text or "").lower()
        if "ошиб" in lowered or "error" in lowered:
            return "errors", None
        if any(word in lowered for word in ("обнов", "проиндекс", "перезап", "reindex")):
            if "файл" in lowered or rag_agent.UUID_RE.search(text or ""):
                return "reindex_single", text
            return "reindex_all", None
        if "change" in lowered or "измен" in lowered:
            return "changes", None
        if any(word in lowered for word in ("статус", "список", "документ", "покажи", "что есть")):
            return "status", None
        return "status", None

    def _handle_rag_librarian(self, project: Project, message_text: str) -> tuple[str, list[dict]]:
        if project is None:
            return "Мне нужен выбранный проект, чтобы управлять документацией.", []
        project_slug = project.slug
        action, hint = self._detect_rag_command(message_text)
        traces: list[dict] = [
            {
                "tool_call_id": f"rag::{action}",
                "function": f"rag_{action}",
                "payload": {"project_slug": project_slug},
            }
        ]
        try:
            if action == "errors":
                errors = rag_agent.get_error_sources(project_slug)
                if not errors:
                    reply = f"В проекте {project_slug} нет источников со статусом error."
                else:
                    lines = [f"⚠ Ошибки документации ({len(errors)}):"]
                    for item in errors:
                        lines.append(
                            f"- {item.get('relative_path') or item.get('path')}: {item.get('error_message') or item.get('last_error')}"
                        )
                    reply = "\n".join(lines)
            elif action == "reindex_all":
                reply = rag_agent.reindex_all(project_slug)
            elif action == "reindex_single":
                source = rag_agent.find_source_by_hint(project_slug, hint or message_text)
                if not source:
                    reply = "Не смог найти файл по описанию. Укажи точный путь или UUID источника."
                else:
                    reply = rag_agent.reindex_single(project_slug, str(source.get("id")))
            elif action == "changes":
                entries = rag_agent.get_changelog(project_slug)
                if not entries:
                    reply = "Изменений в документации пока нет."
                else:
                    lines = [f"Последние изменения (проект {project_slug}):"]
                    for item in entries:
                        summary = item.get("semantic_summary") or ""
                        lines.append(
                            f"- {item.get('source_path')} — {item.get('change_type')} ({summary})"
                        )
                    reply = "\n".join(lines)
            else:
                reply = rag_agent.summarize_sources(project_slug)
        except RAGAgentError as exc:
            logger.error("RAG Librarian API call failed: %s", exc)
            reply = f"Не удалось обратиться к RAG API: {exc}"
        return reply, traces

    def _run_delegate_agent_once(
        self,
        delegate: Agent,
        project: Project,
        conversation: Conversation,
        task: str,
        client: OpenAI,
    ) -> tuple[str, list[dict], RetrievalLog | None, list[dict]]:
        (
            messages,
            retrieval_log,
            retrieved_memories,
            graph_nodes_info,
            rag_docs_info,
        ) = build_agent_llm_messages(
            agent=delegate,
            conversation=conversation,
            latest_user_text=task,
            latest_user_message=None,
        )
        if delegate.slug == RAG_LIBRARIAN_SLUG:
            reply, traces = self._handle_rag_librarian(project, task)
            return reply, traces, None, [], [], []
        mcp_tool_defs, mcp_tool_lookup = self._build_mcp_tools(delegate)
        delegate_tool_defs, delegate_lookup = self._build_delegate_tools(delegate)
        tool_definitions = mcp_tool_defs + delegate_tool_defs
        require_tools = bool(tool_definitions) and delegate.tool_mode == Agent.ToolMode.REQUIRED

        if requires_responses_api(delegate.resolved_model_name):
            reply, traces = self._chat_with_responses(
                client=client,
                base_messages=messages,
                agent=delegate,
                model_name=delegate.resolved_model_name,
                temperature=delegate.temperature,
                max_tokens=delegate.max_tokens,
                tool_definitions=tool_definitions,
                mcp_tool_lookup=mcp_tool_lookup,
                delegate_lookup=delegate_lookup,
                project=project,
                conversation=conversation,
                require_tools=require_tools,
            )
        else:
            reply, traces = self._chat_with_tools(
                client=client,
                base_messages=messages,
                agent=delegate,
                model_name=delegate.resolved_model_name,
                temperature=delegate.temperature,
                max_tokens=delegate.max_tokens,
                tool_definitions=tool_definitions,
                mcp_tool_lookup=mcp_tool_lookup,
                delegate_lookup=delegate_lookup,
                project=project,
                conversation=conversation,
            )
        traces = self._prepend_recall_traces(traces, retrieved_memories, graph_nodes_info, rag_docs_info)
        try:
            extract_memory_events(
                client=client,
                model_name=delegate.resolved_model_name,
                agent=delegate,
                project=project,
                conversation=conversation,
                user_message=task,
                assistant_message=reply,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Memory extraction failed for delegate agent %s in conversation %s: %s",
                delegate.id,
                conversation.id,
                exc,
            )
        maybe_schedule_knowledge_extractor_for_agent(
            project=project,
            agent=delegate,
            text=reply,
            source=f"agent:{delegate.slug}",
        )
        return reply, traces, retrieval_log, retrieved_memories, graph_nodes_info, rag_docs_info

    def _handle_tool_call(
        self,
        tool_call,
        mcp_tool_lookup: dict[str, tuple[MCPServer, MCPTool]],
        delegate_lookup: dict[str, Agent],
        project: Project,
        conversation: Conversation,
        client: OpenAI,
    ) -> dict:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None) or ""
        arguments_raw = getattr(function, "arguments", "") or ""
        try:
            arguments = json.loads(arguments_raw) if arguments_raw else {}
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Не удалось распарсить аргументы инструмента.",
                "raw_arguments": arguments_raw,
            }

        if name in mcp_tool_lookup:
            server, tool = mcp_tool_lookup[name]
            arguments = dict(arguments or {})
            is_web_knowledge_call = (
                project is not None
                and server.slug == WEB_KNOWLEDGE_MCP_SLUG
                and tool.name == WEB_KNOWLEDGE_TOOL_NAME
            )
            if is_web_knowledge_call:
                arguments.setdefault("project_slug", project.slug)
            try:
                result = call_tool(server, tool, arguments or {})
                logger.info(
                    "Tool call succeeded: server=%s tool=%s function=%s args=%s",
                    server.name,
                    tool.name,
                    name,
                    arguments,
                )
                stored_documents = []
                if is_web_knowledge_call and project is not None:
                    try:
                        stored_documents = persist_web_knowledge_documents(
                            project=project,
                            payload=result or {},
                            query=arguments.get("query"),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Failed to persist Web Knowledge docs: %s", exc)
                        stored_documents = []
                    else:
                        if stored_documents:
                            ingest_project_docs.delay(project.id)

                payload_result: dict[str, Any] = result or {}
                if stored_documents:
                    payload_result = dict(payload_result)
                    payload_result["stored_documents"] = stored_documents

                return {
                    "status": "ok",
                    "type": "mcp",
                    "server": server.name,
                    "tool": tool.name,
                    "result": payload_result,
                }
            except MCPClientError as exc:
                logger.warning(
                    "Tool call failed: server=%s tool=%s function=%s error=%s",
                    server.name,
                    tool.name,
                    name,
                    exc,
                )
                return {
                    "status": "error",
                    "type": "mcp",
                    "server": server.name,
                    "tool": tool.name,
                    "message": str(exc),
                    "code": exc.code,
                }

        if name in delegate_lookup:
            delegate = delegate_lookup[name]
            task = arguments.get("task")
            if not isinstance(task, str) or not task.strip():
                return {
                    "status": "error",
                    "type": "delegate",
                    "delegate": delegate.name,
                    "message": "Поле 'task' обязательно и должно быть непустой строкой.",
                }
            try:
                (
                    response,
                    delegate_traces,
                    retrieval_log,
                    retrieved_memories,
                    graph_nodes_info,
                    rag_docs_info,
                ) = self._run_delegate_agent_once(
                    delegate,
                    project,
                    conversation,
                    task,
                    client,
                )
                logger.info(
                    "Delegate tool call succeeded: delegate=%s function=%s task=%s",
                    delegate.name,
                    name,
                    task,
                )
                payload: dict[str, Any] = {
                    "status": "ok",
                    "type": "delegate",
                    "delegate": delegate.name,
                    "response": response,
                    "tool_traces": delegate_traces,
                }
                if retrieval_log is not None:
                    payload["retrieval_log_id"] = str(retrieval_log.id)
                if retrieved_memories:
                    payload["retrieved_memories"] = retrieved_memories
                if graph_nodes_info:
                    payload["graph_nodes"] = graph_nodes_info
                if rag_docs_info:
                    payload["rag_documents"] = rag_docs_info
                return payload
            except OpenAIError as exc:
                logger.error(
                    "Delegate agent %s failed: %s",
                    delegate.id,
                    exc,
                )
                return {
                    "status": "error",
                    "type": "delegate",
                    "delegate": delegate.name,
                    "message": f"Ошибка при вызове агента '{delegate.name}': {exc}",
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error while running delegate agent %s", delegate.id)
                return {
                    "status": "error",
                    "type": "delegate",
                    "delegate": delegate.name,
                    "message": f"Не удалось выполнить задачу делегату: {exc}",
                }

        return {
            "status": "error",
            "message": f"Неизвестный инструмент: {name}",
        }

    def _chat_with_tools(
        self,
        client: OpenAI,
        base_messages: list[dict],
        agent: Agent,
        model_name: str,
        temperature: float,
        max_tokens: int | None,
        tool_definitions: list[dict],
        mcp_tool_lookup: dict[str, tuple[MCPServer, MCPTool]],
        delegate_lookup: dict[str, Agent],
        project: Project,
        conversation: Conversation,
    ) -> tuple[str, list[dict]]:
        base_kwargs = {
            "model": model_name,
            "temperature": temperature,
            "timeout": 30,
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            base_kwargs["max_tokens"] = max_tokens
        require_tools = bool(tool_definitions) and agent.tool_mode == Agent.ToolMode.REQUIRED

        if tool_definitions:
            base_kwargs["tools"] = tool_definitions
            base_kwargs["tool_choice"] = "auto"

        messages = list(base_messages)
        tool_traces: list[dict] = []
        iterations = 0
        used_tools = False

        while True:
            request_kwargs = dict(base_kwargs)
            request_kwargs["messages"] = messages
            completion = client.chat.completions.create(**request_kwargs)
            choice = completion.choices[0].message
            reply_text = choice.content or ""
            tool_calls = getattr(choice, "tool_calls", None) or []

            if require_tools and not tool_calls and not used_tools:
                logger.warning(
                    "Agent %s in REQUIRED tool_mode returned no tool_calls despite available tools",
                    agent.id,
                )
                warning_reply = (
                    "Модель не вызвала ни одного инструмента, хотя должна была. "
                    "Попробуйте повторить запрос или уточнить задачу."
                )
                return warning_reply, tool_traces

            if not tool_calls:
                return reply_text, tool_traces

            assistant_entry = {
                "role": "assistant",
                "content": reply_text or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", ""),
                        },
                    }
                    for tc in tool_calls
                ],
            }
            messages.append(assistant_entry)

            for tc in tool_calls:
                tool_payload = self._handle_tool_call(
                    tc,
                    mcp_tool_lookup,
                    delegate_lookup,
                    project,
                    conversation,
                    client,
                )
                tool_traces.append(
                    {
                        "tool_call_id": tc.id,
                        "function": getattr(tc.function, "name", ""),
                        "payload": tool_payload,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_payload, ensure_ascii=False),
                    }
                )
                used_tools = True

            iterations += 1
            if self.MAX_TOOL_ITERATIONS and iterations >= self.MAX_TOOL_ITERATIONS:
                limit_notice = "\n\n[system] Достигнут лимит вызовов инструментов."
                return reply_text + limit_notice, tool_traces

    def _convert_messages_for_responses(self, base_messages: list[dict]) -> list[dict]:
        converted: list[dict] = []
        type_by_role = {
            "system": "input_text",
            "user": "input_text",
            "assistant": "output_text",
            "tool": "output_text",
        }

        for message in base_messages:
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = message.get("content")
            content_type = type_by_role.get(role, "input_text")
            if isinstance(content, list):
                converted_content = []
                for item in content:
                    if isinstance(item, dict):
                        new_item = dict(item)
                        if new_item.get("type") == "text":
                            new_item["type"] = content_type
                        converted_content.append(new_item)
                    else:
                        converted_content.append(item)
                converted.append({"role": role, "content": converted_content})
                continue
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            converted.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": content_type,
                            "text": text,
                        }
                    ],
                }
            )
        return converted

    def _convert_tool_definitions_for_responses(self, tool_definitions: list[dict]) -> list[dict]:
        converted: list[dict] = []
        for definition in tool_definitions or []:
            if definition.get("type") == "function":
                func = definition.get("function") or {}
                converted.append(
                    {
                        "type": "function",
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": func.get("parameters"),
                        "strict": func.get("strict"),
                    }
                )
            else:
                converted.append(definition)
        return converted

    def _sanitize_response_payload(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            cleaned = {}
            for key, value in payload.items():
                if key == "status":
                    continue
                cleaned[key] = self._sanitize_response_payload(value)
            return cleaned
        if isinstance(payload, list):
            return [self._sanitize_response_payload(item) for item in payload]
        return payload

    def _append_response_items_to_history(self, history: list[dict], response):
        for item in getattr(response, "output", []) or []:
            if hasattr(item, "model_dump"):
                history.append(self._sanitize_response_payload(item.model_dump()))
            else:
                history.append(self._sanitize_response_payload(item))

    def _parse_responses_output(self, response):
        text_parts: list[str] = []
        tool_calls: list[Any] = []
        for item in getattr(response, "output", []) or []:
            item_type = getattr(item, "type", "")
            if item_type == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", "") == "output_text":
                        text_parts.append(content.text or "")
            elif item_type in {"function_call", "function_tool_call"}:
                tool_calls.append(item)
        reply_text = "".join(text_parts).strip()
        if not reply_text:
            reply_text = getattr(response, "output_text", "") or ""
        return reply_text, tool_calls

    def _chat_with_responses(
        self,
        client: OpenAI,
        base_messages: list[dict],
        agent: Agent,
        model_name: str,
        temperature: float,
        max_tokens: int | None,
        tool_definitions: list[dict],
        mcp_tool_lookup: dict[str, tuple[MCPServer, MCPTool]],
        delegate_lookup: dict[str, Agent],
        project: Project,
        conversation: Conversation,
        require_tools: bool,
    ) -> tuple[str, list[dict]]:
        history_inputs = self._convert_messages_for_responses(base_messages)
        used_tools = False
        tool_traces: list[dict] = []
        iterations = 0

        responses_tools = self._convert_tool_definitions_for_responses(tool_definitions)

        while True:
            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "timeout": 30,
                "input": history_inputs,
            }
            if isinstance(max_tokens, int) and max_tokens > 0:
                request_kwargs["max_output_tokens"] = max_tokens
            if responses_tools:
                request_kwargs["tools"] = responses_tools
                request_kwargs["tool_choice"] = "auto"
                request_kwargs["parallel_tool_calls"] = True

            response = client.responses.create(**request_kwargs)
            self._append_response_items_to_history(history_inputs, response)

            reply_text, response_tool_calls = self._parse_responses_output(response)

            if not response_tool_calls:
                if require_tools and not used_tools:
                    warning_reply = (
                        "Модель не вызвала ни одного инструмента, хотя должна была. "
                        "Попробуйте повторить запрос или уточнить задачу."
                    )
                    return warning_reply, tool_traces
                return reply_text, tool_traces

            iterations += 1
            if self.MAX_TOOL_ITERATIONS and iterations >= self.MAX_TOOL_ITERATIONS:
                limit_notice = "\n\n[system] Достигнут лимит вызовов инструментов."
                return reply_text + limit_notice, tool_traces

            for tc in response_tool_calls:
                tc_id = getattr(tc, "call_id", "") or getattr(tc, "id", "")
                function_name = getattr(tc, "name", "")
                arguments = getattr(tc, "arguments", "") or "{}"

                proxy_tc = SimpleNamespace(
                    id=tc_id,
                    type="function",
                    function=SimpleNamespace(
                        name=function_name,
                        arguments=arguments,
                    ),
                )
                tool_payload = self._handle_tool_call(
                    proxy_tc,
                    mcp_tool_lookup,
                    delegate_lookup,
                    project,
                    conversation,
                    client,
                )
                used_tools = True
                tool_traces.append(
                    {
                        "tool_call_id": tc_id,
                        "function": function_name,
                        "payload": tool_payload,
                    }
                )
                tool_result_entry = {
                    "type": "function_call_output",
                    "call_id": tc_id,
                    "output": json.dumps(tool_payload, ensure_ascii=False),
                }
                history_inputs.append(tool_result_entry)

    def _finalize_response(
        self,
        *,
        conversation: Conversation,
        agent: Agent,
        user_message: Message,
        reply_text: str,
        tool_traces: list[dict],
        retrieval_log: RetrievalLog | None,
        retrieved_memories: list[dict],
        graph_nodes_info: list[dict],
        rag_docs_info: list[dict],
        client: OpenAI,
        model_name: str,
        request,
    ):
        tool_traces = self._prepend_recall_traces(
            tool_traces,
            retrieved_memories,
            graph_nodes_info,
            rag_docs_info,
        )

        assistant_meta = {}
        if tool_traces:
            assistant_meta["tool_traces"] = tool_traces
        if retrieval_log is not None:
            assistant_meta["retrieval_log_id"] = str(retrieval_log.id)
        if retrieved_memories:
            assistant_meta["retrieved_memories"] = retrieved_memories
        if graph_nodes_info:
            assistant_meta["graph_nodes"] = graph_nodes_info
        if rag_docs_info:
            assistant_meta["rag_documents"] = rag_docs_info

        assistant_message = Message.objects.create(
            conversation=conversation,
            sender=Message.ROLE_ASSISTANT,
            content=reply_text,
            meta=assistant_meta,
        )

        try:
            extract_memory_events(
                client=client,
                model_name=model_name,
                agent=agent,
                project=conversation.project,
                conversation=conversation,
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory extraction failed for conversation %s: %s", conversation.id, exc)

        maybe_schedule_knowledge_extractor_for_agent(
            project=conversation.project,
            agent=agent,
            text=reply_text,
            source=f"agent:{agent.slug}",
        )

        messages_data = MessageSerializer(
            [user_message, assistant_message],
            many=True,
            context={"request": request},
        ).data

        return Response(
            {
                "conversation_id": conversation.id,
                "agent_id": agent.id,
                "messages": messages_data,
            }
        )

    def post(self, request, project_id: int):
        project = generics.get_object_or_404(Project, pk=project_id)

        is_multipart = bool(request.content_type and request.content_type.startswith("multipart/"))
        attachment_files = request.FILES.getlist("attachments") if is_multipart else []

        message_text = request.data.get("message")
        if not isinstance(message_text, str):
            message_text = ""
        if not message_text.strip() and not attachment_files:
            raise ValidationError({"message": "Это поле обязательно."})

        agent_id = request.data.get("agent_id")
        conversation_id = request.data.get("conversation_id")

        agent = None
        if agent_id is not None:
            try:
                agent = Agent.objects.get(id=agent_id, is_active=True)
            except Agent.DoesNotExist:
                raise ValidationError({"agent_id": "Agent not found or inactive."})
            if agent.project_id not in (None, project.id):
                raise ValidationError(
                    {"agent_id": "Agent must be global or belong to the project."}
                )
        else:
            agent = (
                Agent.objects.filter(
                    project_id=project.id,
                    is_primary=True,
                    is_active=True,
                ).first()
                or Agent.objects.filter(
                    project__isnull=True,
                    is_primary=True,
                    is_active=True,
                ).first()
            )
            if agent is None:
                return Response(
                    {
                        "error": "no_primary_agent",
                        "message": "Для проекта нет назначенного primary-агента и глобального агента по умолчанию.",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        agent = Agent.objects.prefetch_related("delegates").get(pk=agent.id)

        conversation = None
        if conversation_id is not None:
            try:
                conversation = Conversation.objects.get(
                    id=conversation_id, project=project, agent=agent
                )
            except Conversation.DoesNotExist:
                raise ValidationError({"conversation_id": "Conversation not found."})
        else:
            conversation = Conversation.objects.create(
                project=project,
                agent=agent,
                title="",
            )

        user_message = Message.objects.create(
            conversation=conversation,
            sender=Message.ROLE_USER,
            content=message_text,
            meta={},
        )
        attachments = self._save_message_attachments(user_message, attachment_files)

        lowered = message_text.lower()
        should_remember = any(
            phrase in lowered
            for phrase in [
                "запомни",
                "запомнить",
                "меня зовут",
                "я люблю",
                "я предпочитаю",
                "учти",
                "обрати внимание",
            ]
        )

        if should_remember:
            memory_event = MemoryEvent.objects.create(
                agent=agent,
                project=project,
                conversation=conversation,
                message=user_message,
                type=MemoryEvent.TYPE_FACT,
                content=message_text,
                importance=1,
            )
            generate_memory_embedding_for_event.delay(memory_event.id)

        if agent.slug != KNOWLEDGE_EXTRACTOR_SLUG and is_manual_extraction_request(message_text):
            schedule_knowledge_extractor(
                project_id=project.id,
                text=message_text,
                agent_id=agent.id,
                source="user_request",
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return Response(
                {
                    "error": "openai_error",
                    "message": "OPENAI_API_KEY is not set",
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        client = OpenAI(api_key=api_key)
        model_name = agent.resolved_model_name

        if agent.slug == RAG_LIBRARIAN_SLUG:
            reply_text, tool_traces = self._handle_rag_librarian(project, message_text)
            return self._finalize_response(
                conversation=conversation,
                agent=agent,
                user_message=user_message,
                reply_text=reply_text,
                tool_traces=tool_traces,
                retrieval_log=None,
                retrieved_memories=[],
                graph_nodes_info=[],
                rag_docs_info=[],
                client=client,
                model_name=model_name,
                request=request,
            )

        rag_delegate = get_rag_delegate(agent)
        if rag_delegate and self._should_auto_delegate(message_text):
            try:
                (
                    delegate_reply,
                    delegate_traces,
                    delegate_retrieval_log,
                    delegate_memories,
                    delegate_graph_nodes,
                    delegate_rag_docs,
                ) = self._run_delegate_agent_once(
                    rag_delegate,
                    project,
                    conversation,
                    message_text,
                    client,
                )
                reply_text = f"RAG Librarian:\n{delegate_reply}".strip()
                return self._finalize_response(
                    conversation=conversation,
                    agent=agent,
                    user_message=user_message,
                    reply_text=reply_text,
                    tool_traces=delegate_traces,
                    retrieval_log=delegate_retrieval_log,
                    retrieved_memories=delegate_memories,
                    graph_nodes_info=delegate_graph_nodes,
                    rag_docs_info=delegate_rag_docs,
                    client=client,
                    model_name=model_name,
                    request=request,
                )
            except (NotFoundError, AuthenticationError, OpenAIError) as exc:
                logger.warning(
                    "Auto-delegation to RAG Librarian failed for agent %s: %s",
                    agent.id,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Unexpected error during auto-delegation for agent %s", agent.id
                )

        (
            messages,
            retrieval_log,
            retrieved_memories,
            graph_nodes_info,
            rag_docs_info,
        ) = build_agent_llm_messages(
            agent=agent,
            conversation=conversation,
            latest_user_text=message_text,
            latest_user_message=user_message,
        )

        mcp_tool_defs, mcp_tool_lookup = self._build_mcp_tools(agent)
        delegate_tool_defs, delegate_lookup = self._build_delegate_tools(agent)
        tool_definitions = mcp_tool_defs + delegate_tool_defs

        messages_payload = list(messages)

        require_tools = bool(tool_definitions) and agent.tool_mode == Agent.ToolMode.REQUIRED
        if require_tools:
            reminder = (
                "У тебя есть инструменты и/или агенты-делегаты. "
                "Ты ОБЯЗАН вызвать как минимум один инструмент перед тем, как дать итоговый ответ."
            )
            if messages_payload and messages_payload[0].get("role") == "system":
                messages_payload[0]["content"] = messages_payload[0]["content"] + "\n\n" + reminder
            else:
                messages_payload.insert(0, {"role": "system", "content": reminder})
        elif agent.tool_mode == Agent.ToolMode.REQUIRED and not tool_definitions:
            logger.warning(
                "Agent %s is in REQUIRED tool_mode but has no available tools/delegates",
                agent.id,
            )

        use_responses_api_model = requires_responses_api(model_name)

        try:
            if use_responses_api_model:
                reply_text, tool_traces = self._chat_with_responses(
                    client=client,
                    base_messages=messages_payload,
                    agent=agent,
                    model_name=model_name,
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens,
                    tool_definitions=tool_definitions,
                    mcp_tool_lookup=mcp_tool_lookup,
                    delegate_lookup=delegate_lookup,
                    project=project,
                    conversation=conversation,
                    require_tools=require_tools,
                )
            else:
                reply_text, tool_traces = self._chat_with_tools(
                    client=client,
                    base_messages=messages_payload,
                    agent=agent,
                    model_name=model_name,
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens,
                    tool_definitions=tool_definitions,
                    mcp_tool_lookup=mcp_tool_lookup,
                    delegate_lookup=delegate_lookup,
                    project=project,
                    conversation=conversation,
                )
        except NotFoundError as exc:
            logger.warning(
                "OpenAI model_not_found in assistant chat for agent %s: model_name=%s resolved=%s error=%s",
                agent.id,
                agent.model_name,
                agent.resolved_model_name,
                exc,
            )
            reply_text = (
                f"Похоже, выбранная модель ('{agent.resolved_model_name}') больше не доступна в OpenAI. "
                "Зайдите в настройки агента и выберите другую модель из списка."
            )
            tool_traces = []
        except AuthenticationError as exc:
            logger.error(
                "OpenAI authentication error in assistant chat for agent %s: %s",
                agent.id,
                exc,
            )
            reply_text = "Проблема с ключом OpenAI. Обратитесь к администратору или проверьте настройки."
            tool_traces = []
        except OpenAIError as exc:
            logger.error(
                "OpenAI error in assistant chat for agent %s: %s",
                agent.id,
                exc,
            )
            reply_text = "Сейчас не удалось связаться с LLM. Попробуйте ещё раз позже."
            tool_traces = []

        return self._finalize_response(
            conversation=conversation,
            agent=agent,
            user_message=user_message,
            reply_text=reply_text,
            tool_traces=tool_traces,
            retrieval_log=retrieval_log,
            retrieved_memories=retrieved_memories,
            graph_nodes_info=graph_nodes_info,
            rag_docs_info=rag_docs_info,
            client=client,
            model_name=model_name,
            request=request,
        )


class AssistantConversationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id: int):
        project = generics.get_object_or_404(Project, pk=project_id)
        agent_id = request.query_params.get("agent_id")
        if not agent_id:
            return Response(
                {"detail": "Параметр agent_id обязателен."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            agent_id_int = int(agent_id)
        except ValueError as exc:  # noqa: PERF203
            raise ValidationError({"agent_id": "Должен быть целым числом."}) from exc

        agent = (
            Agent.objects.available(project=project)
            .filter(pk=agent_id_int, is_active=True)
            .first()
        )
        if not agent:
            return Response(
                {"detail": "Агент не найден или недоступен для проекта."},
                status=status.HTTP_404_NOT_FOUND,
            )

        conversation = (
            Conversation.objects.filter(project=project, agent=agent, is_archived=False)
            .prefetch_related("messages")
            .order_by("-updated_at")
            .first()
        )
        if conversation is None:
            conversation = Conversation.objects.create(
                project=project,
                agent=agent,
                title=f"{agent.name} · {project.name}",
            )

        serializer = ConversationDetailSerializer(conversation, context={"request": request})
        return Response(serializer.data)

    def post(self, request, project_id: int):
        project = generics.get_object_or_404(Project, pk=project_id)
        agent_id = request.data.get("agent_id")
        if agent_id is None:
            return Response(
                {"detail": "Параметр agent_id обязателен."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            agent_id_int = int(agent_id)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValidationError({"agent_id": "Должен быть целым числом."}) from exc

        agent = (
            Agent.objects.available(project=project)
            .filter(pk=agent_id_int, is_active=True)
            .first()
        )
        if not agent:
            return Response(
                {"detail": "Агент не найден или недоступен для проекта."},
                status=status.HTTP_404_NOT_FOUND,
            )

        Conversation.objects.filter(
            project=project,
            agent=agent,
            is_archived=False,
        ).update(is_archived=True)

        conversation = Conversation.objects.create(
            project=project,
            agent=agent,
            title=f"{agent.name} · {project.name}",
        )
        serializer = ConversationDetailSerializer(conversation, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class MCPServerListCreateView(generics.ListCreateAPIView):
    queryset = MCPServer.objects.all()
    serializer_class = MCPServerSerializer


class MCPServerDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = MCPServer.objects.all()
    serializer_class = MCPServerSerializer


class MCPServerToolsListView(generics.ListAPIView):
    serializer_class = MCPToolSerializer

    def get_queryset(self):
        return MCPTool.objects.filter(server_id=self.kwargs["server_id"])


class MCPServerSyncToolsView(APIView):
    def post(self, request, server_id: int):
        server = generics.get_object_or_404(MCPServer, pk=server_id)

        stats = sync_tools_for_server(server)
        tools_count = MCPTool.objects.filter(server=server, is_active=True).count()
        return Response(
            {
                "server_id": server.id,
                "server_name": server.name,
                "tools_count": tools_count,
                "stats": stats,
            }
        )


class MCPToolInvokeView(APIView):
    def post(self, request, tool_id: int):
        tool = generics.get_object_or_404(MCPTool, pk=tool_id, is_active=True)
        server = tool.server

        arguments = request.data.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValidationError({"arguments": "Must be an object."})

        try:
            result = call_tool(server, tool, arguments)
            return Response(
                {
                    "tool": {
                        "id": tool.id,
                        "name": tool.name,
                        "server": {
                            "id": server.id,
                            "name": server.name,
                        },
                    },
                    "result": result,
                }
            )
        except MCPClientError as exc:
            return Response(
                {
                    "tool": {
                        "id": tool.id,
                        "name": tool.name,
                        "server": {
                            "id": server.id,
                            "name": server.name,
                        },
                    },
                    "error": {
                        "message": str(exc),
                        "code": getattr(exc, "code", None),
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )


class PipelineListCreateView(generics.ListCreateAPIView):
    serializer_class = PipelineSerializer

    def get_queryset(self):
        queryset = Pipeline.objects.all()
        project_id = self.request.query_params.get("project")
        if project_id is not None:
            queryset = queryset.filter(project_id=project_id)
        return queryset


class PipelineDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Pipeline.objects.all()
    serializer_class = PipelineSerializer


class PipelineRunView(APIView):
    def post(self, request, pipeline_id: int):
        pipeline = generics.get_object_or_404(Pipeline, pk=pipeline_id, is_active=True)

        input_payload = request.data.get("input") or {}
        if not isinstance(input_payload, dict):
            raise ValidationError({"input": "Must be an object."})

        created_by_agent_id = request.data.get("created_by_agent_id")
        created_by_agent = None
        if created_by_agent_id is not None:
            try:
                created_by_agent = Agent.objects.get(id=created_by_agent_id)
            except Agent.DoesNotExist:
                raise ValidationError({"created_by_agent_id": "Agent not found."})

        task = Task.objects.create(
            pipeline=pipeline,
            created_by_agent=created_by_agent,
            input_payload=input_payload,
        )

        from .tasks import run_pipeline_task

        run_pipeline_task.delay(task.id)

        return Response(
            {
                "task_id": task.id,
                "status": task.status,
            },
            status=status.HTTP_201_CREATED,
        )


class TaskDetailView(generics.RetrieveAPIView):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer
