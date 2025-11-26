import mimetypes
import os

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
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from .mcp_client import MCPClientError, call_tool, sync_tools_for_server
from command_center.llm_registry import (
    get_chat_primary,
    get_chat_recommended,
    get_embedding_default,
    load_registry,
)
from command_center.tasks import sync_llm_registry_task
from .models import (
    Agent,
    AgentServerBinding,
    Conversation,
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeEmbedding,
    MCPServer,
    MCPTool,
    MemoryEmbedding,
    MemoryEvent,
    Message,
    Pipeline,
    Project,
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
from .tasks import (
    generate_memory_embedding_for_event,
    get_text_embedding,
    process_knowledge_document,
)

import logging


logger = logging.getLogger(__name__)


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

    agents = []
    initial_agent = None

    if current_project is not None:
        agents = list(
            Agent.objects.filter(
                Q(project=current_project) | Q(project__isnull=True),
                is_active=True,
            )
            .order_by("-is_primary", "name")
        )
        for agent in agents:
            if agent.is_primary:
                initial_agent = agent
                break
        if initial_agent is None and agents:
            initial_agent = agents[0]

    mcp_servers = MCPServer.objects.all().order_by("name")
    mcp_tools = MCPTool.objects.select_related("server").all().order_by(
        "server__name", "name"
    )

    pipelines = []
    memory_events = []

    if current_project is not None:
        pipelines = list(
            Pipeline.objects.select_related("owner_agent")
            .filter(project=current_project)
            .order_by("name")
        )
        memory_events = list(
            MemoryEvent.objects.filter(project=current_project)
            .order_by("-created_at")[:20]
        )

    # данные LLM реестра для UI
    llm_chat_primary = get_chat_primary()
    llm_embedding_default = get_embedding_default()

    context = {
        "projects": projects,
        "current_project": current_project,
        "agents": agents,
        "initial_agent": initial_agent,
        "mcp_servers": mcp_servers,
        "mcp_tools": mcp_tools,
        "pipelines": pipelines,
        "memory_events": memory_events,
        "llm_chat_primary": llm_chat_primary,
        "llm_embedding_default": llm_embedding_default,
    }
    return render(request, "core/dashboard.html", context)


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

        query_embedding = get_text_embedding(query)

        embeddings_qs = KnowledgeEmbedding.objects.filter(
            chunk__project_id=project_id,
            chunk__document__status=KnowledgeDocument.STATUS_READY,
        ).annotate(
            distance=CosineDistance("embedding", query_embedding)
        ).order_by("distance")[:top_k]

        results = []
        for embedding in embeddings_qs:
            chunk = embedding.chunk
            document = chunk.document
            knowledge_base = document.knowledge_base

            results.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "knowledge_base_id": knowledge_base.id,
                    "project_id": knowledge_base.project_id,
                    "text": chunk.text,
                    "score": float(getattr(embedding, "distance", 0.0)),
                    "meta": chunk.meta or {},
                }
            )

        return Response(results)


class AgentListView(generics.ListCreateAPIView):
    serializer_class = AgentSerializer

    def get_queryset(self):
        queryset = Agent.objects.all()
        project_id = self.request.query_params.get("project")
        if project_id is not None:
            queryset = queryset.filter(project_id=project_id)
        return queryset


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

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                timeout=30,
            )
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
            .filter(agent=agent)
            .all()
        )

        for binding in bindings:
            server = binding.server
            tools_qs = MCPTool.objects.filter(server=server, is_active=True)
            allowed = binding.allowed_tools or []
            if allowed:
                tools_qs = tools_qs.filter(name__in=allowed)

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


class ConversationDetailView(generics.RetrieveAPIView):
    queryset = Conversation.objects.all()
    serializer_class = ConversationDetailSerializer


class AssistantChatView(APIView):
    def post(self, request, project_id: int):
        project = generics.get_object_or_404(Project, pk=project_id)

        message_text = request.data.get("message")
        if not message_text or not isinstance(message_text, str):
            raise ValidationError({"message": "This field is required and must be a string."})

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
                source_message=user_message,
                type=MemoryEvent.TYPE_FACT,
                content=message_text,
                importance=0.8,
            )
            generate_memory_embedding_for_event.delay(memory_event.id)

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

        try:
            query_embedding = get_text_embedding(message_text)
        except Exception as exc:
            query_embedding = None

        memory_facts = []
        if query_embedding is not None:
            memory_qs = MemoryEmbedding.objects.filter(
                event__agent=agent,
                event__project=project,
            ).annotate(
                distance=CosineDistance("embedding", query_embedding)
            ).order_by("distance")[:5]

            for mem in memory_qs:
                memory_facts.append(mem.event.content)

        base_system_prompt = agent.system_prompt or ""
        memory_section = ""
        if memory_facts:
            bullet_lines = []
            for idx, text in enumerate(memory_facts, start=1):
                bullet_lines.append(f"{idx}) {text}")
            memory_text = "\n".join(bullet_lines)
            memory_section = (
                "Вот важные факты о пользователе и проекте, которые ты знаешь из памяти "
                "(они могут быть полезны при ответе):\n"
                f"{memory_text}\n\n"
                "Если факты не относятся к вопросу, не выдумывай и не навязывай их, "
                "но учитывай, если они действительно релевантны."
            )

        if base_system_prompt and memory_section:
            system_prompt = base_system_prompt + "\n\n" + memory_section
        elif memory_section:
            system_prompt = memory_section
        else:
            system_prompt = base_system_prompt

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message_text})

        model_name = agent.resolved_model_name

        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                timeout=30,
            )
            reply_text = completion.choices[0].message.content or ""
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
        except AuthenticationError as exc:
            logger.error(
                "OpenAI authentication error in assistant chat for agent %s: %s",
                agent.id,
                exc,
            )
            reply_text = "Проблема с ключом OpenAI. Обратитесь к администратору или проверьте настройки."
        except OpenAIError as exc:
            logger.error(
                "OpenAI error in assistant chat for agent %s: %s",
                agent.id,
                exc,
            )
            reply_text = "Сейчас не удалось связаться с LLM. Попробуйте ещё раз позже."

        assistant_message = Message.objects.create(
            conversation=conversation,
            sender=Message.ROLE_ASSISTANT,
            content=reply_text,
            meta={},
        )

        messages_data = MessageSerializer(
            [user_message, assistant_message],
            many=True,
        ).data

        return Response(
            {
                "conversation_id": conversation.id,
                "agent_id": agent.id,
                "messages": messages_data,
            }
        )


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

        try:
            tools_count = sync_tools_for_server(server)
        except MCPClientError as exc:
            return Response(
                {
                    "server_id": server.id,
                    "server_name": server.name,
                    "error": {
                        "message": str(exc),
                        "code": getattr(exc, "code", None),
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(
            {
                "server_id": server.id,
                "server_name": server.name,
                "tools_count": tools_count,
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
