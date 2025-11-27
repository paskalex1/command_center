import json
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


class ConversationDetailView(generics.RetrieveAPIView):
    queryset = Conversation.objects.all()
    serializer_class = ConversationDetailSerializer


class AssistantChatView(APIView):
    MAX_TOOL_ITERATIONS = 5

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

    def _run_delegate_agent_once(self, delegate: Agent, project: Project, task: str, client: OpenAI) -> tuple[str, list[dict]]:
        messages = []
        if delegate.system_prompt:
            messages.append({"role": "system", "content": delegate.system_prompt})
        messages.append({"role": "user", "content": task})

        mcp_tool_defs, mcp_tool_lookup = self._build_mcp_tools(delegate)
        delegate_tool_defs, delegate_lookup = self._build_delegate_tools(delegate)
        tool_definitions = mcp_tool_defs + delegate_tool_defs

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
        )
        return reply, traces

    def _handle_tool_call(
        self,
        tool_call,
        mcp_tool_lookup: dict[str, tuple[MCPServer, MCPTool]],
        delegate_lookup: dict[str, Agent],
        project: Project,
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
            try:
                result = call_tool(server, tool, arguments or {})
                logger.info(
                    "Tool call succeeded: server=%s tool=%s function=%s args=%s",
                    server.name,
                    tool.name,
                    name,
                    arguments,
                )
                return {
                    "status": "ok",
                    "type": "mcp",
                    "server": server.name,
                    "tool": tool.name,
                    "result": result,
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
                response, delegate_traces = self._run_delegate_agent_once(delegate, project, task, client)
                logger.info(
                    "Delegate tool call succeeded: delegate=%s function=%s task=%s",
                    delegate.name,
                    name,
                    task,
                )
                return {
                    "status": "ok",
                    "type": "delegate",
                    "delegate": delegate.name,
                    "response": response,
                    "tool_traces": delegate_traces,
                }
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
                tool_payload = self._handle_tool_call(tc, mcp_tool_lookup, delegate_lookup, project, client)
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
            if iterations >= self.MAX_TOOL_ITERATIONS:
                return reply_text + "\n\n[system] Достигнут лимит вызовов инструментов.", tool_traces

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

        mcp_tool_defs, mcp_tool_lookup = self._build_mcp_tools(agent)
        delegate_tool_defs, delegate_lookup = self._build_delegate_tools(agent)
        tool_definitions = mcp_tool_defs + delegate_tool_defs

        if tool_definitions and agent.tool_mode == Agent.ToolMode.REQUIRED:
            reminder = (
                "У тебя есть инструменты и/или агенты-делегаты. "
                "Ты ОБЯЗАН вызвать как минимум один инструмент перед тем, как дать итоговый ответ."
            )
            if system_prompt:
                messages[0]["content"] = messages[0]["content"] + "\n\n" + reminder
            else:
                messages.insert(0, {"role": "system", "content": reminder})
        elif agent.tool_mode == Agent.ToolMode.REQUIRED and not tool_definitions:
            logger.warning(
                "Agent %s is in REQUIRED tool_mode but has no available tools/delegates",
                agent.id,
            )

        try:
            reply_text, tool_traces = self._chat_with_tools(
                client=client,
                base_messages=messages,
                agent=agent,
                model_name=model_name,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                tool_definitions=tool_definitions,
                mcp_tool_lookup=mcp_tool_lookup,
                delegate_lookup=delegate_lookup,
                project=project,
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

        assistant_meta = {}
        if tool_traces:
            assistant_meta["tool_traces"] = tool_traces

        assistant_message = Message.objects.create(
            conversation=conversation,
            sender=Message.ROLE_ASSISTANT,
            content=reply_text,
            meta=assistant_meta,
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
