import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.utils import timezone
from rest_framework.test import APIClient

from command_center import llm_registry
from command_center.services.llm_models import sync_models_registry
from core.graph_extractor import extract_graph_from_text
from core.models import (
    Agent,
    Conversation,
    KnowledgeEdge,
    KnowledgeEmbedding,
    KnowledgeNode,
    KnowledgeSource,
    Message,
    Project,
)
from core.services.graph_memory import build_graph_memory_block
from core.services.knowledge_discovery import scan_project_docs
from core.services.knowledge_ingest import index_source
from core.tasks import cleanup_graph_memory
from django.conf import settings

FAKE_EMBEDDING = [0.01] * 1536


class LLMRegistryTests(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.registry_path = Path(self.tmpdir.name) / "models_registry.json"

        data = {
            "chat": {
                "primary": "gpt-5.1",
                "recommended": ["gpt-5.1", "gpt-4.1"],
                "models": ["gpt-5.1", "gpt-4.1"],
            },
            "embedding": {
                "default": "text-embedding-3-small",
                "recommended": ["text-embedding-3-small"],
                "models": ["text-embedding-3-small"],
            },
            "lightweight": ["gpt-4.1-mini"],
            "realtime": ["gpt-4o-realtime-preview"],
            "search": ["gpt-4o-search-preview"],
            "deprecated": ["gpt-3.5-turbo"],
        }
        self.registry_path.write_text(json.dumps(data), encoding="utf-8")

        # reset cache
        llm_registry._REGISTRY_CACHE = None
        llm_registry._REGISTRY_MTIME = None

        self._path_patcher = patch(
            "command_center.llm_registry._registry_path", return_value=self.registry_path
        )
        self._path_patcher.start()
        self.addCleanup(self._path_patcher.stop)

    def test_load_registry_caches_by_mtime(self):
        first = llm_registry.load_registry()
        second = llm_registry.load_registry()
        self.assertIs(first, second)

        # emulate file update
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        data["chat"]["primary"] = "gpt-NEW"
        self.registry_path.write_text(json.dumps(data), encoding="utf-8")

        llm_registry._REGISTRY_CACHE = None
        third = llm_registry.load_registry()
        self.assertIsNot(first, third)
        self.assertEqual(third["chat"]["primary"], "gpt-NEW")

    def test_get_chat_primary_returns_value_from_registry(self):
        self.assertEqual(llm_registry.get_chat_primary(), "gpt-5.1")

    def test_get_embedding_default_returns_value_from_registry(self):
        self.assertEqual(llm_registry.get_embedding_default(), "text-embedding-3-small")

    def test_get_chat_recommended_list(self):
        rec = llm_registry.get_chat_recommended()
        self.assertIn("gpt-5.1", rec)
        self.assertIn("gpt-4.1", rec)

    def test_get_all_models_by_type_returns_models_list(self):
        chat_models = llm_registry.get_all_models_by_type("chat")
        self.assertEqual(
            {m["name"] for m in chat_models},
            {"gpt-5.1", "gpt-4.1"},
        )

    def test_get_all_known_model_names_contains_all_models(self):
        names = llm_registry.get_all_known_model_names()
        for expected in [
            "gpt-5.1",
            "gpt-4.1",
            "text-embedding-3-small",
            "gpt-4.1-mini",
            "gpt-4o-realtime-preview",
            "gpt-4o-search-preview",
            "gpt-3.5-turbo",
        ]:
            self.assertIn(expected, names)


class AgentModelTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Test", description="")

    @patch("core.models.get_chat_primary", return_value="gpt-PRIMARY")
    def test_resolved_model_name_uses_primary_when_empty(self, _patched_primary):
        agent = Agent.objects.create(project=None, name="A", system_prompt="")
        self.assertEqual(agent.model_name, "")
        self.assertEqual(agent.resolved_model_name, "gpt-PRIMARY")

    def test_resolved_model_name_uses_custom_when_set(self):
        agent = Agent.objects.create(
            project=None,
            name="A",
            system_prompt="",
            model_name="gpt-4.1-mini",
        )
        self.assertEqual(agent.resolved_model_name, "gpt-4.1-mini")

    @patch("core.models.get_all_known_model_names", return_value=["known-model"])
    def test_agent_save_logs_warning_for_unknown_model(self, _patched_known):
        with self.assertLogs("core.models", level="WARNING") as cm:
            Agent.objects.create(
                project=None,
                name="A",
                system_prompt="",
                model_name="unknown-model",
            )
        self.assertTrue(
            any("unknown-model" in message for message in cm.output),
            msg="Expected warning about unknown model",
        )

    @patch("core.models.is_model_deprecated", return_value=True)
    def test_agent_save_logs_warning_for_deprecated_model(self, _patched_depr):
        with self.assertLogs("core.models", level="WARNING") as cm:
            Agent.objects.create(
                project=None,
                name="A",
                system_prompt="",
                model_name="gpt-3.5-turbo",
            )
        self.assertTrue(
            any("deprecated LLM model" in message for message in cm.output),
            msg="Expected warning about deprecated model",
        )


class AssistantConversationViewTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User = get_user_model()
        self.user = User.objects.create_superuser(
            username="conv-admin",
            email="conv@example.com",
            password="password",
        )
        self.client.force_authenticate(self.user)

        self.project = Project.objects.create(name="Conversation Project", description="")
        self.agent = Agent.objects.create(
            project=self.project,
            name="Helper",
            system_prompt="be helpful",
        )
        self.url = f"/api/projects/{self.project.id}/assistant/conversation/"

    def test_get_creates_conversation_if_missing(self):
        response = self.client.get(f"{self.url}?agent_id={self.agent.id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["agent"], self.agent.id)
        self.assertEqual(data["project"], self.project.id)
        self.assertEqual(data["messages"], [])
        self.assertTrue(
            Conversation.objects.filter(
                project=self.project,
                agent=self.agent,
                is_archived=False,
            ).exists()
        )

    def test_post_resets_conversation(self):
        old_conversation = Conversation.objects.create(
            project=self.project,
            agent=self.agent,
            title="Old",
        )
        Message.objects.create(
            conversation=old_conversation,
            sender=Message.ROLE_USER,
            content="hi",
        )

        response = self.client.post(
            self.url,
            {"agent_id": self.agent.id},
            format="json",
        )
        self.assertEqual(response.status_code, 201)
        data = response.json()

        old_conversation.refresh_from_db()
        self.assertTrue(old_conversation.is_archived)
        self.assertNotEqual(data["id"], old_conversation.id)
        self.assertEqual(data["messages"], [])


class AssistantChatResponsesTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User = get_user_model()
        self.user = User.objects.create_superuser(
            username="responses-admin",
            email="responses@example.com",
            password="password",
        )
        self.client.force_authenticate(self.user)
        os.environ["OPENAI_API_KEY"] = "test-key"

        self.project = Project.objects.create(name="Responses Project", description="")
        self.agent = Agent.objects.create(
            project=self.project,
            name="Coder",
            system_prompt="",
            model_name="gpt-5.1-codex",
        )

    @patch("core.views.requires_responses_api", return_value=True)
    @patch("core.views.extract_memory_events")
    @patch("core.views.OpenAI")
    def test_chat_falls_back_to_responses_api(self, mock_openai, mock_extract_memory, _mock_requires):
        mock_client = MagicMock()
        mock_response = SimpleNamespace(output_text="Привет из Responses API", output=[])
        mock_client.responses.create.return_value = mock_response
        mock_openai.return_value = mock_client
        mock_extract_memory.return_value = None

        url = f"/api/projects/{self.project.id}/assistant/chat/"
        response = self.client.post(
            url,
            {"agent_id": self.agent.id, "message": "Привет"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["messages"][-1]["content"], "Привет из Responses API")
        mock_client.responses.create.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

    @patch("core.views.requires_responses_api", return_value=True)
    @patch("core.views.extract_memory_events")
    @patch("core.views.AssistantChatView._handle_tool_call", return_value={"result": "ok"})
    @patch("core.views.OpenAI")
    def test_responses_api_handles_tool_calls(
        self,
        mock_openai,
        mock_handle_tool,
        mock_extract_memory,
        _mock_requires,
    ):
        delegate = Agent.objects.create(
            project=self.project,
            name="Delegate",
            system_prompt="",
            model_name="gpt-5.1",
        )
        self.agent.delegates.add(delegate)

        first_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name=f"delegate_to_agent_{delegate.id}",
                    arguments='{"task": "ping"}',
                )
            ],
            conversation=SimpleNamespace(id="conv-1"),
            output_text="",
        )
        second_response = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="Готово")],
                )
            ],
            conversation=SimpleNamespace(id="conv-1"),
            output_text="Готово",
        )
        mock_client = MagicMock()
        mock_client.responses.create.side_effect = [first_response, second_response]
        mock_openai.return_value = mock_client
        mock_extract_memory.return_value = None

        url = f"/api/projects/{self.project.id}/assistant/chat/"
        response = self.client.post(
            url,
            {"agent_id": self.agent.id, "message": "Запусти делегата"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_client.responses.create.call_count, 2)
        mock_handle_tool.assert_called_once()
        data = response.json()
        self.assertEqual(data["messages"][-1]["content"], "Готово")


class APIRegistryAndAgentTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        User = get_user_model()
        self.user = User.objects.create_superuser(
            username="admin",
            email="admin@example.com",
            password="password",
        )
        self.client.force_authenticate(self.user)

        self.project = Project.objects.create(name="P1", description="")
        self.agent = Agent.objects.create(
            project=self.project,
            name="Agent1",
            system_prompt="",
        )

    def test_models_registry_endpoint_structure(self):
        response = self.client.get("/api/models/registry/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for key in [
            "chat",
            "embedding",
            "lightweight",
            "realtime",
            "search",
            "deprecated",
            "chat_primary",
            "embedding_default",
            "chat_recommended",
        ]:
            self.assertIn(key, data)

    @patch("core.models.get_chat_primary", return_value="gpt-PRIMARY")
    def test_agent_patch_model_name_empty_uses_default(self, _patched_primary):
        url = f"/api/agents/{self.agent.id}/"
        response = self.client.patch(
            url,
            data={"model_name": ""},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_name"], "")
        self.assertEqual(body["resolved_model_name"], "gpt-PRIMARY")

    def test_agent_patch_model_name_custom(self):
        url = f"/api/agents/{self.agent.id}/"
        response = self.client.patch(
            url,
            data={"model_name": "gpt-4.1-mini"},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_name"], "gpt-4.1-mini")
        self.assertEqual(body["resolved_model_name"], "gpt-4.1-mini")


class SyncLLMRegistryTests(TestCase):
    def test_celery_beat_schedule_configured(self):
        schedule = getattr(settings, "CELERY_BEAT_SCHEDULE", {})
        self.assertIn("sync-llm-registry-monthly", schedule)
        entry = schedule["sync-llm-registry-monthly"]
        self.assertEqual(
            entry.get("task"), "command_center.tasks.sync_llm_registry_task"
        )

    def test_sync_models_registry_writes_file(self):
        # Мы не хотим реально ходить в OpenAI здесь, поэтому замокаем клиент.
        with patch("command_center.services.llm_models.OpenAI") as mocked_client:
            instance = mocked_client.return_value
            instance.models.list.return_value = type(
                "Obj",
                (),
                {
                    "data": [
                        type("M", (), {"id": "gpt-5.1"})(),
                        type("M", (), {"id": "text-embedding-3-small"})(),
                    ]
                },
            )()

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                registry = sync_models_registry()

        self.assertIn("chat", registry)
        self.assertIn("embedding", registry)
        path = (
            Path(settings.BASE_DIR)
            / "command_center"
            / "config"
            / "models_registry.json"
        )
        self.assertTrue(path.exists())

    def test_models_registry_sync_endpoint_requires_admin(self):
        client = APIClient()
        response = client.post("/api/models/registry/sync/")
        self.assertIn(response.status_code, (401, 403))

    def test_models_registry_sync_endpoint_as_admin(self):
        User = get_user_model()
        user = User.objects.create_superuser(
            username="admin2",
            email="admin2@example.com",
            password="password",
        )
        client = APIClient()
        client.force_authenticate(user)

        with patch("command_center.tasks.sync_llm_registry_task.delay") as mocked_delay:
            response = client.post("/api/models/registry/sync/")

        self.assertEqual(response.status_code, 200)
        mocked_delay.assert_called_once()


class GraphExtractorTests(TestCase):
    def setUp(self):
        self.agent = Agent.objects.create(
            name=f"GraphExtractorAgent-{self._testMethodName}",
            model_name="gpt-4o-mini",
        )

    @patch("core.graph_extractor.embed_text", return_value=FAKE_EMBEDDING)
    @patch("core.graph_extractor._get_openai_client")
    def test_extract_graph_creates_nodes_and_edges(self, mock_get_client, _mock_embed):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = """{
          "nodes": [
            {"label": "Command Center", "type": "service", "description": "Django-приложение."},
            {"label": "filesystem MCP", "type": "mcp_server", "description": "MCP-сервер."}
          ],
          "edges": [
            {
              "source_label": "Command Center",
              "target_label": "filesystem MCP",
              "relation": "uses",
              "description": "Command Center использует filesystem MCP."
            }
          ]
        }"""
        mock_choice = MagicMock(message=mock_message)
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        stats = extract_graph_from_text(self.agent, "Command Center использует filesystem MCP")

        self.assertEqual(stats["created_nodes"], 2)
        self.assertEqual(stats["created_edges"], 1)
        self.assertEqual(KnowledgeNode.objects.filter(agent=self.agent).count(), 2)
        self.assertEqual(KnowledgeEdge.objects.filter(agent=self.agent).count(), 1)

    @patch("core.graph_extractor.embed_text", return_value=FAKE_EMBEDDING)
    @patch("core.graph_extractor._get_openai_client")
    def test_extract_graph_idempotent(self, mock_get_client, _mock_embed):
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_message = MagicMock()
        mock_message.content = """{
          "nodes": [
            {"label": "Command Center", "type": "service", "description": "Django-приложение."}
          ],
          "edges": []
        }"""
        mock_choice = MagicMock(message=mock_message)
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        extract_graph_from_text(self.agent, "Описание Command Center")
        extract_graph_from_text(self.agent, "Описание Command Center")

        self.assertEqual(KnowledgeNode.objects.filter(agent=self.agent).count(), 1)
        self.assertEqual(KnowledgeEdge.objects.filter(agent=self.agent).count(), 0)


class GraphMemoryRetrievalTests(TestCase):
    def setUp(self):
        self.agent = Agent.objects.create(
            name=f"GraphRetrievalAgent-{self._testMethodName}",
            model_name="gpt-4o-mini",
        )
        self.node_cc = KnowledgeNode.objects.create(
            agent=self.agent,
            label="Command Center",
            type="service",
            description="Django-приложение для управления AI-агентами.",
            embedding=FAKE_EMBEDDING,
        )
        self.node_fs = KnowledgeNode.objects.create(
            agent=self.agent,
            label="filesystem MCP",
            type="mcp_server",
            description="MCP-сервер для работы с файлами.",
            embedding=FAKE_EMBEDDING,
        )
        KnowledgeEdge.objects.create(
            agent=self.agent,
            source=self.node_cc,
            target=self.node_fs,
            relation="uses",
            description="Command Center использует filesystem MCP.",
        )

    @patch("core.services.graph_memory.embed_text", return_value=FAKE_EMBEDDING)
    def test_build_graph_memory_block_updates_usage(self, _mock_embed):
        block, used_nodes = build_graph_memory_block(
            self.agent,
            "Как Command Center использует filesystem MCP?",
            max_nodes=5,
        )

        self.assertIn("AGENT GRAPH MEMORY", block)
        self.assertIn("Command Center", block)
        self.assertIn("filesystem MCP", block)
        self.assertTrue(used_nodes)
        labels = {node["label"] for node in used_nodes}
        self.assertIn("Command Center", labels)

        self.node_cc.refresh_from_db()
        self.node_fs.refresh_from_db()
        self.assertGreaterEqual(self.node_cc.usage_count, 1)
        self.assertGreaterEqual(self.node_fs.usage_count, 1)
        self.assertIsNotNone(self.node_cc.last_used_at)
        self.assertIsNotNone(self.node_fs.last_used_at)

    @patch("core.services.graph_memory.embed_text", return_value=FAKE_EMBEDDING)
    def test_build_graph_memory_block_empty_without_nodes(self, _mock_embed):
        other_agent = Agent.objects.create(
            name=f"EmptyAgent-{self._testMethodName}",
            model_name="gpt-4o-mini",
        )
        block, used_nodes = build_graph_memory_block(other_agent, "любой вопрос")
        self.assertEqual(block, "")
        self.assertEqual(used_nodes, [])


class GraphCleanupTests(TestCase):
    def setUp(self):
        self.agent = Agent.objects.create(
            name=f"GraphCleanupAgent-{self._testMethodName}",
            model_name="gpt-4o-mini",
        )
        self.nodes_patch = patch("core.tasks.MAX_GRAPH_NODES_PER_AGENT", 2)
        self.edges_patch = patch("core.tasks.MAX_GRAPH_EDGES_PER_AGENT", 2)
        self.nodes_patch.start()
        self.edges_patch.start()
        self.addCleanup(self.nodes_patch.stop)
        self.addCleanup(self.edges_patch.stop)

    def _create_node(self, **kwargs):
        defaults = {
            "agent": self.agent,
            "label": kwargs.get("label") or f"Node-{KnowledgeNode.objects.count()}",
            "type": kwargs.get("type", "temp"),
            "description": kwargs.get("description", "Temporary node"),
            "embedding": kwargs.get("embedding", FAKE_EMBEDDING),
            "usage_count": kwargs.get("usage_count", 0),
            "is_pinned": kwargs.get("is_pinned", False),
        }
        defaults.update(kwargs)
        return KnowledgeNode.objects.create(**defaults)

    def test_pinned_nodes_not_deleted(self):
        pinned = self._create_node(label="PinnedCore", is_pinned=True)
        self._create_node(label="Other1")
        self._create_node(label="Other2")

        cleanup_graph_memory()

        self.assertTrue(KnowledgeNode.objects.filter(id=pinned.id).exists())
        self.assertLessEqual(KnowledgeNode.objects.filter(agent=self.agent).count(), 2)

    def test_cleanup_prefers_isolated_unused_nodes(self):
        isolated = self._create_node(label="Isolated", usage_count=0)
        hub = self._create_node(label="Hub", usage_count=5)
        other = self._create_node(label="Other", usage_count=2)
        KnowledgeEdge.objects.create(
            agent=self.agent,
            source=hub,
            target=other,
            relation="related_to",
            description="Hub связан с Other.",
        )

        cleanup_graph_memory()

        self.assertFalse(KnowledgeNode.objects.filter(id=isolated.id).exists())
        self.assertTrue(KnowledgeNode.objects.filter(id=hub.id).exists())
        self.assertTrue(KnowledgeNode.objects.filter(id=other.id).exists())

    def test_edges_deleted_with_removed_nodes(self):
        removable = self._create_node(label="Removable", usage_count=0)
        survivor = self._create_node(label="Survivor", usage_count=5)
        connecting = self._create_node(label="Connecting", usage_count=4)
        removable_edge = KnowledgeEdge.objects.create(
            agent=self.agent,
            source=removable,
            target=survivor,
            relation="rel",
            description="Удаляемая связь",
        )
        surviving_edge = KnowledgeEdge.objects.create(
            agent=self.agent,
            source=survivor,
            target=connecting,
            relation="rel",
            description="Должна остаться",
        )

        cleanup_graph_memory()

        self.assertFalse(KnowledgeEdge.objects.filter(id=removable_edge.id).exists())
        self.assertTrue(KnowledgeEdge.objects.filter(id=surviving_edge.id).exists())


class RAGLibrarianTests(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.override = override_settings(PROJECT_DOCS_ROOT=self.tmpdir.name)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.project = Project.objects.create(name="Docs project", slug="docs-project")
        self.docs_path = Path(self.project.docs_path)
        self.docs_path.mkdir(parents=True, exist_ok=True)

    def _write_doc(self, relative_path: str, content: str):
        target = self.docs_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def test_scan_project_docs_creates_sources_and_detects_changes(self):
        self._write_doc("README.md", "hello world")
        self._write_doc("nested/info.txt", "initial")

        sources = scan_project_docs(self.project)
        self.assertEqual(len(sources), 2)
        paths = {s.path for s in sources}
        self.assertIn("README.md", paths)
        self.assertIn("nested/info.txt", paths)

        source = KnowledgeSource.objects.get(project=self.project, path="README.md")
        source.status = KnowledgeSource.STATUS_PROCESSED
        source.save(update_fields=["status"])

        self._write_doc("README.md", "changed content")
        updated_sources = scan_project_docs(self.project)
        readme_entry = next(s for s in updated_sources if s.path == "README.md")
        self.assertEqual(readme_entry.status, KnowledgeSource.STATUS_NEW)

    @patch("core.services.knowledge_ingest.embed_text", return_value=[0.1] * 1536)
    def test_index_source_creates_embeddings(self, _patched_embed):
        self._write_doc("guide.md", "Небольшой текст для индексации.")
        source = scan_project_docs(self.project)[0]

        created = index_source(source)
        self.assertGreaterEqual(created, 1)

        source.refresh_from_db()
        self.assertEqual(source.status, KnowledgeSource.STATUS_PROCESSED)
        self.assertEqual(source.last_error, "")

        embeddings = KnowledgeEmbedding.objects.filter(source=source)
        self.assertEqual(embeddings.count(), created)
        self.assertTrue(all(len(entry.embedding) == 1536 for entry in embeddings))
