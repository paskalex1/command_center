import json
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings

from core.models import Agent, Conversation, MCPServer, MCPTool, Project
from core.services.web_knowledge_ingest import persist_web_knowledge_documents
from core.views import AssistantChatView


class WebKnowledgeIngestServiceTests(TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.override = override_settings(PROJECT_DOCS_ROOT=self.tempdir)
        self.override.enable()

    def tearDown(self):
        self.override.disable()
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def test_persist_web_documents_creates_files(self):
        project = Project.objects.create(name="Demo Project", slug="demo-project")
        payload = {
            "query": "django",
            "documents": [
                {
                    "title": "Django Tutorial",
                    "source_url": "https://docs.djangoproject.com/en/stable/",
                    "markdown": "## Intro\n\nDjango is a high-level Python Web framework.",
                    "plain_text": "Django is a high-level Python Web framework.",
                    "hash": "abc123",
                    "language": "en",
                    "tags": ["django", "docs"],
                    "metadata": {"search_snippet": "Django official docs"},
                }
            ],
        }

        saved = persist_web_knowledge_documents(project, payload)

        self.assertEqual(len(saved), 1)
        relative_path = saved[0]["relative_path"]
        self.assertTrue(relative_path.startswith("web_knowledge/"))
        absolute_path = project.docs_path / relative_path
        self.assertTrue(absolute_path.exists())
        content = absolute_path.read_text(encoding="utf-8")
        self.assertIn("Django Tutorial", content)
        self.assertIn("https://docs.djangoproject.com", content)

    def test_persist_web_documents_ignores_empty_payload(self):
        project = Project.objects.create(name="Demo Project", slug="demo-project")
        saved = persist_web_knowledge_documents(project, {"documents": []})
        self.assertEqual(saved, [])


class AssistantChatWebKnowledgeTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Test Project", slug="test-project")
        self.agent = Agent.objects.create(
            name="Main Agent",
            system_prompt="Prompt",
            project=self.project,
        )
        self.conversation = Conversation.objects.create(
            project=self.project,
            agent=self.agent,
            title="Test conversation",
        )
        self.server = MCPServer.objects.get(name="Web Knowledge MCP")
        self.tool = MCPTool.objects.create(
            server=self.server,
            name="search_and_collect_knowledge",
            description="",
            input_schema={},
        )
        self.view = AssistantChatView()

    @patch("core.views.ingest_project_docs")
    @patch("core.views.persist_web_knowledge_documents")
    @patch("core.views.call_tool")
    def test_handle_tool_call_stores_results_and_triggers_ingest(
        self,
        mock_call_tool: Mock,
        mock_persist: Mock,
        mock_ingest: Mock,
    ):
        mock_call_tool.return_value = {
            "query": "django",
            "documents": [
                {
                    "plain_text": "Example",
                    "source_url": "https://example.com",
                    "hash": "abc",
                }
            ],
        }
        mock_persist.return_value = [
            {"relative_path": "web_knowledge/example.md", "source_url": "https://example.com"}
        ]

        function_name = f"mcp_tool_{self.tool.id}"
        tool_call = SimpleNamespace(
            function=SimpleNamespace(name=function_name, arguments=json.dumps({})),
        )
        result = self.view._handle_tool_call(
            tool_call,
            {function_name: (self.server, self.tool)},
            {},
            self.project,
            self.conversation,
            client=None,
        )

        mock_call_tool.assert_called_once()
        self.assertIn("project_slug", mock_call_tool.call_args[0][2])
        self.assertEqual(mock_call_tool.call_args[0][2]["project_slug"], self.project.slug)
        mock_persist.assert_called_once_with(
            project=self.project,
            payload=mock_call_tool.return_value,
            query=None,
        )
        mock_ingest.delay.assert_called_once_with(self.project.id)
        self.assertEqual(result["result"]["stored_documents"], mock_persist.return_value)
