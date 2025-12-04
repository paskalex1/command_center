from unittest import TestCase
from unittest.mock import patch, MagicMock

from django.urls import reverse
from rest_framework.test import APIClient

from core.constants import RAG_LIBRARIAN_SLUG
from core.models import Agent, Project
from core.services import rag_agent


class RagAgentServiceTests(TestCase):
    def setUp(self):
        self.project_slug = "docs-project"

    def _mock_response(self, json_data, status=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = json_data
        mock_resp.text = str(json_data)
        return mock_resp

    @patch("core.services.rag_agent.requests.request")
    def test_get_sources_returns_results(self, mocked_request):
        mocked_request.return_value = self._mock_response(
            {"count": 1, "results": [{"relative_path": "README.md"}], "project_meta": {"rag_error_count": 0}}
        )
        data = rag_agent.get_sources(self.project_slug)
        self.assertEqual(data["count"], 1)
        self.assertIn("project_meta", data)
        mocked_request.assert_called_once()

    @patch("core.services.rag_agent.requests.request")
    def test_reindex_all_returns_message(self, mocked_request):
        mocked_request.return_value = self._mock_response({"status": "queued", "task_id": "abc"})
        message = rag_agent.reindex_all(self.project_slug)
        self.assertIn("task_id", message)

    @patch("core.services.rag_agent.requests.request")
    def test_summarize_sources_formats_text(self, mocked_request):
        mocked_request.return_value = self._mock_response(
            {
                "count": 2,
                "project_meta": {"rag_error_count": 1, "rag_last_full_sync_at": "2025-11-29T10:00:00Z"},
                "results": [
                    {"relative_path": "README.md", "status": "processed"},
                    {"relative_path": "guide.txt", "status": "new"},
                ],
            }
        )
        text = rag_agent.summarize_sources(self.project_slug)
        self.assertIn("Документация", text)
        self.assertIn("README.md", text)
        self.assertIn("Ошибок", text)


class AssistantChatRagTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        slug = f"docs-{self._testMethodName.lower()}"
        self.project = Project.objects.create(name="Docs", slug=slug)
        self.agent, _ = Agent.objects.get_or_create(
            slug=RAG_LIBRARIAN_SLUG,
            defaults={
                "name": "RAG Librarian",
                "system_prompt": "",
                "project": None,
            },
        )

    @patch("core.views.rag_agent.summarize_sources", return_value="Статус: все хорошо")
    def test_rag_agent_answers_via_chat(self, mocked_summary):
        payload = {
            "message": "Покажи статус RAG",
            "agent_id": self.agent.id,
        }
        url = f"/api/projects/{self.project.id}/assistant/chat/"
        resp = self.client.post(url, payload, format="json")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assistant_messages = [m for m in body["messages"] if m["sender"] == "assistant"]
        self.assertTrue(assistant_messages)
        self.assertIn("Статус", assistant_messages[-1]["content"])
        mocked_summary.assert_called_once_with(self.project.slug)

    @patch("core.views.rag_agent.get_error_sources", return_value=[{"relative_path": "bad.md", "error_message": "boom"}])
    def test_rag_agent_errors_command(self, mocked_errors):
        payload = {
            "message": "Покажи ошибки RAG",
            "agent_id": self.agent.id,
        }
        url = f"/api/projects/{self.project.id}/assistant/chat/"
        resp = self.client.post(url, payload, format="json")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assistant_messages = [m for m in body["messages"] if m["sender"] == "assistant"]
        self.assertTrue(assistant_messages)
        self.assertIn("Ошибки документации", assistant_messages[-1]["content"])
        mocked_errors.assert_called_once_with(self.project.slug)

    @patch("core.views.rag_agent.get_changelog", return_value=[{"source_path": "README.md", "change_type": "modified", "semantic_summary": "Добавлено"}])
    def test_rag_agent_changes_command(self, mocked_changes):
        payload = {
            "message": "Покажи изменения в документации",
            "agent_id": self.agent.id,
        }
        url = f"/api/projects/{self.project.id}/assistant/chat/"
        resp = self.client.post(url, payload, format="json")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        assistant_messages = [m for m in body["messages"] if m["sender"] == "assistant"]
        self.assertTrue(assistant_messages)
        self.assertIn("Последние изменения", assistant_messages[-1]["content"])
        mocked_changes.assert_called_once_with(self.project.slug)
