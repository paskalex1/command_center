import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from command_center import llm_registry
from core.models import Agent, Project
from command_center.services.llm_models import sync_models_registry
from django.conf import settings


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
