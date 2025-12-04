from unittest.mock import patch, MagicMock

from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import KnowledgeSource, KnowledgeSourceVersion, KnowledgeChangeLog, Project


class RAGApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = get_user_model().objects.create_user(
            username="tester",
            email="tester@example.com",
            password="pass1234",
        )
        self.project = Project.objects.create(
            name="Docs", slug="docs-project", description=""
        )
        self.source = KnowledgeSource.objects.create(
            project=self.project,
            path="README.md",
            filename="README.md",
            content_hash="hash1",
            mime_type="text/markdown",
        )

    def authenticate(self):
        self.client.force_authenticate(self.user)

    def test_rag_sources_list_success(self):
        self.authenticate()
        other = KnowledgeSource.objects.create(
            project=self.project,
            path="guide/intro.txt",
            filename="intro.txt",
            content_hash="hash2",
            mime_type="text/plain",
            status=KnowledgeSource.STATUS_PROCESSED,
        )

        resp = self.client.get(f"/api/projects/{self.project.slug}/rag/sources/")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["count"], 2)
        self.assertIn("project_meta", payload)
        paths = {item["relative_path"] for item in payload["results"]}
        self.assertIn(self.source.path, paths)
        self.assertIn(other.path, paths)

    @patch("core.api.views.rag.ingest_project_docs.delay")
    def test_rag_ingest_triggers_celery_task(self, patched_delay):
        self.authenticate()
        patched_delay.return_value = MagicMock(id="task-123")

        resp = self.client.post(
            f"/api/projects/{self.project.slug}/rag/ingest/",
            {"mode": "all"},
            format="json",
        )
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(resp.json()["status"], "queued")
        patched_delay.assert_called_once_with(self.project.id)

    def test_rag_sources_requires_auth(self):
        resp = self.client.get(f"/api/projects/{self.project.slug}/rag/sources/")
        self.assertEqual(resp.status_code, 403)

    def test_rag_changelog_list(self):
        self.authenticate()
        version = KnowledgeSourceVersion.objects.create(
            source=self.source,
            project=self.project,
            content_hash="hash1",
            mime_type="text/markdown",
            size_bytes=12,
            text_head="Hello",
            full_text="Hello",
        )
        KnowledgeChangeLog.objects.create(
            project=self.project,
            source=self.source,
            previous_version=None,
            version=version,
            change_type=KnowledgeChangeLog.TYPE_ADDED,
        )

        resp = self.client.get(f"/api/projects/{self.project.slug}/rag/changelog/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["results"][0]["source_path"], self.source.path)
