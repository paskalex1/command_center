from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import Project
from core.tasks import sync_all_projects_rag


class RAGSyncTaskTests(TestCase):
    def setUp(self):
        now = timezone.now()
        self.old_project = Project.objects.create(
            name="Old",
            slug="old",
            rag_last_full_sync_at=now - timedelta(days=10),
        )
        self.recent_project = Project.objects.create(
            name="Recent",
            slug="recent",
            rag_last_full_sync_at=now - timedelta(days=1),
        )

    @override_settings(RAG_AUTO_SYNC_DAYS=3)
    @patch("core.tasks.ingest_project_docs.delay")
    def test_sync_only_stale_projects(self, mocked_delay):
        result = sync_all_projects_rag()
        mocked_delay.assert_called_once_with(self.old_project.id)
        self.assertEqual(result["queued"], 1)
