from unittest.mock import patch

from django.test import TestCase

from core.models import KnowledgeChangeLog, KnowledgeSource, Project
from core.services.rag_versions import (
    create_version_snapshot,
    record_changelog_entry,
)


class RAGVersioningServiceTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            name="Docs",
            slug="docs",
        )
        self.source = KnowledgeSource.objects.create(
            project=self.project,
            path="README.md",
            filename="README.md",
            content_hash="hash-initial",
            mime_type="text/markdown",
            status=KnowledgeSource.STATUS_NEW,
        )

    def test_snapshot_keeps_full_text_for_text_files(self):
        text = "Hello RAG"
        version = create_version_snapshot(self.source, text)
        self.assertEqual(version.project, self.project)
        self.assertEqual(version.full_text, text)
        self.assertEqual(version.text_head, text)
        self.assertEqual(version.size_bytes, len(text.encode("utf-8")))

    def test_snapshot_skips_full_text_for_binary_files(self):
        self.source.mime_type = "application/pdf"
        self.source.content_hash = "hash-pdf"
        version = create_version_snapshot(self.source, "PDF BODY")
        self.assertEqual(version.full_text, "")
        self.assertTrue(version.text_head)

    @patch("core.services.rag_versions.update_graph_with_facts")
    def test_record_changelog_entry_detects_changes(self, mocked_graph):
        self.source.content_hash = "hash-old"
        previous = create_version_snapshot(self.source, "Old fact")
        self.source.content_hash = "hash-new"
        new_version = create_version_snapshot(self.source, "Old fact\nNew fact line")

        entry = record_changelog_entry(
            source=self.source,
            previous_version=previous,
            new_version=new_version,
            current_text="Old fact\nNew fact line",
        )

        self.assertEqual(entry.change_type, KnowledgeChangeLog.TYPE_MODIFIED)
        self.assertIn("New fact line", entry.new_facts)
        mocked_graph.assert_called_once_with(self.project.slug, entry.new_facts)
        self.assertTrue(entry.diff_text)
