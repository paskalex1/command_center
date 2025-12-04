from django.test import TestCase

from core.models import Agent, KnowledgeEdge, KnowledgeNode, Project
from core.services.graph_ingest import ingest_extracted_knowledge


class GraphIngestTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Docs", slug="docs")
        self.agent = Agent.objects.create(name="Primary", project=self.project, slug="primary")

    def test_creates_and_updates_nodes(self):
        nodes = [
            {"text": "Booking API", "type": "entity", "description": "HTTP service"},
            {"text": "Owner Portal", "type": "component", "description": "UI"},
        ]
        ingest_extracted_knowledge(self.agent, nodes, [])
        self.assertEqual(KnowledgeNode.objects.filter(agent=self.agent).count(), 2)

        # second run increments usage_count without duplicating
        ingest_extracted_knowledge(
            self.agent,
            [{"text": "Booking API", "type": "entity", "description": "Main API"}],
            [],
        )
        node = KnowledgeNode.objects.get(agent=self.agent, label="Booking API")
        self.assertGreaterEqual(node.usage_count, 2)

    def test_creates_edges_without_duplicates(self):
        nodes = [
            {"text": "CRM", "type": "system"},
            {"text": "Postgres", "type": "database"},
        ]
        edges = [
            {"source": "CRM", "target": "Postgres", "relation": "depends_on"},
            {"source": "CRM", "target": "Postgres", "relation": "depends_on"},
        ]
        ingest_extracted_knowledge(self.agent, nodes, edges)
        self.assertEqual(KnowledgeEdge.objects.filter(agent=self.agent).count(), 1)
