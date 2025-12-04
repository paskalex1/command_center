from unittest.mock import patch

from django.test import TestCase

from core.constants import KNOWLEDGE_EXTRACTOR_SLUG
from core.models import Agent, KnowledgeNode, Project
from core.tasks import run_knowledge_extractor_for_text


class KnowledgeExtractorTaskTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Docs", slug="docs")
        self.primary_agent = Agent.objects.create(
            name="Primary",
            project=self.project,
            is_primary=True,
        )
        self.extractor, _ = Agent.objects.get_or_create(
            slug=KNOWLEDGE_EXTRACTOR_SLUG,
            defaults={
                "name": "Knowledge Extractor",
                "project": None,
                "system_prompt": "",
            },
        )

    @patch("core.tasks.call_knowledge_extractor")
    def test_task_updates_specified_agent(self, mocked_call):
        mocked_call.return_value = {
            "nodes": [
                {"text": "Booking API", "type": "entity", "description": "Service"},
                {"text": "Owner Portal", "type": "component"},
            ],
            "edges": [
                {"source": "Booking API", "target": "Owner Portal", "relation": "serves"},
            ],
        }
        run_knowledge_extractor_for_text(
            project_id=self.project.id,
            text="Booking API serves Owner Portal",
            target_agent_id=self.primary_agent.id,
            source="test",
        )
        self.assertEqual(
            KnowledgeNode.objects.filter(agent=self.primary_agent).count(),
            2,
        )

    @patch("core.tasks.call_knowledge_extractor")
    def test_task_falls_back_to_project_agents(self, mocked_call):
        mocked_call.return_value = {"nodes": [{"text": "CRM", "type": "system"}], "edges": []}
        run_knowledge_extractor_for_text(
            project_id=self.project.id,
            text="CRM definitions",
            target_agent_id=None,
            source="test",
        )
        self.assertTrue(
            KnowledgeNode.objects.filter(agent=self.primary_agent, label="CRM").exists()
        )
