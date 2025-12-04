from unittest.mock import patch

from django.test import TestCase

from core.models import Agent, Project
from core.services.knowledge_extraction import (
    maybe_schedule_knowledge_extractor_for_agent,
)


class OrchestratorGraphHookTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name="Docs", slug="docs")
        self.agent = Agent.objects.create(name="Primary", project=self.project)

    @patch("core.services.knowledge_extraction.schedule_knowledge_extractor")
    def test_knowledge_rich_text_triggers_scheduler(self, mocked_schedule):
        text = "Сущность Property: содержит поля address, rooms, owner_id.\n" * 10
        maybe_schedule_knowledge_extractor_for_agent(
            project=self.project,
            agent=self.agent,
            text=text,
            source="test",
        )
        mocked_schedule.assert_called_once()

    @patch("core.services.knowledge_extraction.schedule_knowledge_extractor")
    def test_short_text_does_not_trigger(self, mocked_schedule):
        maybe_schedule_knowledge_extractor_for_agent(
            project=self.project,
            agent=self.agent,
            text="ok",
            source="test",
        )
        mocked_schedule.assert_not_called()
