import json
import logging
from typing import Union

from openai import OpenAI, OpenAIError

from .models import Agent, Conversation, MemoryEvent, Message, Project
from .tasks import generate_memory_embedding_for_event, process_memory_event_graph

EXTRACTOR_PROMPT = (
    "Ты — Memory Extractor. Твоя задача — анализировать последние сообщения агента "
    "и вытаскивать важные факты, планы, решения, задачи, схемы. "
    "Пиши кратко и по делу. Если нечего запоминать — возвращай {\"events\": []}.\n"
    "Формат ответа (JSON):\n"
    '{"events": [{"type": "fact|task|plan|schema|decision", "importance": 1-3, "content": "..."}, ...]}\n'
    "Важность: 1 — низкая, 2 — нормальная, 3 — высокая."
)

logger = logging.getLogger(__name__)


MessageLike = Union[Message, str]


def extract_memory_events(
    client: OpenAI,
    model_name: str,
    agent: Agent,
    project: Project,
    conversation: Conversation,
    user_message: MessageLike,
    assistant_message: MessageLike,
):
    user_text = user_message.content if isinstance(user_message, Message) else str(user_message)
    assistant_text = (
        assistant_message.content if isinstance(assistant_message, Message) else str(assistant_message)
    )

    payload = {
        "conversation_id": conversation.id,
        "user_message": user_text,
        "assistant_message": assistant_text,
    }
    events = []
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": EXTRACTOR_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0,
            timeout=15,
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        events = data.get("events", []) or []
    except (OpenAIError, json.JSONDecodeError) as exc:
        logger.warning("Memory extractor failed: %s", exc)
        return

    assistant_message_obj = assistant_message if isinstance(assistant_message, Message) else None

    for event_data in events:
        content = (event_data or {}).get("content", "").strip()
        if not content:
            continue
        event_type = event_data.get("type", MemoryEvent.TYPE_FACT)
        importance = int(event_data.get("importance", 1))
        importance = max(1, min(3, importance))
        memory_event = MemoryEvent.objects.create(
            agent=agent,
            project=project,
            conversation=conversation,
            message=assistant_message_obj,
            type=event_type,
            content=content,
            importance=importance,
        )
        generate_memory_embedding_for_event.delay(memory_event.id)
        process_memory_event_graph.delay(memory_event.id)
