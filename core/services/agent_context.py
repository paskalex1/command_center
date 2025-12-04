import logging
from typing import List, Optional, Tuple

from pgvector.django import CosineDistance

from ..embeddings import embed_text
from ..models import Agent, AgentMemory, Conversation, Message, RetrievalLog
from .graph_memory import build_graph_memory_block
from .rag_memory import build_rag_memory_block
from .rag_delegate import get_rag_delegate


logger = logging.getLogger(__name__)

ROLE_MAPPING = {
    Message.ROLE_USER: "user",
    Message.ROLE_ASSISTANT: "assistant",
    Message.ROLE_SYSTEM: "system",
}


def _human_file_size(num_bytes: int) -> str:
    if not num_bytes:
        return ""
    units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.1f} {units[idx]}"


def _with_attachments_note(message: Message, base_text: str) -> str:
    attachments_manager = getattr(message, "attachments", None)
    if attachments_manager is None or not hasattr(attachments_manager, "all"):
        attachments = []
    else:
        attachments = list(attachments_manager.all())
    if not attachments:
        return base_text
    lines = ["", "", "Прикреплённые файлы:"]
    for attachment in attachments:
        name = attachment.original_name or attachment.file.name
        size_label = _human_file_size(attachment.size)
        url = getattr(attachment.file, "url", "")
        parts = [f"- {name}"]
        if size_label:
            parts.append(f"({size_label})")
        if url:
            parts.append(f"URL: {url}")
        lines.append(" ".join(parts))
    return (base_text or "") + "\n".join(lines)


def build_agent_llm_messages(
    *,
    agent: Agent,
    conversation: Conversation,
    latest_user_text: str,
    latest_user_message: Optional[Message] = None,
    history_limit: int = 50,
) -> Tuple[List[dict], Optional[RetrievalLog], List[dict], List[dict], List[dict]]:
    """
    Build a full message list for invoking LLM on behalf of a specific agent.

    Returns a tuple:
      * messages — список сообщений для OpenAI;
      * retrieval_log — запись о выборке памяти или None;
      * retrieved_memories — метаданные использованных AgentMemory;
      * graph_nodes_info — узлы графа, попавшие в контекст;
      * rag_docs_info — документы RAG, добавленные в контекст.
    """

    messages: List[dict] = []

    query_embedding = None
    try:
        query_embedding = embed_text(latest_user_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to embed user text for agent %s conversation %s: %s",
            agent.id,
            conversation.id,
            exc,
        )

    retrieval_log: Optional[RetrievalLog] = None
    retrieved_memories: List[dict] = []
    memory_section = ""

    if query_embedding is not None:
        memory_qs = (
            AgentMemory.objects.filter(agent=agent)
            .annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")[:10]
        )

        memory_facts = []
        for mem in memory_qs:
            memory_facts.append(mem.content)
            retrieved_memories.append(
                {
                    "id": str(mem.id),
                    "content": mem.content,
                    "importance": mem.importance,
                    "distance": float(getattr(mem, "distance", 0.0)),
                    "created_at": mem.created_at.isoformat(),
                }
            )
        if memory_facts:
            bullet_lines = [f"{idx}) {text}" for idx, text in enumerate(memory_facts, start=1)]
            facts_text = "\n".join(bullet_lines)
            memory_section = (
                "Вот важные факты из памяти агента (используй их только если они релевантны запросу):\n"
                f"{facts_text}"
            )
            retrieval_log = RetrievalLog.objects.create(
                agent=agent,
                query=latest_user_text,
                used_memories=[mem.id for mem in memory_qs],
            )

    system_prompt = agent.system_prompt or ""
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    rag_delegate = get_rag_delegate(agent)
    if rag_delegate and rag_delegate.id != agent.id:
        delegate_tool_name = f"delegate_to_agent_{rag_delegate.id}"
        messages.append(
            {
                "role": "system",
                "content": (
                    "Если пользователь просит добавить или обновить документацию, проиндексировать файлы, "
                    "узнать статус RAG или выполнить действия с /docs, делегируй задачу агенту "
                    f"«{rag_delegate.name}» через инструмент `{delegate_tool_name}`. "
                    "Сформулируй задачу максимально конкретно и дождись результата перед ответом пользователю."
                ),
            }
        )
    if memory_section:
        messages.append({"role": "system", "content": memory_section})

    graph_block = ""
    graph_nodes_info: List[dict] = []
    try:
        graph_block, graph_nodes_info = build_graph_memory_block(agent, latest_user_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build graph memory block for agent %s: %s", agent.id, exc)
    if graph_block:
        messages.append({"role": "system", "content": graph_block})

    rag_block = ""
    rag_docs_info: List[dict] = []
    try:
        rag_block, rag_docs_info = build_rag_memory_block(agent, latest_user_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build RAG memory block for agent %s: %s", agent.id, exc)
    if rag_block:
        messages.append({"role": "system", "content": rag_block})

    history_qs = (
        Message.objects.filter(conversation=conversation)
        .prefetch_related("attachments")
        .order_by("-created_at")
    )
    if latest_user_message is not None:
        history_qs = history_qs.exclude(id=latest_user_message.id)

    if history_limit:
        history_buffer = list(history_qs[: history_limit])
        history_buffer.reverse()
    else:
        history_buffer = list(history_qs.order_by("created_at"))

    for history_message in history_buffer:
        content = history_message.content or ""
        content = _with_attachments_note(history_message, content)
        if not content.strip():
            continue
        role = ROLE_MAPPING.get(history_message.sender, "user")
        messages.append({"role": role, "content": content})

    latest_text_for_model = latest_user_text
    if latest_user_message is not None:
        latest_text_for_model = _with_attachments_note(latest_user_message, latest_user_text)

    messages.append({"role": "user", "content": latest_text_for_model})

    return messages, retrieval_log, retrieved_memories, graph_nodes_info, rag_docs_info
