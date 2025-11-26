import logging
import os
from typing import List

from celery import shared_task
from django.db import transaction
from django.utils import timezone
from openai import OpenAI

try:  # безопасный импорт для разных версий SDK
    from openai import AuthenticationError, NotFoundError, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    AuthenticationError = NotFoundError = OpenAIError = Exception
from pypdf import PdfReader

from command_center.llm_registry import get_embedding_default
from .models import (
    Agent,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEmbedding,
    MemoryEmbedding,
    MemoryEvent,
    Pipeline,
    PipelineStep,
    Project,
    Task,
)
from .mcp_client import call_tool

logger = logging.getLogger(__name__)

TARGET_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _split_text_into_chunks(text: str) -> List[str]:
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) <= 1:
        return _split_text_by_chars(text)

    chunks: List[str] = []
    current_paragraphs: List[str] = []

    for paragraph in paragraphs:
        if not current_paragraphs:
            current_paragraphs.append(paragraph)
            continue

        tentative = "\n\n".join(current_paragraphs + [paragraph])

        if len(tentative) <= TARGET_CHUNK_SIZE:
            current_paragraphs.append(paragraph)
        else:
            chunk_text = "\n\n".join(current_paragraphs).strip()
            if chunk_text:
                chunks.append(chunk_text)

            overlap_paragraph = current_paragraphs[-1]
            current_paragraphs = [overlap_paragraph, paragraph]

    if current_paragraphs:
        final_chunk = "\n\n".join(current_paragraphs).strip()
        if final_chunk:
            chunks.append(final_chunk)

    return chunks


def _split_text_by_chars(text: str) -> List[str]:
    chunks: List[str] = []

    if not text:
        return chunks

    step = max(TARGET_CHUNK_SIZE - CHUNK_OVERLAP, 1)
    start = 0
    length = len(text)

    while start < length:
        end = start + TARGET_CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step

    return chunks


def _extract_text_from_document(document: KnowledgeDocument) -> str:
    file_path = document.file.path
    mime_type = document.mime_type or ""

    if mime_type.startswith("application/pdf") or file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                pages_text.append(page_text)
        return "\n\n".join(pages_text)

    if mime_type.startswith("text/markdown") or file_path.lower().endswith(".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if mime_type.startswith("text/") or file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    with open(file_path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


def get_text_embedding(text: str) -> List[float]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set, cannot create embeddings")
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    model_name = get_embedding_default() or "text-embedding-3-small"
    response = client.embeddings.create(
        model=model_name,
        input=text,
    )
    return response.data[0].embedding


@shared_task
def generate_memory_embedding_for_event(event_id: int) -> None:
    try:
        event = MemoryEvent.objects.get(id=event_id)
    except MemoryEvent.DoesNotExist:
        logger.warning("MemoryEvent %s does not exist for embeddings", event_id)
        return

    try:
        vector = get_text_embedding(event.content)
        MemoryEmbedding.objects.update_or_create(
            event=event,
            defaults={
                "embedding": vector,
            },
        )
    except Exception as exc:
        logger.exception("Failed to generate embedding for MemoryEvent %s", event_id)


@shared_task
def generate_embeddings_for_document(document_id: int) -> None:
    try:
        document = KnowledgeDocument.objects.select_related(
            "knowledge_base", "knowledge_base__project"
        ).get(id=document_id)
    except KnowledgeDocument.DoesNotExist:
        logger.warning("KnowledgeDocument %s does not exist for embeddings", document_id)
        return

    project = document.knowledge_base.project

    try:
        with transaction.atomic():
            KnowledgeEmbedding.objects.filter(chunk__document=document).delete()

            chunk_qs = KnowledgeChunk.objects.filter(
                document=document, project=project
            ).order_by("chunk_index")

            for chunk in chunk_qs:
                if not chunk.text:
                    continue

                vector = get_text_embedding(chunk.text)
                KnowledgeEmbedding.objects.create(chunk=chunk, embedding=vector)

        meta = document.meta or {}
        meta["embeddings_generated_at"] = timezone.now().isoformat()
        document.meta = meta
        document.save(update_fields=["meta"])

    except Exception as exc:
        logger.exception(
            "Failed to generate embeddings for document %s", document_id
        )
        meta = document.meta or {}
        meta["embeddings_error"] = str(exc)
        document.meta = meta
        document.save(update_fields=["meta"])


@shared_task
def process_knowledge_document(document_id: int) -> None:
    try:
        document = KnowledgeDocument.objects.select_related(
            "knowledge_base", "knowledge_base__project"
        ).get(id=document_id)
    except KnowledgeDocument.DoesNotExist:
        logger.warning("KnowledgeDocument %s does not exist", document_id)
        return

    document.status = KnowledgeDocument.STATUS_PROCESSING
    document.meta = {
        **(document.meta or {}),
        "started_at": timezone.now().isoformat(),
    }
    document.save(update_fields=["status", "meta"])

    try:
        text = _extract_text_from_document(document)
        chunks = _split_text_into_chunks(text)

        project: Project = document.knowledge_base.project

        with transaction.atomic():
            KnowledgeChunk.objects.filter(document=document).delete()

            chunk_objects = []
            for index, chunk_text in enumerate(chunks):
                chunk_objects.append(
                    KnowledgeChunk(
                        document=document,
                        project=project,
                        text=chunk_text,
                        chunk_index=index,
                        meta={},
                    )
                )

            if chunk_objects:
                KnowledgeChunk.objects.bulk_create(chunk_objects)

            document.status = KnowledgeDocument.STATUS_READY
            document.meta = {
                **(document.meta or {}),
                "chunk_count": len(chunks),
                "completed_at": timezone.now().isoformat(),
            }
            document.save(update_fields=["status", "meta"])

        generate_embeddings_for_document.delay(document.id)

    except Exception as exc:
        logger.exception("Failed to process KnowledgeDocument %s", document_id)
        document.status = KnowledgeDocument.STATUS_ERROR
        document.meta = {
            **(document.meta or {}),
            "error": str(exc),
            "failed_at": timezone.now().isoformat(),
        }
        document.save(update_fields=["status", "meta"])


@shared_task
def run_pipeline_task(task_id: int) -> None:
    try:
        task = Task.objects.select_related("pipeline").get(id=task_id)
    except Task.DoesNotExist:
        logger.warning("Task %s does not exist", task_id)
        return

    task.status = Task.STATUS_RUNNING
    task.current_step_index = 0
    task.error_message = ""
    task.save(update_fields=["status", "current_step_index", "error_message", "updated_at"])

    steps = list(task.pipeline.steps.all().order_by("order"))

    # Copy input payload to avoid mutating original dict
    context = dict(task.input_payload or {})
    if "steps" not in context:
        context["steps"] = []

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None

    for index, step in enumerate(steps):
        task.current_step_index = index
        task.save(update_fields=["current_step_index", "updated_at"])

        try:
            if step.type == PipelineStep.TYPE_AGENT:
                agent: Agent | None = step.agent
                if agent is None:
                    raise RuntimeError("Pipeline step configured as 'agent' but agent is not set")

                message = step.config.get("prompt") if step.config else None
                if not message:
                    message = context.get("message", "")

                if not api_key or client is None:
                    raise RuntimeError("OPENAI_API_KEY is not set for agent step")

                system_prompt = agent.system_prompt or ""
                model_name = agent.resolved_model_name

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": message})

                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=agent.temperature,
                        max_tokens=agent.max_tokens,
                        timeout=30,
                    )
                    reply = completion.choices[0].message.content or ""
                except NotFoundError as exc:
                    logger.warning(
                        "OpenAI model_not_found in pipeline task %s step %s: agent=%s model_name=%s resolved=%s error=%s",
                        task.id,
                        step.id,
                        agent.id,
                        agent.model_name,
                        agent.resolved_model_name,
                        exc,
                    )
                    raise RuntimeError(
                        f"Модель '{agent.resolved_model_name}' больше не доступна в OpenAI. "
                        "Обновите модель агента в настройках."
                    ) from exc
                except AuthenticationError as exc:
                    logger.error(
                        "OpenAI authentication error in pipeline task %s step %s: %s",
                        task.id,
                        step.id,
                        exc,
                    )
                    raise RuntimeError(
                        "Проблема с ключом OpenAI при выполнении пайплайна."
                    ) from exc
                except OpenAIError as exc:
                    logger.error(
                        "OpenAI error in pipeline task %s step %s: %s",
                        task.id,
                        step.id,
                        exc,
                    )
                    raise RuntimeError(
                        "Не удалось связаться с LLM при выполнении пайплайна."
                    ) from exc

                context_key = f"agent_step_{step.order}_reply"
                context[context_key] = reply
                context["last_reply"] = reply

                context["steps"].append(
                    {
                        "step_id": step.id,
                        "order": step.order,
                        "type": step.type,
                        "status": "done",
                        "output": {
                            "reply": reply,
                        },
                    }
                )

            elif step.type == PipelineStep.TYPE_TOOL:
                tool = step.tool
                if tool is None:
                    raise RuntimeError("Pipeline step configured as 'tool' but tool is not set")

                arguments = {}
                if step.config:
                    arguments = step.config.get("arguments") or {}

                result = call_tool(tool.server, tool, arguments)

                context_key = f"tool_step_{step.order}_result"
                context[context_key] = result
                context["last_tool_result"] = result

                context["steps"].append(
                    {
                        "step_id": step.id,
                        "order": step.order,
                        "type": step.type,
                        "status": "done",
                        "output": {
                            "result": result,
                        },
                    }
                )
            else:
                raise RuntimeError(f"Unknown pipeline step type: {step.type}")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error while executing step %s of task %s", step.id, task.id)
            task.status = Task.STATUS_ERROR
            task.error_message = str(exc)
            task.result_payload = {
                "input": task.input_payload,
                "steps": context.get("steps", []),
                "context": context,
                "error": str(exc),
            }
            task.save(
                update_fields=[
                    "status",
                    "error_message",
                    "result_payload",
                    "updated_at",
                ]
            )
            return

    task.status = Task.STATUS_DONE
    task.result_payload = {
        "input": task.input_payload,
        "steps": context.get("steps", []),
        "context": context,
    }
    task.save(update_fields=["status", "result_payload", "updated_at"])
