import hashlib
import logging
from datetime import timedelta
from typing import List, Optional

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.db.models import Count, F, Q
from django.utils import timezone
try:  # безопасный импорт для разных версий SDK
    from openai import AuthenticationError, NotFoundError, OpenAIError  # type: ignore
except Exception:  # pragma: no cover
    AuthenticationError = NotFoundError = OpenAIError = Exception
from pypdf import PdfReader

from core.constants import KNOWLEDGE_EXTRACTOR_SLUG, RAG_LIBRARIAN_SLUG
from core.embeddings import embed_text
from .graph_extractor import extract_graph_from_text
from .models import (
    Agent,
    AgentMemory,
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeEmbedding,
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeSource,
    MemoryEmbedding,
    MemoryEvent,
    Pipeline,
    PipelineStep,
    Project,
    Task,
)
from .services.knowledge_discovery import scan_project_docs
from .services.knowledge_ingest import index_source
from .services.graph_ingest import ingest_extracted_knowledge
from .services.knowledge_extraction import call_knowledge_extractor
from .mcp_client import call_tool

logger = logging.getLogger(__name__)

TARGET_CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

AGENT_MEMORY_RETENTION_LOW_DAYS = int(
    getattr(settings, "AGENT_MEMORY_RETENTION_LOW_DAYS", 60)
)
AGENT_MEMORY_RETENTION_NORMAL_DAYS = int(
    getattr(settings, "AGENT_MEMORY_RETENTION_NORMAL_DAYS", 180)
)
MAX_AGENT_MEMORY_PER_AGENT = int(
    getattr(settings, "MAX_AGENT_MEMORY_PER_AGENT", 2000)
)
MAX_GRAPH_NODES_PER_AGENT = int(
    getattr(settings, "MAX_GRAPH_NODES_PER_AGENT", 2000)
)
MAX_GRAPH_EDGES_PER_AGENT = int(
    getattr(settings, "MAX_GRAPH_EDGES_PER_AGENT", 5000)
)

IMPORTANCE_PRIORITY = {
    AgentMemory.IMPORTANCE_LOW: 0,
    AgentMemory.IMPORTANCE_NORMAL: 1,
    AgentMemory.IMPORTANCE_HIGH: 2,
}


def _normalize_memory_content(text: str) -> str:
    return " ".join((text or "").lower().split())


def _compute_content_hash(text: str) -> str:
    normalized = _normalize_memory_content(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _pick_max_importance(*values: Optional[str]) -> str:
    current = AgentMemory.IMPORTANCE_NORMAL
    for value in values:
        if value not in IMPORTANCE_PRIORITY:
            continue
        if IMPORTANCE_PRIORITY[value] > IMPORTANCE_PRIORITY[current]:
            current = value
    return current


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


@shared_task
def generate_memory_embedding_for_event(event_id: int) -> None:
    try:
        event = MemoryEvent.objects.get(id=event_id)
    except MemoryEvent.DoesNotExist:
        logger.warning("MemoryEvent %s does not exist for embeddings", event_id)
        return

    try:
        vector = embed_text(event.content)
        MemoryEmbedding.objects.update_or_create(
            event=event,
            defaults={
                "embedding": vector,
            },
        )
        importance_map = {
            1: AgentMemory.IMPORTANCE_LOW,
            2: AgentMemory.IMPORTANCE_NORMAL,
            3: AgentMemory.IMPORTANCE_HIGH,
        }
        new_importance = importance_map.get(event.importance, AgentMemory.IMPORTANCE_NORMAL)

        content_hash = _compute_content_hash(event.content)
        existing: Optional[AgentMemory] = None
        if content_hash:
            existing = (
                AgentMemory.objects.filter(agent=event.agent, content_hash=content_hash)
                .order_by("-updated_at")
                .first()
            )

        if existing is None and content_hash:
            normalized_current = _normalize_memory_content(event.content)
            for memory in AgentMemory.objects.filter(agent=event.agent, content_hash=""):
                if _normalize_memory_content(memory.content) == normalized_current:
                    existing = memory
                    break

        if existing:
            existing.content = event.content
            existing.embedding = vector
            if content_hash:
                existing.content_hash = content_hash
            existing.importance = _pick_max_importance(existing.importance, new_importance)
            existing.save(update_fields=["content", "embedding", "content_hash", "importance", "updated_at"])
        else:
            AgentMemory.objects.create(
                agent=event.agent,
                content=event.content,
                content_hash=content_hash,
                embedding=vector,
                importance=new_importance,
            )
    except Exception as exc:
        logger.exception("Failed to generate embedding for MemoryEvent %s", event_id)


@shared_task
def process_memory_event_graph(event_id: int) -> None:
    try:
        event = MemoryEvent.objects.select_related("agent").get(id=event_id)
    except MemoryEvent.DoesNotExist:
        logger.warning("MemoryEvent %s does not exist for graph processing", event_id)
        return

    if event.graph_processed:
        return

    stats = extract_graph_from_text(event.agent, event.content)
    event.graph_processed = True
    event.save(update_fields=["graph_processed", "updated_at"])
    logger.info(
        "Graph extracted for MemoryEvent %s: %s",
        event.id,
        stats,
    )


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

                vector = embed_text(chunk.text)
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
def ingest_knowledge_source(source_id: str) -> None:
    try:
        source = KnowledgeSource.objects.get(id=source_id)
    except KnowledgeSource.DoesNotExist:
        logger.warning("KnowledgeSource %s does not exist", source_id)
        return

    source.status = KnowledgeSource.STATUS_QUEUED
    source.last_error = ""
    source.save(update_fields=["status", "last_error", "updated_at"])

    try:
        index_source(source)
    except Exception as exc:
        logger.exception("Failed to ingest KnowledgeSource %s: %s", source_id, exc)


@shared_task
def ingest_project_docs(project_id: int) -> None:
    try:
        project = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        logger.warning("Project %s does not exist for RAG ingest", project_id)
        return

    sources = scan_project_docs(project)
    for source in sources:
        if source.status in (KnowledgeSource.STATUS_NEW, KnowledgeSource.STATUS_ERROR):
            ingest_knowledge_source.delay(str(source.id))
    project.rag_last_full_sync_at = timezone.now()
    project.save(update_fields=["rag_last_full_sync_at"])


@shared_task
def sync_all_projects_rag() -> dict:
    from django.conf import settings

    max_age_days = getattr(settings, "RAG_AUTO_SYNC_DAYS", 3)
    threshold = timezone.now() - timezone.timedelta(days=max_age_days)
    queued = 0
    for project in Project.objects.all():
        last_sync = project.rag_last_full_sync_at
        if last_sync is None or last_sync < threshold:
            ingest_project_docs.delay(project.id)
            queued += 1
    logger.info("sync_all_projects_rag queued %s projects", queued)
    return {"queued": queued}


def _resolve_graph_target_agents(project: Project, target_agent_id: int | None) -> List[Agent]:
    targets: List[Agent] = []
    if target_agent_id:
        agent = Agent.objects.filter(id=target_agent_id, is_active=True).first()
        if agent:
            targets.append(agent)
        return targets

    project_agents = Agent.objects.filter(project=project, is_active=True)
    primary = list(project_agents.filter(is_primary=True))
    if primary:
        targets.extend(primary)
    else:
        targets.extend(list(project_agents))

    rag_agent = Agent.objects.filter(slug=RAG_LIBRARIAN_SLUG, is_active=True).first()
    if rag_agent and rag_agent not in targets:
        targets.append(rag_agent)

    seen = set()
    unique_targets: List[Agent] = []
    for agent in targets:
        if agent.id in seen:
            continue
        seen.add(agent.id)
        unique_targets.append(agent)
    return unique_targets


@shared_task
def run_knowledge_extractor_for_text(
    project_id: int,
    text: str,
    target_agent_id: int | None = None,
    source: str | None = None,
) -> dict:
    text = (text or "").strip()
    if not text:
        return {"status": "skipped", "reason": "empty"}
    try:
        project = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        logger.warning("Knowledge extractor: project %s not found", project_id)
        return {"status": "skipped", "reason": "project_missing"}

    extractor_agent = Agent.objects.filter(
        slug=KNOWLEDGE_EXTRACTOR_SLUG,
        is_active=True,
    ).first()
    if extractor_agent is None:
        logger.warning("Knowledge extractor agent is missing")
        return {"status": "skipped", "reason": "extractor_missing"}

    targets = _resolve_graph_target_agents(project, target_agent_id)
    if not targets:
        logger.info("Knowledge extractor: no target agents for project %s", project.slug)
        return {"status": "skipped", "reason": "no_targets"}

    payload = call_knowledge_extractor(
        extractor_agent,
        project,
        text,
        source=source,
    )
    nodes = payload.get("nodes") or []
    edges = payload.get("edges") or []

    stats = {}
    for agent in targets:
        result = ingest_extracted_knowledge(agent, nodes, edges)
        stats[str(agent.id)] = result

    return {
        "status": "ok",
        "targets": len(targets),
        "nodes": len(nodes),
        "edges": len(edges),
        "stats": stats,
    }


@shared_task
def cleanup_agent_memory() -> None:
    logger.info("Started cleanup_agent_memory task")
    total_deleted = 0
    now = timezone.now()

    low_cutoff = now - timedelta(days=AGENT_MEMORY_RETENTION_LOW_DAYS)
    normal_cutoff = now - timedelta(days=AGENT_MEMORY_RETENTION_NORMAL_DAYS)

    low_deleted, _ = AgentMemory.objects.filter(
        importance=AgentMemory.IMPORTANCE_LOW,
        created_at__lt=low_cutoff,
    ).delete()
    total_deleted += low_deleted

    normal_deleted, _ = AgentMemory.objects.filter(
        importance=AgentMemory.IMPORTANCE_NORMAL,
        created_at__lt=normal_cutoff,
    ).delete()
    total_deleted += normal_deleted

    if MAX_AGENT_MEMORY_PER_AGENT > 0:
        agent_ids = (
            AgentMemory.objects.values_list("agent_id", flat=True).distinct()
        )
        for agent_id in agent_ids:
            current_count = AgentMemory.objects.filter(agent_id=agent_id).count()
            excess = current_count - MAX_AGENT_MEMORY_PER_AGENT
            if excess <= 0:
                continue

            ids_to_delete: List[str] = []

            if excess > 0:
                low_ids = list(
                    AgentMemory.objects.filter(
                        agent_id=agent_id,
                        importance=AgentMemory.IMPORTANCE_LOW,
                    )
                    .order_by("created_at")
                    .values_list("id", flat=True)
                )
                take_low = min(excess, len(low_ids))
                ids_to_delete.extend(low_ids[:take_low])
                excess -= take_low

            if excess > 0:
                normal_ids = list(
                    AgentMemory.objects.filter(
                        agent_id=agent_id,
                        importance=AgentMemory.IMPORTANCE_NORMAL,
                    )
                    .order_by("created_at")
                    .values_list("id", flat=True)
                )
                take_normal = min(excess, len(normal_ids))
                ids_to_delete.extend(normal_ids[:take_normal])
                excess -= take_normal

            if ids_to_delete:
                deleted_count, _ = AgentMemory.objects.filter(id__in=ids_to_delete).delete()
                total_deleted += deleted_count

    totals = {
        importance: AgentMemory.objects.filter(importance=importance).count()
        for importance, _label in AgentMemory.IMPORTANCE_CHOICES
    }
    total_count = sum(totals.values())

    logger.info(
        "cleanup_agent_memory completed: deleted=%s, remaining=%s (low=%s, normal=%s, high=%s)",
        total_deleted,
        total_count,
        totals.get(AgentMemory.IMPORTANCE_LOW, 0),
        totals.get(AgentMemory.IMPORTANCE_NORMAL, 0),
        totals.get(AgentMemory.IMPORTANCE_HIGH, 0),
    )


@shared_task
def cleanup_graph_memory() -> None:
    logger.info("Starting cleanup_graph_memory task")
    total_nodes_deleted = 0
    total_edges_deleted = 0

    agent_ids = (
        KnowledgeNode.objects.values_list("agent_id", flat=True).distinct()
    )

    for agent_id in agent_ids:
        nodes_deleted, edges_deleted = _cleanup_graph_for_agent(agent_id)
        total_nodes_deleted += nodes_deleted
        total_edges_deleted += edges_deleted

    remaining_nodes = KnowledgeNode.objects.count()
    remaining_edges = KnowledgeEdge.objects.count()
    logger.info(
        "cleanup_graph_memory finished: nodes_deleted=%s edges_deleted=%s remaining_nodes=%s remaining_edges=%s",
        total_nodes_deleted,
        total_edges_deleted,
        remaining_nodes,
        remaining_edges,
    )


def _cleanup_graph_for_agent(agent_id: int) -> tuple[int, int]:
    nodes_deleted = 0
    edges_deleted = 0

    node_qs = KnowledgeNode.objects.filter(agent_id=agent_id)
    total_nodes = node_qs.count()

    if total_nodes > MAX_GRAPH_NODES_PER_AGENT:
        to_delete = total_nodes - MAX_GRAPH_NODES_PER_AGENT
        candidate_qs = (
            node_qs.filter(is_pinned=False)
            .annotate(
                deg=Count("outgoing_edges", distinct=True) + Count("incoming_edges", distinct=True),
            )
            .order_by(
                "usage_count",
                "deg",
                "last_used_at",
                "created_at",
            )
        )
        candidate_ids = list(candidate_qs.values_list("id", flat=True))
        selected_ids = candidate_ids[:to_delete]

        if selected_ids:
            edge_delete = KnowledgeEdge.objects.filter(
                Q(source_id__in=selected_ids) | Q(target_id__in=selected_ids)
            )
            deleted_edges, _ = edge_delete.delete()
            edges_deleted += deleted_edges

            deleted_nodes, _ = KnowledgeNode.objects.filter(id__in=selected_ids).delete()
            nodes_deleted += deleted_nodes

    nodes_deleted += _cleanup_zero_degree_nodes(agent_id)
    edges_deleted += _cleanup_edges_for_agent(agent_id)
    return nodes_deleted, edges_deleted


def _cleanup_zero_degree_nodes(agent_id: int) -> int:
    zero_deg_ids = list(
        KnowledgeNode.objects.filter(agent_id=agent_id, is_pinned=False)
        .annotate(
            deg=Count("outgoing_edges", distinct=True) + Count("incoming_edges", distinct=True),
        )
        .filter(deg=0, usage_count=0)
        .values_list("id", flat=True)
    )
    if not zero_deg_ids:
        return 0
    deleted_edges, _ = KnowledgeEdge.objects.filter(
        Q(source_id__in=zero_deg_ids) | Q(target_id__in=zero_deg_ids)
    ).delete()
    deleted_nodes, _ = KnowledgeNode.objects.filter(id__in=zero_deg_ids).delete()
    logger.info(
        "cleanup_graph_memory removed zero-degree nodes: nodes=%s edges=%s agent=%s",
        deleted_nodes,
        deleted_edges,
        agent_id,
    )
    return deleted_nodes


def _cleanup_edges_for_agent(agent_id: int) -> int:
    edge_qs = KnowledgeEdge.objects.filter(agent_id=agent_id)
    total_edges = edge_qs.count()

    if total_edges <= MAX_GRAPH_EDGES_PER_AGENT:
        return 0

    excess = total_edges - MAX_GRAPH_EDGES_PER_AGENT
    old_ids = list(
        edge_qs.order_by("created_at").values_list("id", flat=True)[:excess]
    )
    if not old_ids:
        return 0
    deleted, _ = KnowledgeEdge.objects.filter(id__in=old_ids).delete()
    return deleted


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

                request_kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": agent.temperature,
                    "timeout": 30,
                }
                if isinstance(agent.max_tokens, int) and agent.max_tokens > 0:
                    request_kwargs["max_tokens"] = agent.max_tokens

                try:
                    completion = client.chat.completions.create(**request_kwargs)
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
