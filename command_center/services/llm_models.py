import json
import os
from pathlib import Path
from typing import Dict, List, Set

from django.conf import settings
from openai import OpenAI

from command_center.llm_registry import load_registry


REGISTRY_PATH = Path(settings.BASE_DIR) / "command_center" / "config" / "models_registry.json"

RESPONSES_ONLY_KEYWORDS = ("-codex", "-code")


PREFERRED_CHAT_ORDER: List[str] = [
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-chat-latest",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
]

PREFERRED_EMBEDDING_ORDER: List[str] = [
    "text-embedding-3-small",
    "text-embedding-3-large",
]


def _classify_model(model_id: str) -> str:
    mid = model_id.lower()
    if "embedding" in mid:
        return "embedding"
    if "realtime" in mid or "audio" in mid:
        return "realtime"
    if "search" in mid:
        return "search"
    if "codex" in mid or "-code" in mid:
        return "code"
    if any(
        prefix in mid
        for prefix in ["gpt-3.5", "text-embedding-ada", "babbage", "davinci", "curie"]
    ):
        return "deprecated"
    return "chat"


def sync_models_registry() -> Dict[str, object]:
    """
    Получает доступные модели через OpenAI API,
    строит структуру реестра, сохраняет models_registry.json
    и возвращает итоговый dict.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не установлен")

    client = OpenAI(api_key=api_key)
    response = client.models.list()

    all_models: List[str] = [m.id for m in response.data]  # type: ignore[attr-defined]

    chat_models: Set[str] = set()
    lightweight_models: Set[str] = set()
    code_models: Set[str] = set()
    embedding_models: Set[str] = set()
    realtime_models: Set[str] = set()
    search_models: Set[str] = set()
    deprecated_models: Set[str] = set()

    for m_id in all_models:
        category = _classify_model(m_id)
        if category == "chat":
            chat_models.add(m_id)
            if any(x in m_id.lower() for x in ["mini", "nano", "small"]):
                lightweight_models.add(m_id)
        elif category == "embedding":
            embedding_models.add(m_id)
        elif category == "code":
            code_models.add(m_id)
            chat_models.add(m_id)
            if any(x in m_id.lower() for x in ["mini", "nano", "small"]):
                lightweight_models.add(m_id)
        elif category == "realtime":
            realtime_models.add(m_id)
        elif category == "search":
            search_models.add(m_id)
        elif category == "deprecated":
            deprecated_models.add(m_id)

    primary_chat = None
    for candidate in PREFERRED_CHAT_ORDER:
        if candidate in chat_models:
            primary_chat = candidate
            break
    if primary_chat is None and chat_models:
        primary_chat = sorted(chat_models)[0]

    recommended_chat = [m for m in PREFERRED_CHAT_ORDER if m in chat_models]

    default_embedding = None
    for candidate in PREFERRED_EMBEDDING_ORDER:
        if candidate in embedding_models:
            default_embedding = candidate
            break
    if default_embedding is None and embedding_models:
        default_embedding = sorted(embedding_models)[0]

    recommended_embedding = [
        m for m in PREFERRED_EMBEDDING_ORDER if m in embedding_models
    ]

    registry: Dict[str, object] = {
        "chat": {
            "primary": primary_chat,
            "recommended": recommended_chat,
            "models": sorted(chat_models),
        },
        "lightweight": sorted(lightweight_models),
        "code": sorted(code_models),
        "embedding": {
            "default": default_embedding,
            "recommended": recommended_embedding,
            "models": sorted(embedding_models),
        },
        "realtime": sorted(realtime_models),
        "search": sorted(search_models),
        "deprecated": sorted(deprecated_models),
    }

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2, sort_keys=True)

    return registry


def requires_responses_api(model_name: str) -> bool:
    """
    Возвращает True, если указанная модель доступна только через Responses API.
    """
    if not model_name:
        return False
    lowered = model_name.lower()
    if any(keyword in lowered for keyword in RESPONSES_ONLY_KEYWORDS):
        return True

    # Проверяем секцию "code" в реестре моделей.
    registry = load_registry()
    code_models = registry.get("code") or []
    if isinstance(code_models, list):
        return lowered in {str(item).lower() for item in code_models if isinstance(item, str)}
    return False
