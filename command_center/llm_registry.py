import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings


_REGISTRY_CACHE: Dict[str, Any] | None = None
_REGISTRY_MTIME: float | None = None


def _registry_path() -> Path:
    return Path(settings.BASE_DIR) / "command_center" / "config" / "models_registry.json"


def load_registry() -> Dict[str, Any]:
    """
    Загружает models_registry.json c простым in-memory кэшем.
    При отсутствии файла или ошибке парсинга возвращает пустой словарь.
    """
    global _REGISTRY_CACHE, _REGISTRY_MTIME

    path = _registry_path()
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        _REGISTRY_CACHE = {}
        _REGISTRY_MTIME = None
        return {}

    if _REGISTRY_CACHE is not None and _REGISTRY_MTIME == mtime:
        return _REGISTRY_CACHE

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _REGISTRY_CACHE = {}
        _REGISTRY_MTIME = mtime
        return {}

    _REGISTRY_CACHE = data
    _REGISTRY_MTIME = mtime
    return data


def get_chat_primary() -> Optional[str]:
    registry = load_registry()
    chat = registry.get("chat") or {}
    primary = chat.get("primary")
    if isinstance(primary, str) and primary.strip():
        return primary.strip()
    return None


def get_embedding_default() -> Optional[str]:
    registry = load_registry()
    embedding = registry.get("embedding") or {}
    default = embedding.get("default")
    if isinstance(default, str) and default.strip():
        return default.strip()
    return None


def get_chat_recommended() -> List[str]:
    registry = load_registry()
    chat = registry.get("chat") or {}
    rec = chat.get("recommended") or []
    if not isinstance(rec, list):
        return []
    return [str(x) for x in rec if isinstance(x, str)]


def get_all_models_by_type(kind: str) -> List[Dict[str, Any]]:
    """
    Возвращает список моделей указанного типа в виде
    [{"name": "...", ...}, ...].

    Типы:
    - "chat", "embedding": берутся из вложенных секций.
    - "lightweight", "code", "realtime", "search", "deprecated": из верхнеуровневых списков.
    """
    registry = load_registry()

    if kind in {"chat", "embedding"}:
        section = registry.get(kind) or {}
        models = section.get("models")
        if not isinstance(models, list):
            # Совместимость: если отдельного списка models нет, собираем из primary/recommended
            names: List[str] = []
            if kind == "chat":
                primary = section.get("primary")
                if isinstance(primary, str):
                    names.append(primary)
                rec = section.get("recommended") or []
                if isinstance(rec, list):
                    names.extend(str(x) for x in rec if isinstance(x, str))
            elif kind == "embedding":
                default = section.get("default")
                if isinstance(default, str):
                    names.append(default)
                rec = section.get("recommended") or []
                if isinstance(rec, list):
                    names.extend(str(x) for x in rec if isinstance(x, str))
            # Убираем дубли
            seen = set()
            models = []
            for name in names:
                if name not in seen:
                    seen.add(name)
                    models.append(name)
        return [{"name": str(m)} for m in models if isinstance(m, str)]

    # Остальные секции представлены как списки строк.
    values = registry.get(kind) or []
    if not isinstance(values, list):
        return []
    return [{"name": str(v)} for v in values if isinstance(v, str)]


def get_all_known_model_names() -> List[str]:
    """
    Утилита для валидации: возвращает множество всех известных имён моделей.
    """
    registry = load_registry()
    names: set[str] = set()

    chat = registry.get("chat") or {}
    if isinstance(chat.get("primary"), str):
        names.add(chat["primary"])
    rec = chat.get("recommended") or []
    if isinstance(rec, list):
        names.update(str(x) for x in rec if isinstance(x, str))

    embedding = registry.get("embedding") or {}
    if isinstance(embedding.get("default"), str):
        names.add(embedding["default"])
    emb_rec = embedding.get("recommended") or []
    if isinstance(emb_rec, list):
        names.update(str(x) for x in emb_rec if isinstance(x, str))

    for key in ("lightweight", "code", "realtime", "search", "deprecated"):
        vals = registry.get(key) or []
        if isinstance(vals, list):
            names.update(str(x) for x in vals if isinstance(x, str))

    return sorted(names)


def is_model_deprecated(name: str) -> bool:
    """
    Возвращает True, если модель помечена как deprecated в реестре.
    """
    deprecated = get_all_models_by_type("deprecated")
    for item in deprecated:
        if item.get("name") == name:
            return True
    return False

