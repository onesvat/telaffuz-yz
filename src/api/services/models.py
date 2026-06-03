"""Wav2vec model registry surface exposed to the demo API."""

from __future__ import annotations

from assess.registry import ModelEntry, ModelRegistry


def _load_registry() -> ModelRegistry | None:
    try:
        return ModelRegistry.from_config()
    except Exception:
        return None


def default_alias() -> str:
    registry = _load_registry()
    return registry.default_alias if registry is not None else ""


def list_models() -> list[dict[str, object]]:
    registry = _load_registry()
    if registry is None:
        return []
    default = registry.default_alias
    items: list[dict[str, object]] = []
    for alias in sorted(registry.aliases(), key=lambda a: (a != default, a)):
        entry = registry.get(alias)
        items.append(
            {
                "id": alias,
                "label": entry.display_name,
                "available": entry.available,
                "frozen": entry.frozen,
                "is_default": alias == default,
            }
        )
    return items


def resolve_alias(alias: str) -> ModelEntry | None:
    registry = _load_registry()
    if registry is None:
        return None
    alias = alias.strip()
    if not alias or alias not in registry.aliases():
        return None
    return registry.get(alias)
