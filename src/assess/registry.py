"""Model registry — alias → wav2vec checkpoint resolution.

Reads from configs/models.toml. A missing checkpoint is not an error
(available=False); the calling layer returns a graceful model_unavailable
response.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "models.toml"


class UnknownModelError(KeyError):
    """Requested alias is not in the registry."""


@dataclass(frozen=True)
class ModelEntry:
    alias: str
    display_name: str
    checkpoint: Path
    processor: Path
    frozen: bool

    @property
    def available(self) -> bool:
        return self.checkpoint.exists()


class ModelRegistry:
    def __init__(self, entries: dict[str, ModelEntry], default_alias: str) -> None:
        if default_alias not in entries:
            raise ValueError(f"default {default_alias!r} not in registry")
        self._entries = entries
        self._default = default_alias

    @classmethod
    def from_config(
        cls,
        config_path: Path | None = None,
        *,
        base_dir: Path | None = None,
    ) -> ModelRegistry:
        path = config_path or DEFAULT_CONFIG
        root = base_dir or REPO_ROOT
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        default_alias = str(data["default"])
        entries: dict[str, ModelEntry] = {}
        for alias, spec in data.get("models", {}).items():
            entries[alias] = ModelEntry(
                alias=alias,
                display_name=str(spec["display_name"]),
                checkpoint=cls._resolve(root, spec["checkpoint"]),
                processor=cls._resolve(root, spec["processor"]),
                frozen=bool(spec.get("frozen", False)),
            )
        return cls(entries, default_alias)

    @staticmethod
    def _resolve(root: Path, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (root / p)

    @property
    def default_alias(self) -> str:
        return self._default

    def aliases(self) -> set[str]:
        return set(self._entries)

    def get(self, alias: str | None = None) -> ModelEntry:
        key = alias or self._default
        try:
            return self._entries[key]
        except KeyError as exc:
            raise UnknownModelError(key) from exc

    def list(self) -> list[dict[str, object]]:
        return [
            {
                "alias": e.alias,
                "display_name": e.display_name,
                "frozen": e.frozen,
                "available": e.available,
                "checkpoint": str(e.checkpoint),
            }
            for e in self._entries.values()
        ]
