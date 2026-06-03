from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


@dataclass(frozen=True)
class AppConfig:
    data_root: Path = DEFAULT_DATA_ROOT

    @property
    def sandbox_dir(self) -> Path:
        return self.data_root / "app" / "sandbox"

    @property
    def app_recordings_dir(self) -> Path:
        return self.data_root / "app" / "recordings"

    @property
    def validation_dir(self) -> Path:
        return self.data_root / "validation"

    @property
    def prompts_tsv(self) -> Path:
        return self.validation_dir / "prompts.tsv"

    @property
    def recordings_dir(self) -> Path:
        return self.validation_dir / "recordings"

    @property
    def demo_capture_dir(self) -> Path:
        return self.data_root / "demo_capture"

    @property
    def demo_prompts_tsv(self) -> Path:
        return self.demo_capture_dir / "prompts.tsv"

    @property
    def demo_recordings_dir(self) -> Path:
        return self.demo_capture_dir / "recordings"


def load_config(*, data_root: Path | None = None) -> AppConfig:
    return AppConfig(data_root=data_root or DEFAULT_DATA_ROOT)
