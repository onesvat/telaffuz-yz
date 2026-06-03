"""assess.registry — alias→checkpoint çözümü + graceful availability."""

from __future__ import annotations

from pathlib import Path

import pytest

from assess.registry import ModelRegistry, UnknownModelError


def _write_config(tmp_path: Path, *, ckpt_rel: str, missing_rel: str) -> Path:
    (tmp_path / ckpt_rel).mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "models.toml"
    cfg.write_text(
        "default = \"mms-1b\"\n\n"
        "[models.mms-1b]\n"
        'display_name = "MMS-1B test"\n'
        f'checkpoint = "{ckpt_rel}"\n'
        f'processor = "{ckpt_rel}"\n'
        "frozen = true\n\n"
        "[models.xls-r-300m]\n"
        'display_name = "XLS-R test"\n'
        f'checkpoint = "{missing_rel}"\n'
        f'processor = "{missing_rel}"\n'
        "frozen = false\n",
        encoding="utf-8",
    )
    return cfg


def test_default_alias_and_known_aliases(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, ckpt_rel="runs/a", missing_rel="runs/b")
    reg = ModelRegistry.from_config(cfg, base_dir=tmp_path)

    assert reg.default_alias == "mms-1b"
    assert reg.aliases() == {"mms-1b", "xls-r-300m"}


def test_get_resolves_relative_paths_against_base_dir(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, ckpt_rel="runs/a", missing_rel="runs/b")
    reg = ModelRegistry.from_config(cfg, base_dir=tmp_path)

    entry = reg.get("mms-1b")
    assert entry.checkpoint == tmp_path / "runs" / "a"
    assert entry.checkpoint.is_absolute()
    assert entry.available is True


def test_get_none_returns_default(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, ckpt_rel="runs/a", missing_rel="runs/b")
    reg = ModelRegistry.from_config(cfg, base_dir=tmp_path)

    assert reg.get().alias == "mms-1b"


def test_missing_checkpoint_is_unavailable_not_error(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, ckpt_rel="runs/a", missing_rel="runs/gone")
    reg = ModelRegistry.from_config(cfg, base_dir=tmp_path)

    entry = reg.get("xls-r-300m")
    assert entry.available is False  # graceful: yokluk hata değil


def test_unknown_alias_raises(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, ckpt_rel="runs/a", missing_rel="runs/b")
    reg = ModelRegistry.from_config(cfg, base_dir=tmp_path)

    with pytest.raises(UnknownModelError):
        reg.get("does-not-exist")


def test_list_has_alias_display_available(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path, ckpt_rel="runs/a", missing_rel="runs/gone")
    reg = ModelRegistry.from_config(cfg, base_dir=tmp_path)

    items = {item["alias"]: item for item in reg.list()}
    assert items["mms-1b"]["display_name"] == "MMS-1B test"
    assert items["mms-1b"]["available"] is True
    assert items["xls-r-300m"]["available"] is False


def test_shipped_config_loads_with_expected_aliases() -> None:
    """Repo'daki configs/models.toml parse edilebilir + tutarlı olmalı."""
    reg = ModelRegistry.from_config()

    assert reg.default_alias == "mms-1b"
    assert reg.aliases() == {"mms-1b", "xls-r-300m"}
