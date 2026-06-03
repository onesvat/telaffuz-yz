"""CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from g2p.cli import main


def test_cli_default_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ekmek"]) == 0
    assert capsys.readouterr().out.strip() == "/ek.ˈmec/"


def test_cli_phones_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--phones", "ekmek"]) == 0
    assert capsys.readouterr().out.strip() == "e k ˈ m e c"


def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "ekmek"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["word"] == "ekmek"
    assert payload["ipa"] == "/ek.ˈmec/"
    assert payload["phonemes"] == ["e", "k", "ˈ", "m", "e", "c"]


def test_cli_no_reference_uses_rule_pipeline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--json", "--no-reference", "ekmek"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "rules"


def test_cli_no_pedagogical_strips_aspiration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--no-reference", "--no-pedagogical", "para"]) == 0
    assert capsys.readouterr().out.strip() == "/pa.ˈɾa/"


def test_cli_no_pedagogical_strips_r_devoicing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--no-reference", "--no-pedagogical", "bir"]) == 0
    assert capsys.readouterr().out.strip() == "/biɾ/"


def test_cli_text_joins_positional_args(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["anne", "baba"]) == 0
    assert capsys.readouterr().out.strip() == "/an.ˈne ba.ˈba/"


def test_cli_batch_tsv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    batch = tmp_path / "words.txt"
    batch.write_text("ekmek\nboğaz\n", encoding="utf-8")
    assert main(["--batch", str(batch)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ekmek\t/ek.ˈmec/",
        "boğaz\t/bo.ˈaz/",
    ]


def test_cli_batch_jsonl(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    batch = tmp_path / "words.txt"
    batch.write_text("ekmek\nboğaz\n", encoding="utf-8")
    assert main(["--json", "--batch", str(batch)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["word"] for row in rows] == ["ekmek", "boğaz"]
    assert [row["source"] for row in rows] == ["rules", "rules"]


def test_cli_empty_input_returns_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out
