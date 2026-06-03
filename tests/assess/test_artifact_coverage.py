"""Static runtime artifact coverage checks."""

from __future__ import annotations

import json
from pathlib import Path

from g2p.constants import ALL_PHONEMES, STRESS


SEGMENT_PHONEMES = frozenset(phone for phone in ALL_PHONEMES if phone != STRESS)


def test_gmm_quality_artifact_covers_all_49_segment_phones() -> None:
    data = json.loads(Path("configs/coach_gmm_v1.json").read_text(encoding="utf-8"))
    phones = set(data.get("phones", {}))

    assert len(SEGMENT_PHONEMES) == 49
    assert SEGMENT_PHONEMES <= phones
    assert {"yː", "ɯː"} <= phones


def test_decision_authority_covers_all_directional_non_self_pairs() -> None:
    data = json.loads(Path("configs/decision_authority.json").read_text(encoding="utf-8"))
    rows = data.get("contrasts")
    assert isinstance(rows, list)
    assert data.get("artifact_schema_version") == 3
    pairs = {
        (str(row.get("target")), str(row.get("alternative")))
        for row in rows
        if isinstance(row, dict)
    }
    expected = {
        (target, alternative)
        for target in SEGMENT_PHONEMES
        for alternative in SEGMENT_PHONEMES
        if target != alternative
    }

    assert len(expected) == 49 * 48
    assert pairs == expected
    assert all(
        isinstance(row, dict)
        and row.get("override_policy") in {"hard_override", "quality_only", "no_override"}
        and "threshold" in row
        and "fp_estimate" in row
        and "reason" in row
        for row in rows
    )
    assert all(
        not isinstance(row, dict)
        or row.get("override_policy") != "hard_override"
        or isinstance(row.get("detector"), dict)
        and bool(row.get("detector", {}).get("features"))
        for row in rows
    )
