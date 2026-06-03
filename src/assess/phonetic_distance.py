"""Generic soft phonetic distance scoring for target-observed phone pairs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from g2p.constants import (
    ALL_CONSONANTS,
    ALL_VOWELS,
    ASPIRATED_MAP,
    BACK_VOWELS,
    FRONT_VOWELS,
    LONG_TO_SHORT,
    ROUNDED_VOWELS,
    SONORANTS,
    UNROUNDED_VOWELS,
    VOICED_OBSTRUENTS,
    VOICELESS_CONS,
)

_COLON_LENGTH: dict[str, str] = {
    "a:": "aː",
    "e:": "eː",
    "i:": "iː",
    "o:": "oː",
    "u:": "uː",
    "y:": "yː",
    "œ:": "œː",
    "ɯ:": "ɯː",
}

_VOWEL_HEIGHT: dict[str, int] = {
    "i": 0,
    "y": 0,
    "ɯ": 0,
    "u": 0,
    "e": 1,
    "œ": 1,
    "o": 1,
    "æ": 2,
    "a": 2,
}

_CONSONANT_FEATURES: dict[str, dict[str, object]] = {
    "p": {"place": "bilabial", "manner": "stop", "voiced": False},
    "pʰ": {"place": "bilabial", "manner": "stop", "voiced": False, "aspirated": True},
    "b": {"place": "bilabial", "manner": "stop", "voiced": True},
    "t": {"place": "alveolar", "manner": "stop", "voiced": False},
    "tʰ": {"place": "alveolar", "manner": "stop", "voiced": False, "aspirated": True},
    "d": {"place": "alveolar", "manner": "stop", "voiced": True},
    "k": {"place": "velar", "manner": "stop", "voiced": False},
    "kʰ": {"place": "velar", "manner": "stop", "voiced": False, "aspirated": True},
    "ɡ": {"place": "velar", "manner": "stop", "voiced": True},
    "c": {"place": "palatal", "manner": "stop", "voiced": False},
    "cʰ": {"place": "palatal", "manner": "stop", "voiced": False, "aspirated": True},
    "ɟ": {"place": "palatal", "manner": "stop", "voiced": True},
    "t͡ʃ": {"place": "postalveolar", "manner": "affricate", "voiced": False},
    "d͡ʒ": {"place": "postalveolar", "manner": "affricate", "voiced": True},
    "f": {"place": "labiodental", "manner": "fricative", "voiced": False},
    "v": {"place": "labiodental", "manner": "fricative", "voiced": True},
    "β": {"place": "bilabial", "manner": "fricative", "voiced": True},
    "β̞": {"place": "bilabial", "manner": "approximant", "voiced": True},
    "s": {"place": "alveolar", "manner": "fricative", "voiced": False},
    "z": {"place": "alveolar", "manner": "fricative", "voiced": True},
    "ʃ": {"place": "postalveolar", "manner": "fricative", "voiced": False},
    "ʒ": {"place": "postalveolar", "manner": "fricative", "voiced": True},
    "h": {"place": "glottal", "manner": "fricative", "voiced": False},
    "m": {"place": "bilabial", "manner": "nasal", "voiced": True},
    "n": {"place": "alveolar", "manner": "nasal", "voiced": True},
    "ɲ": {"place": "palatal", "manner": "nasal", "voiced": True},
    "ŋ": {"place": "velar", "manner": "nasal", "voiced": True},
    "l": {"place": "alveolar", "manner": "lateral", "voiced": True},
    "ɫ": {
        "place": "alveolar",
        "manner": "lateral",
        "voiced": True,
        "secondary": "velarized",
    },
    "ɾ": {"place": "alveolar", "manner": "tap", "voiced": True},
    "ɾ̞̊": {"place": "alveolar", "manner": "tap", "voiced": False},
    "j": {"place": "palatal", "manner": "approximant", "voiced": True},
}

_CLOSE_PLACE_PENALTIES: dict[frozenset[str], float] = {
    frozenset(("bilabial", "labiodental")): 0.12,
    frozenset(("alveolar", "postalveolar")): 0.14,
    frozenset(("postalveolar", "palatal")): 0.18,
    frozenset(("palatal", "velar")): 0.20,
    frozenset(("alveolar", "palatal")): 0.24,
}


@dataclass(frozen=True)
class PhoneDistanceScore:
    score: float
    reason: str
    distance_features: dict[str, object]

    @property
    def score_percent(self) -> float:
        return self.score * 100.0


def score_phone(
    expected: str | None,
    observed: str | None,
    *,
    raw_observed: str | None = None,
    analysis_observed: str | None = None,
    duration_evidence: Mapping[str, object] | None = None,
    pairwise_candidates: list[Mapping[str, object]] | None = None,
) -> PhoneDistanceScore:
    """Return one normalized 0-1 score for a target phone.

    This scorer is intentionally generic: it uses phone features, optional
    target-length evidence, and existing pairwise detector rows only as small
    nudges. Exact analysis correctness always remains full credit.
    """
    if expected is None:
        return PhoneDistanceScore(
            score=0.0,
            reason="extra_penalty",
            distance_features={"status": "extra"},
        )
    target = normalize_phone(expected)
    if observed is None:
        return PhoneDistanceScore(
            score=0.0,
            reason="missing",
            distance_features={"status": "missing", "expected": target},
        )
    heard = normalize_phone(observed)
    raw = normalize_phone(raw_observed) if raw_observed is not None else None
    analysis = (
        normalize_phone(analysis_observed)
        if analysis_observed is not None
        else heard
    )

    if heard == target:
        reason = "analysis_corrected" if raw is not None and raw != target else "exact"
        return PhoneDistanceScore(
            score=1.0,
            reason=reason,
            distance_features={
                "class": _match_entry(_broad_class(target), _broad_class(heard)),
                "exact": True,
                "raw_observed": raw,
                "analysis_observed": analysis,
            },
        )

    score, features = _base_distance_score(target, heard)
    modifiers: list[dict[str, object]] = []
    score = _apply_duration_adjustment(
        score,
        target,
        heard,
        duration_evidence,
        modifiers,
    )
    score = _apply_pairwise_nudge(
        score,
        target,
        heard,
        raw,
        pairwise_candidates or [],
        modifiers,
    )
    if modifiers:
        features["modifiers"] = modifiers
    return PhoneDistanceScore(
        score=round(_clamp(score), 6),
        reason="phonetic_distance",
        distance_features=features,
    )


def normalize_phone(phone: str) -> str:
    return _COLON_LENGTH.get(phone, phone)


def _base_distance_score(target: str, observed: str) -> tuple[float, dict[str, object]]:
    target_class = _broad_class(target)
    observed_class = _broad_class(observed)
    features: dict[str, object] = {
        "class": _match_entry(target_class, observed_class),
    }
    if target_class != observed_class:
        features["differences"] = ["class"]
        return 0.05, features
    if target_class == "vowel":
        return _vowel_score(target, observed, features)
    if target_class == "consonant":
        return _consonant_score(target, observed, features)
    features["differences"] = ["unknown_phone"]
    return 0.10, features


def _vowel_score(
    target: str,
    observed: str,
    features: dict[str, object],
) -> tuple[float, dict[str, object]]:
    t = _vowel_features(target)
    o = _vowel_features(observed)
    penalties: list[tuple[str, float]] = []
    height_delta = abs(int(t["height"]) - int(o["height"]))
    if height_delta:
        penalties.append(("height", min(0.24, 0.12 * height_delta)))
    if t["backness"] != o["backness"]:
        penalties.append(("backness", 0.20))
    if t["rounding"] != o["rounding"]:
        penalties.append(("rounding", 0.16))
    if t["length"] != o["length"]:
        penalties.append(("length", 0.24))
    features.update(
        {
            "height": _match_entry(t["height_label"], o["height_label"]),
            "backness": _match_entry(t["backness"], o["backness"]),
            "rounding": _match_entry(t["rounding"], o["rounding"]),
            "length": _match_entry(t["length"], o["length"]),
            "differences": [name for name, _ in penalties],
        }
    )
    return max(0.30, 1.0 - sum(value for _, value in penalties)), features


def _consonant_score(
    target: str,
    observed: str,
    features: dict[str, object],
) -> tuple[float, dict[str, object]]:
    t = _consonant_features(target)
    o = _consonant_features(observed)
    if t is None or o is None:
        features["differences"] = ["unknown_consonant"]
        return 0.15, features

    penalties: list[tuple[str, float]] = []
    manner_penalty = _manner_penalty(str(t["manner"]), str(o["manner"]))
    if manner_penalty:
        penalties.append(("manner", manner_penalty))
    place_penalty = _place_penalty(str(t["place"]), str(o["place"]))
    if place_penalty:
        penalties.append(("place", place_penalty))
    if bool(t["voiced"]) != bool(o["voiced"]):
        penalties.append(("voicing", 0.12 if _sonorant_like(t, o) else 0.17))
    if bool(t["aspirated"]) != bool(o["aspirated"]):
        penalties.append(("aspiration", 0.10))
    if t["secondary"] != o["secondary"]:
        penalties.append(("secondary", 0.07))

    features.update(
        {
            "place": _match_entry(t["place"], o["place"]),
            "manner": _match_entry(t["manner"], o["manner"]),
            "voicing": _match_entry(t["voiced"], o["voiced"]),
            "aspiration": _match_entry(t["aspirated"], o["aspirated"]),
            "secondary": _match_entry(t["secondary"], o["secondary"]),
            "differences": [name for name, _ in penalties],
        }
    )
    return max(0.12, 1.0 - sum(value for _, value in penalties)), features


def _apply_duration_adjustment(
    score: float,
    target: str,
    observed: str,
    duration_evidence: Mapping[str, object] | None,
    modifiers: list[dict[str, object]],
) -> float:
    if duration_evidence is None or not _same_vowel_quality(target, observed):
        return score
    if _is_long(target) == _is_long(observed):
        return score
    reliable = duration_evidence.get("reliable") is True
    note = duration_evidence.get("note")
    modifiers.append(
        {
            "source": "duration",
            "note": note,
            "reliable": reliable,
            "score": _finite_float(duration_evidence.get("score")),
        }
    )
    if not reliable or note not in {"short", "expected", "long"}:
        return score
    if _is_long(target):
        if note in {"expected", "long"}:
            modifiers[-1]["effect"] = "supports_target_length"
            return max(score, 0.90)
        modifiers[-1]["effect"] = "rejects_target_length"
        return min(score, 0.58)
    if note in {"expected", "short"}:
        modifiers[-1]["effect"] = "supports_target_length"
        return max(score, 0.90)
    modifiers[-1]["effect"] = "rejects_target_length"
    return min(score, 0.58)


def _apply_pairwise_nudge(
    score: float,
    target: str,
    observed: str,
    raw_observed: str | None,
    candidates: list[Mapping[str, object]],
    modifiers: list[dict[str, object]],
) -> float:
    if not candidates or target == observed:
        return score
    adjusted = score
    for candidate in candidates:
        if str(candidate.get("source") or "") != "pairwise_contrast":
            continue
        candidate_from = candidate.get("from")
        if raw_observed is not None and candidate_from is not None:
            if normalize_phone(str(candidate_from)) != raw_observed:
                continue
        to_phone = candidate.get("to")
        if to_phone is None:
            continue
        to_phone = normalize_phone(str(to_phone))
        passed = candidate.get("passed") is True
        margin = _candidate_margin(candidate)
        if to_phone == target and passed:
            adjusted = min(0.95, adjusted + 0.07)
            modifiers.append(
                {
                    "source": "pairwise",
                    "effect": "supports_target",
                    "to": to_phone,
                    "margin": margin,
                }
            )
        elif to_phone == target and _strong_pairwise_reject(candidate):
            adjusted = max(0.0, adjusted - 0.05)
            modifiers.append(
                {
                    "source": "pairwise",
                    "effect": "rejects_target",
                    "to": to_phone,
                    "margin": margin,
                }
            )
        elif to_phone == observed and passed:
            adjusted = max(0.0, adjusted - 0.05)
            modifiers.append(
                {
                    "source": "pairwise",
                    "effect": "supports_observed",
                    "to": to_phone,
                    "margin": margin,
                }
            )
    return adjusted


def _strong_pairwise_reject(candidate: Mapping[str, object]) -> bool:
    margin = _candidate_margin(candidate)
    if margin is not None and margin <= -0.05:
        return True
    return any(
        "below" in reason or "not_passed" in reason
        for reason in _candidate_reasons(candidate)
    )


def _candidate_margin(candidate: Mapping[str, object]) -> float | None:
    margin = _finite_float(candidate.get("margin"))
    if margin is not None:
        return margin
    evidence = candidate.get("evidence")
    if isinstance(evidence, Mapping):
        return _finite_float(evidence.get("margin"))
    return None


def _candidate_reasons(candidate: Mapping[str, object]) -> list[str]:
    raw = candidate.get("reasons")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str)]


def _vowel_features(phone: str) -> dict[str, object]:
    base = _short_vowel(phone)
    height = _VOWEL_HEIGHT.get(base, 1)
    return {
        "height": height,
        "height_label": ("high" if height == 0 else "mid" if height == 1 else "low"),
        "backness": "front" if base in FRONT_VOWELS else "back" if base in BACK_VOWELS else "unknown",
        "rounding": "rounded" if base in ROUNDED_VOWELS else "unrounded" if base in UNROUNDED_VOWELS else "unknown",
        "length": "long" if _is_long(phone) else "short",
    }


def _consonant_features(phone: str) -> dict[str, object] | None:
    if phone in _CONSONANT_FEATURES:
        data = dict(_CONSONANT_FEATURES[phone])
    else:
        data = {}
    if not data:
        return None
    data.setdefault("aspirated", phone in ASPIRATED_MAP.values() or phone.endswith("ʰ"))
    data.setdefault(
        "voiced",
        phone in VOICED_OBSTRUENTS or phone in SONORANTS or phone not in VOICELESS_CONS,
    )
    data.setdefault("secondary", None)
    return data


def _broad_class(phone: str) -> str:
    base = _short_vowel(phone)
    if base in ALL_VOWELS:
        return "vowel"
    if phone in ALL_CONSONANTS:
        return "consonant"
    return "unknown"


def _manner_penalty(target: str, observed: str) -> float:
    if target == observed:
        return 0.0
    pair = {target, observed}
    if pair <= {"stop", "affricate"}:
        return 0.20
    if pair <= {"fricative", "affricate"}:
        return 0.22
    obstruents = {"stop", "affricate", "fricative"}
    sonorants = {"nasal", "lateral", "tap", "approximant"}
    if target in obstruents and observed in obstruents:
        return 0.38
    if target in sonorants and observed in sonorants:
        return 0.28
    return 0.46


def _place_penalty(target: str, observed: str) -> float:
    if target == observed:
        return 0.0
    close = _CLOSE_PLACE_PENALTIES.get(frozenset((target, observed)))
    if close is not None:
        return close
    coronals = {"alveolar", "postalveolar", "palatal"}
    dorsals = {"palatal", "velar"}
    if target in coronals and observed in coronals:
        return 0.28
    if target in dorsals and observed in dorsals:
        return 0.28
    if "glottal" in {target, observed}:
        return 0.42
    return 0.36


def _same_vowel_quality(target: str, observed: str) -> bool:
    return _short_vowel(target) == _short_vowel(observed) and _broad_class(target) == "vowel"


def _short_vowel(phone: str) -> str:
    return LONG_TO_SHORT.get(phone, phone)


def _is_long(phone: str) -> bool:
    return phone in LONG_TO_SHORT or phone.endswith("ː")


def _sonorant_like(first: Mapping[str, object], second: Mapping[str, object]) -> bool:
    manners = {str(first["manner"]), str(second["manner"])}
    return manners <= {"nasal", "lateral", "tap", "approximant"}


def _match_entry(expected: object, observed: object) -> dict[str, object]:
    return {
        "expected": expected,
        "observed": observed,
        "match": expected == observed,
    }


def _finite_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
