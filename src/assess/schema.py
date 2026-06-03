"""Versioned, JSON-stable output contract.

API and evaluation interfaces consume this schema; any field change requires
a SCHEMA_VERSION bump. All dataclasses serialise to plain JSON via as_dict()
(IPA characters preserved with ensure_ascii=False).
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 9


@dataclass(frozen=True)
class PhoneTiming:
    ipa: str
    start_ms: int
    end_ms: int
    confidence: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "ipa": self.ipa,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class TranscribeResult:
    model: str
    ipa_string: str
    phones: list[PhoneTiming]
    status: str = "ok"
    error_code: str | None = None
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "error_code": self.error_code,
            "model": self.model,
            "ipa_string": self.ipa_string,
            "phones": [p.as_dict() for p in self.phones],
        }
