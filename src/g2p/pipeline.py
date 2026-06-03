"""G2P pipeline orchestration."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass

from g2p.constants import (
    ALL_PHONEMES,
    CTC_BLANK,
    CTC_PAD,
    CTC_SPC,
    CTC_UNK,
    STRESS,
)
from g2p.exceptions import ExceptionEntry, lookup_case_insensitive
from g2p.grapheme_map import _lower_tr, to_ipa
from g2p.morphology import MorphAnalysis, analyze
from g2p.phones import canonicalize_stress_position, parse_ipa_text
from g2p.rules import aspiration, e_open, l_allophony, n_assimilation
from g2p.rules import palatalization, r_devoicing, soft_g, stress
from g2p.rules import v_allophony, vowel_harmony
from g2p.syllabifier import syllabify


@dataclass(frozen=True)
class TraceStep:
    """Single observable step of the G2P pipeline.

    ``fired`` is True iff ``input`` and ``output`` differ; the trace UI uses
    this to dim untriggered rules. ``note`` carries optional context
    (e.g. matched exception type, morphology root, reference IPA string).
    """

    name: str
    input: tuple[str, ...]
    output: tuple[str, ...]
    fired: bool
    note: str = ""


@dataclass(frozen=True)
class G2PResult:
    word: str
    normalized: str
    phonemes: tuple[str, ...]
    syllables: tuple[tuple[str, ...], ...]
    ipa: str
    source: str
    exception_type: str | None
    warnings: tuple[str, ...] = ()
    trace: tuple[TraceStep, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(frozen=True)
class TextG2PWord:
    text: str
    normalized: str
    tokens: tuple[str, ...]
    token_count: int
    ipa: str
    source: str
    exception_type: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class TextG2PResult:
    text: str
    normalized: str
    tokens: tuple[str, ...]
    words: tuple[TextG2PWord, ...]
    warnings: tuple[str, ...]
    token_count: int
    drop_reason: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


TRAINING_FORBIDDEN_TOKENS = frozenset({CTC_BLANK, CTC_PAD, CTC_UNK, "<sil>"})
TRAINING_ALLOWED_TOKENS = ALL_PHONEMES | {CTC_SPC}
_TEXT_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


REFERENCE_IPA: dict[str, str] = {
    # NOT: G2P motoru "final" duruma getirildiğinde REFERENCE_IPA küçültüldü.
    # Çoğu örnek artık kural+exception_seed kombinasyonu ile doğru üretiliyor.
    # Burada sadece kısaltma/akronim ve sabit fonotaktik test örnekleri tutulur.
    "tbmm": "/teˈbeˈmeˈme/",
    "trt": "/teˈɾeˈte/",
    "kdv": "/kaːˈdeˈve/",
    "avm": "/aˈveˈme/",
}


def normalize_word(word: str) -> str:
    normalized = unicodedata.normalize("NFC", word.strip())
    return normalized.strip(' \t\n\r.,;:!?()[]{}"“”')


def transcribe_word(
    word: str,
    *,
    use_reference: bool = True,
    pedagogical_allophones: bool = True,
    trace: bool = False,
) -> G2PResult:
    recorder: list[TraceStep] | None = [] if trace else None
    normalized = normalize_word(word)
    if recorder is not None:
        _record(
            recorder, "normalize", (word,), (normalized,),
            note="strip + NFC; punctuation trimmed",
        )
    warnings: list[str] = []
    if not normalized:
        return _result(word, "", [], "empty", None, warnings, recorder)

    reference = REFERENCE_IPA.get(_lower_tr(normalized)) if use_reference else None
    if recorder is not None:
        _record(
            recorder,
            "reference_lookup",
            (normalized,),
            (normalized,),
            fired=reference is not None,
            note=f"hit: {reference}" if reference else "miss",
        )
    if reference is not None:
        phones = parse_ipa(reference)
        if recorder is not None:
            _record(recorder, "parse_reference_ipa", (reference,), tuple(phones))
        return _result(word, normalized, phones, "reference", None, warnings, recorder)

    entry = lookup_case_insensitive(normalized)
    if recorder is not None:
        _record(
            recorder,
            "exception_lookup",
            (normalized,),
            (normalized,),
            fired=entry is not None,
            note=(
                f"hit: type={entry.type}" if entry is not None else "miss"
            ),
        )
    if entry is not None:
        phonemes = list(entry.phonemes)
        source = "exception"
        exception_type = entry.type
    elif _is_acronym(normalized):
        phonemes = _acronym_phonemes(normalized)
        source = "acronym"
        exception_type = None
        if recorder is not None:
            _record(
                recorder,
                "acronym_expand",
                (normalized,),
                tuple(phonemes),
                note="acronym letter-by-letter",
            )
    else:
        phonemes = to_ipa(normalized)
        source = "rules"
        exception_type = None
        if recorder is not None:
            _record(
                recorder,
                "grapheme_map",
                (normalized,),
                tuple(phonemes),
                note="context-free letter→IPA",
            )

    morph = None if entry is not None else _safe_analyze(normalized)
    if recorder is not None and entry is None:
        if morph is not None:
            note = (
                f"root={morph.root}; "
                f"suffixes={'+'.join(morph.suffixes) or '∅'}; "
                f"pos={morph.pos}; proper={morph.is_proper}"
            )
        else:
            note = "morph analyzer unavailable or no analysis"
        _record(recorder, "morphology", (normalized,), (normalized,), fired=morph is not None, note=note)
    phonemes = _apply_rules(
        phonemes,
        word=normalized,
        entry=entry,
        morphology=morph,
        exception_type=exception_type or "",
        pedagogical_allophones=pedagogical_allophones,
        recorder=recorder,
    )
    warnings.extend(_inventory_warnings(phonemes))
    return _result(word, normalized, phonemes, source, exception_type, warnings, recorder)


def transcribe_text(
    text: str,
    *,
    use_reference: bool = True,
    pedagogical_allophones: bool = True,
    trace: bool = False,
) -> G2PResult:
    training = transcribe_training_text(
        text,
        use_reference=use_reference,
        pedagogical_allophones=pedagogical_allophones,
        trace=trace,
    )
    trace_steps: list[TraceStep] | None = None
    source = "text"
    exception_type: str | None = None
    if len(training.words) == 1:
        word = training.words[0]
        source = word.source
        exception_type = word.exception_type
        if trace:
            traced = transcribe_word(
                word.text,
                use_reference=use_reference,
                pedagogical_allophones=pedagogical_allophones,
                trace=True,
            )
            trace_steps = list(traced.trace)
    return _result(
        text,
        training.normalized,
        list(training.tokens),
        source,
        exception_type,
        list(training.warnings),
        trace_steps,
    )


def transcribe_training_text(
    text: str,
    *,
    use_reference: bool = True,
    pedagogical_allophones: bool = True,
    trace: bool = False,
) -> TextG2PResult:
    raw = unicodedata.normalize("NFC", text.strip())
    word_surfaces = _tokenize_training_words(raw)
    normalized = " ".join(word_surfaces)
    warnings: list[str] = []
    if not raw:
        return TextG2PResult(text=text, normalized="", tokens=(), words=(), warnings=(), token_count=0, drop_reason="empty_text")
    if not word_surfaces:
        return TextG2PResult(
            text=text,
            normalized="",
            tokens=(),
            words=(),
            warnings=("no word tokens found",),
            token_count=0,
            drop_reason="no_word_tokens",
        )

    words: list[TextG2PWord] = []
    tokens: list[str] = []
    for idx, surface in enumerate(word_surfaces):
        result = transcribe_word(
            surface,
            use_reference=use_reference,
            pedagogical_allophones=pedagogical_allophones,
            trace=trace and len(word_surfaces) == 1,
        )
        word_tokens = tuple(result.phonemes)
        word_warnings = list(result.warnings)
        if CTC_SPC in word_tokens:
            word_warnings.append("space token emitted inside word")
        if idx:
            tokens.append(CTC_SPC)
        tokens.extend(word_tokens)
        warnings.extend(f"{result.normalized}: {warning}" for warning in word_warnings)
        words.append(
            TextG2PWord(
                text=surface,
                normalized=result.normalized,
                tokens=word_tokens,
                token_count=len(word_tokens),
                ipa=result.ipa,
                source=result.source,
                exception_type=result.exception_type,
                warnings=tuple(word_warnings),
            )
        )

    validation_warnings, drop_reason = _training_token_policy(tokens)
    warnings.extend(validation_warnings)
    return TextG2PResult(
        text=text,
        normalized=normalized,
        tokens=tuple(tokens),
        words=tuple(words),
        warnings=tuple(warnings),
        token_count=len(tokens),
        drop_reason=drop_reason,
    )


def _tokenize_training_words(text: str) -> list[str]:
    return [match.group(0).strip("'’") for match in _TEXT_WORD_RE.finditer(text) if match.group(0).strip("'’")]


def _training_token_policy(tokens: list[str]) -> tuple[list[str], str | None]:
    warnings: list[str] = []
    drop_reason: str | None = None
    for idx, token in enumerate(tokens):
        reason: str | None = None
        if token in TRAINING_FORBIDDEN_TOKENS:
            warnings.append(f"forbidden emitted token: {token}")
            reason = f"forbidden_token:{token}"
        elif token not in TRAINING_ALLOWED_TOKENS:
            warnings.append(f"unknown emitted token: {token}")
            reason = f"unknown_token:{token}"
        elif token == CTC_SPC and (idx == 0 or idx == len(tokens) - 1):
            warnings.append("space token emitted at text boundary")
            reason = "space_token_boundary"
        elif token == CTC_SPC and tokens[idx - 1] == CTC_SPC:
            warnings.append("consecutive space tokens emitted")
            reason = "consecutive_space_tokens"
        if drop_reason is None and reason is not None:
            drop_reason = reason
    return warnings, drop_reason


def parse_ipa(ipa: str) -> list[str]:
    return list(parse_ipa_text(ipa).phones)


def format_ipa(phonemes: list[str] | tuple[str, ...]) -> str:
    phones = list(phonemes)
    if CTC_SPC in phones:
        chunks: list[str] = []
        current: list[str] = []
        for phone in phones:
            if phone == CTC_SPC:
                chunks.append(format_ipa(current).strip("/"))
                current = []
            else:
                current.append(phone)
        chunks.append(format_ipa(current).strip("/"))
        return "/" + " ".join(chunks) + "/"

    plain = [p for p in phones if p != STRESS]
    syllables = syllabify(plain)
    stress_idx = _stress_syllable_index(phones, syllables)
    parts: list[str] = []
    for idx, syllable in enumerate(syllables):
        prefix = STRESS if idx == stress_idx else ""
        parts.append(prefix + "".join(syllable))
    return "/" + ".".join(parts) + "/"


def _apply_rules(
    phonemes: list[str],
    *,
    word: str,
    entry: ExceptionEntry | None,
    morphology: MorphAnalysis | None,
    exception_type: str,
    pedagogical_allophones: bool = True,
    recorder: list[TraceStep] | None = None,
) -> list[str]:
    is_proper = bool(word[:1].isupper()) or bool(morphology and morphology.is_proper)

    phones = phonemes
    phones = _run(recorder, "vowel_harmony", phones, vowel_harmony.apply, phones)
    phones = _run(
        recorder, "soft_g", phones,
        soft_g.apply, phones, word=word, is_proper=is_proper,
    )
    phones = _run(recorder, "palatalization", phones, palatalization.apply, phones)
    phones = _run(
        recorder, "l_allophony", phones,
        l_allophony.apply, phones, word=word, exception_type=exception_type,
    )
    phones = _run(recorder, "v_allophony", phones, v_allophony.apply, phones)
    phones = _run(recorder, "n_assimilation", phones, n_assimilation.apply, phones)
    phones = _run(
        recorder, "e_open", phones,
        e_open.apply, phones, word=word, exception_type=exception_type,
    )
    if pedagogical_allophones:
        phones = _run(recorder, "aspiration", phones, aspiration.apply, phones)
        phones = _run(recorder, "r_devoicing", phones, r_devoicing.apply, phones)
    elif recorder is not None:
        _record(
            recorder, "aspiration", tuple(phones), tuple(phones),
            fired=False, note="skipped (pedagogical_allophones=False)",
        )
        _record(
            recorder, "r_devoicing", tuple(phones), tuple(phones),
            fired=False, note="skipped (pedagogical_allophones=False)",
        )
    phones = _run(
        recorder, "stress", phones,
        stress.apply, phones,
        word=word,
        exception_type=exception_type,
        ipa_detail=entry.ipa_detail if entry is not None else "",
        morphology=morphology,
    )
    final = list(canonicalize_stress_position(phones))
    if recorder is not None:
        _record(
            recorder, "canonicalize_stress", tuple(phones), tuple(final),
            note="move stress marker to syllable-onset (IPA standard)",
        )
    return final


def _run(
    recorder: list[TraceStep] | None,
    name: str,
    phones: list[str],
    fn,
    *args,
    **kwargs,
) -> list[str]:
    input_tokens = tuple(phones)
    output_tokens = list(fn(*args, **kwargs))
    if recorder is not None:
        _record(recorder, name, input_tokens, tuple(output_tokens))
    return output_tokens


def _record(
    recorder: list[TraceStep],
    name: str,
    input_tokens: tuple[str, ...],
    output_tokens: tuple[str, ...],
    *,
    fired: bool | None = None,
    note: str = "",
) -> None:
    recorder.append(
        TraceStep(
            name=name,
            input=input_tokens,
            output=output_tokens,
            fired=input_tokens != output_tokens if fired is None else fired,
            note=note,
        )
    )


def _result(
    word: str,
    normalized: str,
    phonemes: list[str],
    source: str,
    exception_type: str | None,
    warnings: list[str],
    trace: list[TraceStep] | None,
) -> G2PResult:
    plain = [p for p in phonemes if p not in {STRESS, CTC_SPC}]
    return G2PResult(
        word=word,
        normalized=normalized,
        phonemes=tuple(phonemes),
        syllables=tuple(tuple(s) for s in syllabify(plain)),
        ipa=format_ipa(phonemes),
        source=source,
        exception_type=exception_type,
        warnings=tuple(warnings),
        trace=tuple(trace) if trace is not None else (),
    )


def _safe_analyze(word: str) -> MorphAnalysis | None:
    try:
        return analyze(word)
    except Exception:
        return None


def _is_acronym(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return len(letters) > 1 and all(c.upper() == c for c in letters)


LETTER_IPA: dict[str, str] = {
    "a": "/a/",
    "b": "/ˈbe/",
    "c": "/ˈd͡ʒe/",
    "ç": "/ˈt͡ʃe/",
    "d": "/ˈde/",
    "e": "/e/",
    "f": "/ˈfe/",
    "g": "/ˈɟe/",
    "h": "/ˈhe/",
    "ı": "/ɯ/",
    "i": "/i/",
    "j": "/ˈʒe/",
    "k": "/kaː/",
    "l": "/ˈle/",
    "m": "/ˈme/",
    "n": "/ˈne/",
    "o": "/o/",
    "ö": "/œ/",
    "p": "/ˈpe/",
    "r": "/ˈɾe/",
    "s": "/ˈse/",
    "ş": "/ˈʃe/",
    "t": "/ˈte/",
    "u": "/u/",
    "ü": "/y/",
    "v": "/ˈve/",
    "w": "/t͡ʃiftˈve/",
    "x": "/iks/",
    "y": "/ˈje/",
    "z": "/ˈze/",
}


def _acronym_phonemes(word: str) -> list[str]:
    out: list[str] = []
    for char in _lower_tr(word):
        out.extend(parse_ipa(LETTER_IPA.get(char, "")))
    # LETTER_IPA girdilerinde her harfin kendi stres'i var; bunları çıkar
    # ki stress.apply son hecede tek vurgu yerleştirsin (Türkçe varsayılanı).
    return [phone for phone in out if phone != STRESS]


def _inventory_warnings(phonemes: list[str]) -> list[str]:
    allowed = ALL_PHONEMES | {CTC_SPC}
    return [f"unknown token: {p}" for p in phonemes if p not in allowed]


def _stress_syllable_index(phones: list[str], syllables: list[list[str]]) -> int | None:
    plain_index = 0
    starts: list[tuple[int, int]] = []
    cursor = 0
    for idx, syllable in enumerate(syllables):
        starts.append((idx, cursor))
        cursor += len(syllable)
    for phone in phones:
        if phone == STRESS:
            for idx, start in reversed(starts):
                if plain_index >= start:
                    return idx
            return 0
        plain_index += 1
    return None
