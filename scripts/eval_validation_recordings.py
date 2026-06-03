"""Değerlendirme motorunu el yapımı öğrenci-hatası doğrulama setine karşı koştur.

Her `data/validation/recordings/<konuşmacı>/<prompt>-a01.wav` kaydını çalışan
`/api/v1/assess` uç noktasına gönderir ve fonem başına sürekli skor + durum
(`correct`/`incorrect`/`missing`/`extra`) tablolarını yazar.

Güvenlik–tespit dengesi şu tanımlarla raporlanır (hepsi sürekli puanlama
sözleşmesi `phonetic-distance-v1` üzerinden):

* **işaretli (flagged)** — hedef fonemin durumu `correct` değil (yani
  `incorrect` ya da `missing`); sistem öğrencinin o fonemdeki üretimini
  kabul etmemiştir.
* **sert yanlış-pozitif (hard FP)** — doğal-doğru kaydında (CTL/W_NAT)
  işaretli hedef fonem oranı.
* **tespit (detection)** — W_ERR'de kasıtlı-hata konumlarında işaretli
  fonem oranı.
* **kesinlik (precision)** — tüm işaretli konumların içinde gerçek
  kasıtlı-hata konumuna düşenlerin oranı.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data/validation/prompts.tsv"
RECORDINGS = ROOT / "data/validation/recordings"
DEFAULT_OUT_DIR = ROOT / ".smoke"
DEFAULT_OUT_STEM = "eval_validation"

# Stüdyo değerlendirmesi iki anadili konuşmacı üzerinde yapılır (sert yanlış-pozitif
# yalnız anadili-doğru konuşmada anlamlıdır). Diğer konuşmacılar --speakers ile eklenebilir.
DEFAULT_SPEAKERS = ("onur", "gamze")
DEFAULT_MODEL = "mms-1b"
DEFAULT_API = "http://localhost:8000/api/v1/assess"

# Hedef fonemin "işaretli" (sisteme göre kabul edilmemiş) sayıldığı durumlar.
FLAGGED_STATUSES = {"incorrect", "missing"}


def kind_of(prompt_id: str) -> str:
    if prompt_id.startswith("CTL_"):
        return "CTL"
    if prompt_id.startswith("W_NAT"):
        return "W_NAT"
    if prompt_id.startswith("W_ERR"):
        return "W_ERR"
    return "?"


def parse_ipa(s: str | None) -> list[str]:
    if not s:
        return []
    return [p for p in s.strip().split() if p]


def strip_stress(phones: list[str]) -> list[str]:
    return [p for p in phones if p != "ˈ"]


def intended_error_positions(canonical: list[str], wrong: list[str]) -> set[int]:
    """Vurgu-arındırılmış kanonik dizide kasıtlı-yanlışın ayrıştığı konumlar."""
    c = strip_stress(canonical)
    w = strip_stress(wrong)
    if len(c) != len(w):
        return set()
    return {i for i, (a, b) in enumerate(zip(c, w)) if a != b}


def load_prompts() -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(PROMPTS.open(encoding="utf-8"), delimiter="\t"))
    return {r["prompt_id"]: r for r in rows}


def post_assessment(api_url: str, wav: Path, phones: list[str], model: str) -> dict:
    with wav.open("rb") as fh:
        files = {"audio": (wav.name, fh, "audio/wav")}
        data = {
            "model_id": model,
            "expected_phonemes": " ".join(phones),
        }
        r = requests.post(api_url, data=data, files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def _result_phones(response: dict) -> list[dict]:
    result = response.get("result")
    if not isinstance(result, dict):
        return []
    return result.get("phones") or []


def _analysis_phones(response: dict) -> list[dict]:
    analysis = response.get("analysis")
    if not isinstance(analysis, dict):
        return []
    return analysis.get("phones") or []


def _expected_class(phone: dict) -> str:
    feats = phone.get("distance_features") or {}
    cls = feats.get("class") or {}
    return str(cls.get("expected") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakers", nargs="+", default=list(DEFAULT_SPEAKERS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--limit", type=int, default=0,
                        help="Konuşmacı başına en çok N prompt işle (0=hepsi).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-stem", default=DEFAULT_OUT_STEM,
                        help="Ara çıktı dosya kökü (default: eval_validation).")
    args = parser.parse_args()

    prompts = load_prompts()

    per_phone: list[dict] = []
    per_prompt: list[dict] = []
    full_obs: list[dict] = []
    agg: dict[str, int] = defaultdict(int)
    speaker_agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    score_sums: dict[str, float] = defaultdict(float)
    error_detection: dict[str, int] = {
        "intended_positions": 0,
        "caught": 0,
        "flagged_total": 0,
    }

    out_dir = args.out_dir
    out_phones_csv = out_dir / f"{args.out_stem}_phones.csv"
    out_prompts_csv = out_dir / f"{args.out_stem}_prompts.csv"
    out_summary = out_dir / f"{args.out_stem}_summary.json"
    out_obs = out_dir / f"{args.out_stem}_obs.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)

    for speaker in args.speakers:
        speaker_dir = RECORDINGS / speaker
        if not speaker_dir.exists():
            print(f"skip speaker {speaker} (no dir)", file=sys.stderr)
            continue
        count = 0
        for prompt_id, prompt in prompts.items():
            if args.limit and count >= args.limit:
                break
            wav = speaker_dir / f"{prompt_id}-a01.wav"
            if not wav.exists():
                continue
            kind = kind_of(prompt_id)
            canonical_full = parse_ipa(prompt.get("canonical_ipa"))
            canonical = strip_stress(canonical_full)
            wrong_full = parse_ipa(prompt.get("intended_wrong_ipa"))
            err_positions = (
                intended_error_positions(canonical_full, wrong_full)
                if kind == "W_ERR"
                else set()
            )

            print(f"[{speaker}] {prompt_id} {kind} ({len(canonical)} ph)", file=sys.stderr)

            response = post_assessment(args.api, wav, canonical_full, args.model)
            if response.get("status") != "ok":
                print(f"  -> status={response.get('status')} (atlandı)", file=sys.stderr)
                continue
            phones = _result_phones(response)
            analysis_phones = _analysis_phones(response)

            n_correct = n_incorrect = n_missing = n_extra = 0
            n_flagged = 0
            caught_at_err = 0
            prompt_score_sum = 0.0
            prompt_score_n = 0
            for ph in phones:
                status = str(ph.get("status") or "")
                idx_exp = ph.get("index_expected")
                is_err_pos = idx_exp in err_positions if idx_exp is not None else False
                is_flagged = status in FLAGGED_STATUSES
                score = ph.get("score")
                length = ph.get("length") or {}
                quality = ph.get("quality")
                idx_obs = ph.get("index_observed")
                analysis_phone = (
                    analysis_phones[idx_obs]
                    if isinstance(idx_obs, int) and 0 <= idx_obs < len(analysis_phones)
                    else {}
                )

                if status == "correct":
                    n_correct += 1
                elif status == "incorrect":
                    n_incorrect += 1
                elif status == "missing":
                    n_missing += 1
                elif status == "extra":
                    n_extra += 1

                # Hedef-konumlu satırlar (extra dışında) puan ortalamasına girer.
                if status != "extra" and isinstance(score, (int, float)):
                    prompt_score_sum += float(score)
                    prompt_score_n += 1

                if is_flagged:
                    n_flagged += 1
                    if is_err_pos:
                        caught_at_err += 1

                per_phone.append({
                    "speaker": speaker, "prompt_id": prompt_id, "kind": kind,
                    "index_expected": idx_exp,
                    "expected": ph.get("expected") or "",
                    "observed": ph.get("observed") or "",
                    "expected_class": _expected_class(ph),
                    "status": status,
                    "score": score,
                    "score_reason": ph.get("score_reason") or "",
                    "quality_score": (
                        quality.get("quality_score")
                        if isinstance(quality, dict)
                        else quality
                    ),
                    "analysis_quality_score": (
                        analysis_phone.get("quality")
                        if isinstance(analysis_phone, dict)
                        else None
                    ),
                    "length_score": length.get("score") if isinstance(length, dict) else None,
                    "length_note": length.get("note") if isinstance(length, dict) else None,
                    "changed_by_analysis": bool(ph.get("changed_by_analysis")),
                    "analysis_source": (
                        analysis_phone.get("analysis_source")
                        if isinstance(analysis_phone, dict)
                        else None
                    ),
                    "raw_observed": ph.get("raw_observed") or "",
                    "analysis_observed": ph.get("analysis_observed") or "",
                    "wav2vec_confidence": (
                        analysis_phone.get("confidence")
                        if isinstance(analysis_phone, dict)
                        else None
                    ),
                    "is_intended_error_pos": int(bool(is_err_pos)),
                    "is_flagged": int(is_flagged),
                })
                full_obs.append({
                    "speaker": speaker, "prompt_id": prompt_id, "kind": kind,
                    "index_expected": idx_exp,
                    "is_intended_error_pos": int(bool(is_err_pos)),
                    "phone": ph,
                })

            # Kind bazında toplamlar (hedef fonem = correct+incorrect+missing).
            target_n = n_correct + n_incorrect + n_missing
            for key, val in (
                ("total", target_n), ("correct", n_correct), ("incorrect", n_incorrect),
                ("missing", n_missing), ("extra", n_extra), ("flagged", n_flagged),
            ):
                agg[f"{kind}_{key}"] += val
                speaker_agg[speaker][f"{kind}_{key}"] += val
            score_sums[f"{kind}_score_sum"] += prompt_score_sum
            score_sums[f"{kind}_score_n"] += prompt_score_n

            if kind == "W_ERR":
                error_detection["intended_positions"] += len(err_positions)
                error_detection["caught"] += caught_at_err
                error_detection["flagged_total"] += n_flagged

            per_prompt.append({
                "speaker": speaker, "prompt_id": prompt_id, "kind": kind,
                "target_phones": target_n,
                "correct": n_correct, "incorrect": n_incorrect,
                "missing": n_missing, "extra": n_extra,
                "flagged": n_flagged,
                "caught_at_err": caught_at_err,
                "intended_err_positions": len(err_positions),
                "word_score": (response.get("result") or {}).get("score", {}).get("item_score"),
            })
            count += 1

    # CSV'ler
    if per_phone:
        with out_phones_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_phone[0].keys()))
            w.writeheader()
            w.writerows(per_phone)
    if per_prompt:
        with out_prompts_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_prompt[0].keys()))
            w.writeheader()
            w.writerows(per_prompt)
    if full_obs:
        with out_obs.open("w", encoding="utf-8") as fh:
            for row in full_obs:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    def mean_score(source: dict[str, float], kind: str) -> float | None:
        n = source.get(f"{kind}_score_n", 0)
        if not n:
            return None
        return round(source.get(f"{kind}_score_sum", 0.0) / n, 4)

    def kind_summary(counts: dict[str, int], scores: dict[str, float], kind: str) -> dict:
        total = counts.get(f"{kind}_total", 0)
        flagged = counts.get(f"{kind}_flagged", 0)
        incorrect = counts.get(f"{kind}_incorrect", 0)
        data: dict[str, object] = {
            "total": total,
            "correct": counts.get(f"{kind}_correct", 0),
            "incorrect": incorrect,
            "missing": counts.get(f"{kind}_missing", 0),
            "extra": counts.get(f"{kind}_extra", 0),
            "flagged": flagged,
            "mean_phone_score": mean_score(scores, kind),
        }
        if kind in {"CTL", "W_NAT"}:
            data["hard_fp_rate"] = rate(flagged, total)
            data["incorrect_rate"] = rate(incorrect, total)
        return data

    by_kind = {
        kind: kind_summary(agg, score_sums, kind)
        for kind in ("CTL", "W_NAT", "W_ERR")
    }
    by_kind["W_ERR"]["intended_positions"] = error_detection["intended_positions"]
    by_kind["W_ERR"]["caught"] = error_detection["caught"]
    by_kind["W_ERR"]["error_detection_rate"] = rate(
        error_detection["caught"], error_detection["intended_positions"]
    )
    by_kind["W_ERR"]["precision_at_err"] = rate(
        error_detection["caught"], error_detection["flagged_total"]
    )

    by_speaker = {
        speaker: {
            kind: kind_summary(counts, score_sums, kind)
            for kind in ("CTL", "W_NAT", "W_ERR")
        }
        for speaker, counts in sorted(speaker_agg.items())
    }

    summary = {
        "speakers": args.speakers,
        "model": args.model,
        "score_version": "phonetic-distance-v1",
        "flagged_statuses": sorted(FLAGGED_STATUSES),
        "by_kind": by_kind,
        "by_speaker": by_speaker,
    }

    out_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
