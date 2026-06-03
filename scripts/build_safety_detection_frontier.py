"""Güvenlik–tespit dengesi (frontier) tablosunu commit'li per-fonem doğrulama
çıktısından üretir.

Değerlendirme motoru her hedef fonem için bir durum (`correct`/`incorrect`/
`missing`/`extra`) ve sürekli fonetik mesafe skoru verir. Bu script, varsayılan
(agresif) çalışma noktasının yanı sıra, `incorrect`/`missing` kararlarını
kontrast sınıfına ve fonetik mesafeye göre yumuşatan birkaç temkinli politikayı
aynı dondurulmuş kayıt kümesi üzerinde yeniden hesaplar. Amaç, düşük yanlış-pozitif
oranının ancak tespitten ödün vererek elde edilebildiği temel dengeyi
göstermektir (modeli yeniden koşmaya gerek yoktur; karar yeniden yorumlamadır).

Girdi : reports/assessment/validation-recordings-mms1b-phones.csv
Çıktı : reports/assessment/safety-detection-frontier.{csv,md}
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from g2p.constants import ALL_VOWELS, LONG_TO_SHORT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "reports/assessment/validation-recordings-mms1b-phones.csv"
OUT_CSV = ROOT / "reports/assessment/safety-detection-frontier.csv"
OUT_MD = ROOT / "reports/assessment/safety-detection-frontier.md"

FLAGGED = {"incorrect", "missing"}

# Allofonik taban eşlemesi (aynı fonemin yüzey varyantları hata sayılmaz).
_ALLOPHONE_TO_BASE = {
    "æ": "e", "c": "k", "ɟ": "ɡ", "ɲ": "n", "ŋ": "n", "ɫ": "l",
    "β": "v", "β̞": "v", "ɾ̞̊": "ɾ",
    **{long: short for long, short in LONG_TO_SHORT.items()},
    **{f"{base}ʰ": base for base in ("p", "t", "k", "c")},
}


def _short(p: str) -> str:
    return LONG_TO_SHORT.get(p, p)


def phonemic_base(phone: str) -> str:
    seen: set[str] = set()
    cur = phone
    while cur in _ALLOPHONE_TO_BASE and cur not in seen:
        seen.add(cur)
        cur = _ALLOPHONE_TO_BASE[cur]
    return cur


def are_allophonic(a: str, b: str) -> bool:
    return bool(a) and bool(b) and a != b and phonemic_base(a) == phonemic_base(b)


def is_vowel(p: str) -> bool:
    return bool(p) and _short(p) in ALL_VOWELS


def load_rows() -> list[dict]:
    rows = []
    for r in csv.DictReader(IN_CSV.open(encoding="utf-8")):
        try:
            score = float(r["score"]) if r.get("score") not in (None, "") else None
        except ValueError:
            score = None
        rows.append(
            {
                "kind": r["kind"],
                "expected": r.get("expected") or "",
                "observed": r.get("observed") or "",
                "status": r.get("status") or "",
                "score": score,
                "intended": int(r.get("is_intended_error_pos") or 0),
            }
        )
    return rows


def vowel_sub(r: dict) -> bool:
    """Non-allofonik ünlü→ünlü ikamesi (tanıyıcının güvenilir olduğu kontrast)."""
    return is_vowel(r["expected"]) and is_vowel(r["observed"]) and not are_allophonic(
        r["expected"], r["observed"]
    )


def metrics(rows: list[dict], keep_incorrect, keep_missing) -> dict:
    flagged = {"CTL": 0, "W_NAT": 0}
    target = {"CTL": 0, "W_NAT": 0}
    wi = wc = wf = 0
    for r in rows:
        st, k = r["status"], r["kind"]
        if st == "incorrect":
            is_flag = keep_incorrect(r)
        elif st == "missing":
            is_flag = keep_missing(r)
        else:
            is_flag = False
        if k in target and st != "extra":
            target[k] += 1
            if is_flag:
                flagged[k] += 1
        if k == "W_ERR" and r["intended"]:
            wi += 1
            if is_flag:
                wc += 1
        if k == "W_ERR" and is_flag:
            wf += 1
    return {
        "ctl_flagged": flagged["CTL"], "ctl_total": target["CTL"],
        "ctl_fp": round(flagged["CTL"] / target["CTL"], 4) if target["CTL"] else None,
        "wnat_flagged": flagged["W_NAT"], "wnat_total": target["W_NAT"],
        "wnat_fp": round(flagged["W_NAT"] / target["W_NAT"], 4) if target["W_NAT"] else None,
        "intended": wi, "caught": wc,
        "detection": round(wc / wi, 4) if wi else None,
        "werr_flagged": wf,
        "precision": round(wc / wf, 4) if wf else None,
    }


KEEP_ALL_M = lambda r: True  # noqa: E731
SCORE_LO = 0.30
BANDS = [
    ("baseline", "Band yok — varsayılan agresif çalışma noktası",
     lambda r: True, KEEP_ALL_M),
    ("allofonik", "Allofonik ikameler (aynı fonem) yumuşatılır",
     lambda r: not are_allophonic(r["expected"], r["observed"]), KEEP_ALL_M),
    ("vowel_trust", "Yalnız non-allofonik ünlü kontrastları işaretlenir",
     lambda r: vowel_sub(r), KEEP_ALL_M),
    ("vowel_trust_miss", "Ünlü kontrastı + 'missing' yumuşatılır",
     lambda r: vowel_sub(r), lambda r: False),
    ("vowel_or_far_cons", f"Ünlü kontrastı VEYA büyük-mesafe ünsüz (score<{SCORE_LO})",
     lambda r: vowel_sub(r) or (not are_allophonic(r["expected"], r["observed"])
                                and r["score"] is not None and r["score"] < SCORE_LO),
     KEEP_ALL_M),
    ("vowel_or_far_cons_miss", "Önceki + 'missing' yumuşatılır",
     lambda r: vowel_sub(r) or (not are_allophonic(r["expected"], r["observed"])
                                and r["score"] is not None and r["score"] < SCORE_LO),
     lambda r: False),
]


def main() -> int:
    rows = load_rows()
    out = []
    for key, label, ki, km in BANDS:
        m = metrics(rows, ki, km)
        m["band"] = key
        m["label"] = label
        out.append(m)

    cols = ["band", "label", "ctl_flagged", "ctl_total", "ctl_fp",
            "wnat_flagged", "wnat_total", "wnat_fp",
            "caught", "intended", "detection", "werr_flagged", "precision"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for m in out:
            w.writerow({c: m[c] for c in cols})

    def pct(x):
        return f"%{x * 100:.2f}".replace(".", ",") if x is not None else "—"

    lines = [
        "# Güvenlik–Tespit Dengesi (frontier)",
        "",
        "İki anadili konuşmacılı (K1, K2) 150 kayıtlık doğrulama setinde, varsayılan "
        "(agresif) çalışma noktası ile `incorrect`/`missing` kararlarını kontrast "
        "sınıfına ve fonetik mesafeye göre yumuşatan temkinli politikalar. Tüm satırlar "
        "aynı dondurulmuş kayıt kümesinden, deterministik biçimde türetilmiştir.",
        "",
        "| Politika | CTL FP | W_NAT FP | Tespit | Precision@err |",
        "|---|---|---|---|---|",
    ]
    for m in out:
        det = f"{pct(m['detection'])} ({m['caught']}/{m['intended']})"
        prec = f"{pct(m['precision'])} ({m['caught']}/{m['werr_flagged']})"
        ctl = f"{pct(m['ctl_fp'])} ({m['ctl_flagged']}/{m['ctl_total']})"
        nat = f"{pct(m['wnat_fp'])} ({m['wnat_flagged']}/{m['wnat_total']})"
        lines.append(f"| {m['label']} | {ctl} | {nat} | {det} | {prec} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"\nYazıldı: {OUT_CSV} ve {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
