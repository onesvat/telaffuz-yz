"""Doğrulama-seti eval çıktısından commit edilebilir rapor + figür CSV'leri üret.

Girdi (gitignore'lu ara çıktılar):
  .smoke/eval_validation_summary.json
  .smoke/eval_validation_phones.csv

Çıktı (commit edilen final raporlar — tez ve figürlerin veri kaynağı):
  reports/assessment/validation-recordings-mms1b.csv        (kind bazında oranlar)
  reports/assessment/validation-recordings-mms1b-by-class.csv (fonem sınıfı bazında)
  reports/assessment/validation-recordings-mms1b-phones.csv (fonem satırları)
  reports/assessment/validation-recordings-mms1b.md          (özet anlatı)

Yeniden üretim:
  uv run python scripts/eval_validation_recordings.py --model mms-1b --api <url>
  uv run python scripts/build_validation_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / ".smoke"
OUT = ROOT / "reports/assessment"
DEFAULT_SUMMARY = SMOKE / "eval_validation_summary.json"
DEFAULT_PHONES = SMOKE / "eval_validation_phones.csv"
DEFAULT_STEM = "validation-recordings-mms1b"
MODEL_LABELS = {
    "mms-1b": "MMS-1B",
    "xls-r-300m": "XLS-R-300M",
}


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--phones", type=Path, default=DEFAULT_PHONES)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--model-label", default="")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.phones.open(encoding="utf-8")))
    bk = summary["by_kind"]
    model = summary.get("model") or ""
    model_label = args.model_label or MODEL_LABELS.get(model, str(model) or "model")
    out_kind_csv = args.out_dir / f"{args.stem}.csv"
    out_class_csv = args.out_dir / f"{args.stem}-by-class.csv"
    out_phones_csv = args.out_dir / f"{args.stem}-phones.csv"
    out_md = args.out_dir / f"{args.stem}.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) kind bazında oranlar
    with out_kind_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "kind", "target_phones", "correct", "incorrect", "missing", "extra",
            "flagged", "hard_fp_rate", "incorrect_rate", "mean_phone_score",
            "intended_positions", "caught", "detection_rate", "precision_at_err",
        ])
        for kind in ("CTL", "W_NAT", "W_ERR"):
            d = bk[kind]
            w.writerow([
                kind, d.get("total"), d.get("correct"), d.get("incorrect"),
                d.get("missing"), d.get("extra"), d.get("flagged"),
                d.get("hard_fp_rate"), d.get("incorrect_rate"), d.get("mean_phone_score"),
                d.get("intended_positions"), d.get("caught"),
                d.get("error_detection_rate"), d.get("precision_at_err"),
            ])

    # 2) fonem sınıfı bazında: doğal-doğru sert-FP + W_ERR tespit
    nat_total: dict[str, int] = defaultdict(int)
    nat_flag: dict[str, int] = defaultdict(int)
    err_int: dict[str, int] = defaultdict(int)
    err_caught: dict[str, int] = defaultdict(int)
    for r in rows:
        cls = r["expected_class"] or ("(missing)" if r["status"] == "missing" else "(other)")
        if r["kind"] in ("CTL", "W_NAT"):
            nat_total[cls] += 1
            if r["is_flagged"] == "1":
                nat_flag[cls] += 1
        elif r["kind"] == "W_ERR" and r["is_intended_error_pos"] == "1":
            err_int[cls] += 1
            if r["is_flagged"] == "1":
                err_caught[cls] += 1
    classes = sorted(set(nat_total) | set(err_int))
    with out_class_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "phone_class", "native_total", "native_flagged", "hard_fp_rate",
            "err_intended", "err_caught", "detection_rate",
        ])
        for cls in classes:
            nt, nf = nat_total[cls], nat_flag[cls]
            ei, ec = err_int[cls], err_caught[cls]
            w.writerow([
                cls, nt, nf, round(nf / nt, 4) if nt else "",
                ei, ec, round(ec / ei, 4) if ei else "",
            ])

    # 3) markdown özet
    ctl, wnat, werr = bk["CTL"], bk["W_NAT"], bk["W_ERR"]
    md = f"""# Doğrulama-seti değerlendirmesi — {model_label} (phonetic-distance-v1)

Konuşmacılar: {", ".join(summary["speakers"])}. Skorlayıcı: `{summary["score_version"]}`.
"İşaretli" (flagged) = hedef fonem durumu `correct` değil ({", ".join(summary["flagged_statuses"])}).

| Küme | Hedef fonem | Sert yanlış-pozitif | Yalnız incorrect | Ort. fonem skoru | Tespit | Kesinlik@hata |
|---|---|---|---|---|---|---|
| CTL (kontrol) | {ctl["total"]} | {pct(ctl["hard_fp_rate"])} | {pct(ctl["incorrect_rate"])} | {ctl["mean_phone_score"]} | — | — |
| W_NAT (doğal doğru) | {wnat["total"]} | {pct(wnat["hard_fp_rate"])} | {pct(wnat["incorrect_rate"])} | {wnat["mean_phone_score"]} | — | — |
| W_ERR (kasıtlı hata) | {werr["total"]} | — | — | {werr["mean_phone_score"]} | {pct(werr["error_detection_rate"])} ({werr["caught"]}/{werr["intended_positions"]}) | {pct(werr["precision_at_err"])} |

Doğal-doğru konuşmada sert yanlış-pozitif oranı düşüktür (CTL {pct(ctl["hard_fp_rate"])},
W_NAT {pct(wnat["hard_fp_rate"])}); kasıtlı-hata konumlarının {pct(werr["error_detection_rate"])}'i
yakalanır. Kesinlik@hata yalnız *tasarlanmış* hata konumunu doğru kabul eder; öğrencinin
aynı sözcükte yaptığı diğer gerçek sapmalar paydada görünür, dolayısıyla bu değer alt sınırdır.

Sınıf kırılımı `{out_class_csv.name}`'de; ham fonem tablosu `{out_phones_csv.name}`
çıktısındadır.
"""
    out_md.write_text(md, encoding="utf-8")

    with out_phones_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out_kind_csv}\nwrote {out_class_csv}\nwrote {out_phones_csv}\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
