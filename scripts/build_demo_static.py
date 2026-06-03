from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo_static"
INDEX = DEMO_DIR / "index.html"
API_BASE = "https://telaffuz-yz.onurnesvat.com/api/v1"
MODELS = ("mms-1b", "xls-r-300m")
HTTP_USER_AGENT = "curl/8.0 telaffuz-demo-static"

# Tez demosu kayıt seti — her hata tarzı decision_authority.json'daki bir
# kontrasta denk gelir, böylece sistem güvenle yakalar (çiçek hariç: t͡ʃ→ʃ
# tabloda yok, bilinçli "belirsiz"/güvenlik örneği). Dosya adı: idx//2+1.
FALLBACK_SLOTS = [
    {"idx": 0, "kind": "correct", "word": "dede", "ipa": "/de.ˈde/"},
    {"idx": 1, "kind": "wrong", "word": "dede", "ipa": "/de.ˈde/"},
    {"idx": 2, "kind": "correct", "word": "baba", "ipa": "/ba.ˈba/"},
    {"idx": 3, "kind": "wrong", "word": "baba", "ipa": "/ba.ˈba/"},
    {"idx": 4, "kind": "correct", "word": "kapı", "ipa": "/kʰa.ˈpʰɯ/"},
    {"idx": 5, "kind": "wrong", "word": "kapı", "ipa": "/kʰa.ˈpʰɯ/"},
    {"idx": 6, "kind": "correct", "word": "cam", "ipa": "/d͡ʒam/"},
    {"idx": 7, "kind": "wrong", "word": "cam", "ipa": "/d͡ʒam/"},
    {"idx": 8, "kind": "correct", "word": "kas", "ipa": "/kʰas/"},
    {"idx": 9, "kind": "wrong", "word": "kas", "ipa": "/kʰas/"},
    {"idx": 10, "kind": "correct", "word": "göz", "ipa": "/ɟœz/"},
    {"idx": 11, "kind": "wrong", "word": "göz", "ipa": "/ɟœz/"},
    {"idx": 12, "kind": "correct", "word": "kâr", "ipa": "/cʰaːɾ̞̊/"},
    {"idx": 13, "kind": "wrong", "word": "kâr", "ipa": "/cʰaːɾ̞̊/"},
    {"idx": 14, "kind": "correct", "word": "çiçek", "ipa": "/t͡ʃi.ˈt͡ʃec/"},
    {"idx": 15, "kind": "wrong", "word": "çiçek", "ipa": "/t͡ʃi.ˈt͡ʃec/"},
    {"idx": 16, "kind": "correct", "word": "tencere", "ipa": "/tʰeɲ.ˈd͡ʒe.ɾe/"},
    {"idx": 17, "kind": "wrong", "word": "tencere", "ipa": "/tʰeɲ.ˈd͡ʒe.ɾe/"},
    {"idx": 18, "kind": "correct", "word": "ışık", "ipa": "/ɯ.ˈʃɯk/"},
    {"idx": 19, "kind": "wrong", "word": "ışık", "ipa": "/ɯ.ˈʃɯk/"},
]


def load_slots() -> list[dict[str, object]]:
    if INDEX.exists():
        text = INDEX.read_text(encoding="utf-8")
        match = re.search(r"const (?:SLOTS|SLIDES) = (\[.*?\]);\n(?:const PHONEME_LABELS|let cur)", text, re.S)
        if match:
            return [
                {
                    "idx": int(item["idx"]),
                    "kind": str(item["kind"]),
                    "word": str(item["word"]),
                    "ipa": str(item["ipa"]),
                }
                for item in json.loads(match.group(1))
            ]
    return FALLBACK_SLOTS


def load_annotations() -> dict[str, str]:
    ann_path = DEMO_DIR / "annotations.json"
    if ann_path.exists():
        return json.loads(ann_path.read_text(encoding="utf-8"))
    return {}


def load_existing_slides() -> list[dict[str, object]]:
    if not INDEX.exists():
        raise FileNotFoundError(INDEX)
    text = INDEX.read_text(encoding="utf-8")
    match = re.search(r"const SLIDES = (\[.*?\]);\nconst PHONEME_LABELS", text, re.S)
    if not match:
        raise RuntimeError(f"could not find SLIDES payload in {INDEX}")
    slides = json.loads(match.group(1))
    if not isinstance(slides, list):
        raise RuntimeError(f"SLIDES payload is not a list in {INDEX}")
    return [dict(item) for item in slides if isinstance(item, dict)]


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return int(round(frames * 1000 / rate)) if rate else 1000


def get_json(url: str) -> object:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": HTTP_USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def post_multipart(url: str, fields: dict[str, str], file_field: str, file_path: Path) -> object:
    boundary = f"----telaffuz-demo-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    filename = file_path.name
    mime = mimetypes.guess_type(filename)[0] or "audio/wav"
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    chunks.append(f"Content-Type: {mime}\r\n\r\n".encode())
    chunks.append(file_path.read_bytes())
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "User-Agent": HTTP_USER_AGENT,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(req, timeout=300) as res:
        return json.loads(res.read().decode("utf-8"))


def fetch_slot(slot: dict[str, object]) -> dict[str, object]:
    idx = int(slot["idx"])
    kind = str(slot["kind"])
    word = str(slot["word"])
    file_no = idx // 2 + 1
    audio_rel = f"audio/{file_no:02d}-{kind}.wav"
    audio_path = DEMO_DIR / audio_rel
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    qs = urllib.parse.urlencode({"word": word, "pedagogical": "true", "use_reference": "true"})
    g2p = get_json(f"{API_BASE}/g2p/inspect?{qs}")
    expected = g2p.get("phonemes") if isinstance(g2p, dict) else []
    if not isinstance(expected, list) or not expected:
        raise RuntimeError(f"empty G2P phonemes for {word}")

    duration = wav_duration_ms(audio_path)
    assessments: dict[str, object] = {}
    for model in MODELS:
        fields = {
            "model_id": model,
            "word": word,
            "expected_phonemes": " ".join(str(p) for p in expected),
            "duration_ms": str(duration),
            "source": "static_demo",
            "test_set": "thesis_static_demo",
            "test_index": str(idx),
            "target_kind": "canonical" if kind == "correct" else "intended_wrong",
            "consent_audio": "false",
        }
        assessments[model] = post_multipart(f"{API_BASE}/assess", fields, "audio", audio_path)
    return {
        **slot,
        "audio": audio_rel,
        "duration_ms": duration,
        "g2p": g2p,
        "assessments": assessments,
    }


def build_runner() -> object:
    """Build one CoachRunner to reuse across slots so each wav2vec model is
    loaded once, not reloaded from disk per slot."""
    sys.path.insert(0, str(ROOT / "src"))
    from api.services.coach_runner import CoachRunner

    return CoachRunner()


def local_slot(slot: dict[str, object], runner: object) -> dict[str, object]:
    sys.path.insert(0, str(ROOT / "src"))
    from api.services.audio import inspect_wav
    from g2p.pipeline import transcribe_word

    idx = int(slot["idx"])
    kind = str(slot["kind"])
    word = str(slot["word"])
    file_no = idx // 2 + 1
    audio_rel = f"audio/{file_no:02d}-{kind}.wav"
    audio_path = DEMO_DIR / audio_rel
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    g2p_result = transcribe_word(word, use_reference=True, pedagogical_allophones=True, trace=True)
    g2p = {
        "word": g2p_result.word,
        "normalized": g2p_result.normalized,
        "ipa": g2p_result.ipa,
        "phonemes": list(g2p_result.phonemes),
        "syllables": [list(s) for s in g2p_result.syllables],
        "source": g2p_result.source,
        "exception_type": g2p_result.exception_type,
        "warnings": list(g2p_result.warnings),
        "trace": [
            {
                "name": step.name,
                "input": list(step.input),
                "output": list(step.output),
                "fired": step.fired,
                "note": step.note,
            }
            for step in g2p_result.trace
        ],
    }

    duration = wav_duration_ms(audio_path)
    quality = inspect_wav(audio_path, fallback_duration_ms=duration).as_dict()
    expected = [str(p) for p in g2p["phonemes"]]
    assessments: dict[str, object] = {}
    for model in MODELS:
        outcome = runner.run(model_id=model, wav_path=audio_path, expected_phones=expected)
        assessments[model] = {
            "recording_id": f"local-{idx:02d}-{kind}-{model}",
            "recording_quality": quality,
            "expected_phonemes": expected,
            "model_id": model,
            **outcome,
        }
    return {
        **slot,
        "audio": audio_rel,
        "duration_ms": duration,
        "g2p": g2p,
        "assessments": assessments,
    }


def js_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


PHONEME_LABELS_TR: dict[str, str] = {
    "a": "a", "e": "kapalı e", "i": "i", "ɯ": "ı (kalın)", "o": "o", "œ": "ö",
    "u": "u", "y": "ü", "aː": "uzun a", "eː": "uzun e", "iː": "uzun i",
    "ɯː": "uzun ı", "oː": "uzun o", "œː": "uzun ö", "uː": "uzun u", "yː": "uzun ü",
    "æ": "açık e (â/e)", "p": "p", "b": "b", "t": "t", "d": "d", "k": "kalın k",
    "ɡ": "g", "t͡ʃ": "ç", "d͡ʒ": "c", "f": "f", "v": "v", "s": "s", "z": "z",
    "ʃ": "ş", "ʒ": "j", "h": "h", "m": "m", "n": "n", "l": "ince l",
    "ɾ": "tek vuruşlu r", "j": "y", "c": "ince k", "ɟ": "ince g",
    "ɲ": "ny (genizsi n)", "ŋ": "genizden n (ng)", "ɫ": "kalın l",
    "β": "yumuşak v", "β̞": "yumuşak v", "pʰ": "soluklu p", "tʰ": "soluklu t",
    "kʰ": "soluklu k", "cʰ": "soluklu ince k", "ɾ̞̊": "sessiz r (kelime sonu)",
    "ˈ": "vurgu", ".": "hece sınırı",
}


_TEMPLATE = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Türkçe Telaffuz Koçu — Tez Demosu</title>
<style>
:root{
  --canvas:#eef1f6; --surface:#ffffff; --surface-2:#f6f8fb; --line:#e1e6ef;
  --ink:#171d2b; --muted:#697086; --faint:#9aa1b3;
  --brand:#2d4b8e; --brand-soft:#e9eff9;
  --ok:#16794a; --ok-soft:#e6f3ec;
  --bad:#bf3349; --bad-soft:#fae9ec;
  --warn:#a76410; --warn-soft:#fbf0db;
  --extra:#5750ad; --extra-soft:#ecebf8;
  --shadow:0 1px 2px rgba(23,29,43,.05), 0 22px 50px rgba(23,29,43,.08);
  --serif:Georgia,"Times New Roman",serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
*{box-sizing:border-box;}
html,body{margin:0;min-height:100%;}
body{
  background:linear-gradient(180deg,rgba(233,239,249,.9),rgba(238,241,246,0) 360px),var(--canvas);
  color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;
}
button{font:inherit;}
.shell{max-width:1180px;margin:0 auto;padding:14px 22px 40px;}
.topbar{
  position:sticky;top:0;z-index:30;display:grid;grid-template-columns:auto 1fr auto;
  align-items:center;gap:14px;padding:12px 0 14px;
  background:linear-gradient(180deg,rgba(238,241,246,.98),rgba(238,241,246,.86));
  backdrop-filter:blur(12px);border-bottom:1px solid var(--line);
}
.navgroup{display:flex;gap:8px;}
.iconbtn{
  min-width:44px;height:40px;padding:0 12px;border:1px solid var(--line);border-radius:10px;
  background:var(--surface);color:var(--ink);cursor:pointer;font-size:15px;
  box-shadow:0 1px 1px rgba(23,29,43,.04);
}
.iconbtn:hover{border-color:rgba(45,75,142,.45);background:var(--brand-soft);}
.iconbtn:disabled{opacity:.35;cursor:not-allowed;background:var(--surface);}
.iconbtn.active{background:var(--brand);color:#fff;border-color:var(--brand);}
.meta{min-width:0;text-align:center;}
.counter{font-family:var(--mono);font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);}
.titleline{display:flex;align-items:baseline;justify-content:center;gap:10px;min-width:0;}
.word{font-family:var(--serif);font-size:26px;font-weight:700;line-height:1.05;}
.pill{display:inline-flex;align-items:center;border-radius:999px;padding:4px 10px;font-size:11px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;}
.pill.ok{color:var(--ok);background:var(--ok-soft);}
.pill.bad{color:var(--bad);background:var(--bad-soft);}
.pill.warn{color:var(--warn);background:var(--warn-soft);}
.pill.brand{color:var(--brand);background:var(--brand-soft);}

/* pipeline badge */
.pipe{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:14px;}
.pstage{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border:1px solid var(--line);border-radius:999px;background:var(--surface);font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.04em;}
.pstage.done{color:var(--brand);border-color:rgba(45,75,142,.25);background:var(--brand-soft);}
.pstage.active{color:#fff;background:var(--brand);border-color:var(--brand);font-weight:700;}
.parrow{color:var(--faint);font-size:12px;}

.slide{animation:fade .18s ease-out both;}
@keyframes fade{from{opacity:0;transform:translateY(5px);}to{opacity:1;transform:none;}}
.slidegrid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px;align-items:start;}
.col{display:flex;flex-direction:column;gap:16px;min-width:0;}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow);padding:18px;}
.eyebrow{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.13em;text-transform:uppercase;margin-bottom:10px;}

.hero{display:grid;grid-template-columns:118px 1fr;gap:20px;align-items:center;}
.ringwrap{position:relative;width:112px;height:112px;}
.ringtext{position:absolute;inset:0;display:grid;place-items:center;text-align:center;}
.score{font-family:var(--serif);font-size:31px;font-weight:700;line-height:1;}
.score small{display:block;margin-top:3px;font-family:var(--mono);font-size:9px;letter-spacing:.13em;color:var(--muted);}
.hero h1{margin:0 0 4px;font-family:var(--serif);font-size:46px;line-height:.98;}
.hero .ipa{font-family:var(--mono);font-size:16px;color:var(--muted);}
.pairline{margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;font-size:13px;color:var(--muted);}

/* annotation callout */
.anno-box{background:var(--brand-soft);border-left:3px solid var(--brand);border-radius:8px;padding:10px 14px;margin-top:12px;font-size:13px;line-height:1.5;color:var(--ink);}
.anno-label{font-family:var(--mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--brand);margin-bottom:4px;}
.pairbtn{border:1px solid var(--line);background:var(--surface-2);border-radius:999px;padding:3px 10px;font-size:12px;color:var(--brand);cursor:pointer;}
.pairbtn:hover{background:var(--brand-soft);}
.audiorow{margin-top:14px;display:flex;align-items:center;gap:12px;}
.audiobtn{display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:50%;border:1px solid var(--line);background:var(--surface-2);color:var(--brand);cursor:pointer;transition:.15s;box-shadow:0 1px 2px rgba(23,29,43,.05);}
.audiobtn:hover{background:var(--brand-soft);border-color:rgba(45,75,142,.45);}
.audiobtn.playing{background:var(--brand);color:#fff;border-color:var(--brand);}
.audiobtn svg{width:20px;height:20px;}
.audiocap{font-size:13px;color:var(--muted);}

/* aligned strip */
.hintline{font-size:12px;color:var(--faint);margin-top:8px;}
.diff-strip{display:flex;flex-wrap:wrap;gap:7px;}
.tok{display:inline-flex;min-width:50px;flex-direction:column;overflow:hidden;border:1px solid;border-radius:8px;font-family:var(--mono);line-height:1;background:var(--surface);cursor:pointer;padding:0;transition:transform .1s;}
.tok:hover{transform:translateY(-1px);}
.tok:disabled{cursor:default;opacity:.85;}
.tok.sel{outline:2px solid var(--brand);outline-offset:1px;}
.tok-line{display:flex;align-items:center;justify-content:center;gap:5px;padding:8px 9px;font-size:15px;}
.tok-line+.tok-line{border-top:1px solid rgba(0,0,0,.08);}
.tok-label{font-family:ui-sans-serif,system-ui,sans-serif;font-size:8px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;opacity:.55;}
.tok-equal{border-color:rgba(22,121,74,.3);color:var(--ok);}
.tok-equal .tok-line:first-child{background:var(--ok-soft);}
.tok-substitution{border-color:rgba(191,51,73,.32);color:var(--bad);}
.tok-substitution .tok-line:last-child{background:var(--bad-soft);}
.tok-deletion{border-color:rgba(167,100,16,.34);color:var(--warn);background:var(--warn-soft);}
.tok-insertion{border-color:rgba(87,80,173,.3);color:var(--extra);background:var(--extra-soft);}
.tok-masked{border-color:var(--line);color:var(--muted);background:var(--surface-2);}
.tok-scoreline{position:relative;justify-content:flex-end;padding:3px 7px;background:var(--surface-2);}
.tok-fill{position:absolute;left:0;top:0;bottom:0;border-radius:0;opacity:.5;}
.tok-fill-ok{background:var(--ok);} .tok-fill-warn{background:var(--warn);} .tok-fill-bad{background:var(--bad);}
.tok-pct{position:relative;font-size:9px;font-weight:800;color:var(--ink);opacity:.85;}
.tok-flag{margin-left:3px;font-size:11px;color:var(--brand);}

/* feedback */
.headline{font-family:var(--serif);font-size:19px;margin:0 0 12px;}
.fb-row{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:11px;padding:9px 0;border-top:1px solid var(--line);}
.fb-row:first-of-type{border-top:none;}
.fb-glyph{width:34px;height:34px;border-radius:8px;display:grid;place-items:center;font-family:var(--mono);font-size:14px;font-weight:700;border:1px solid;}
.fb-text{font-size:13.5px;line-height:1.45;}
.fb-badge{font-size:10px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;display:block;margin-bottom:2px;}
.fb-pct{font-family:var(--mono);font-size:12px;font-weight:700;padding:3px 8px;border-radius:7px;border:1px solid;}
.t-ok{color:var(--ok);} .b-ok{background:var(--ok-soft);border-color:rgba(22,121,74,.3);}
.t-bad{color:var(--bad);} .b-bad{background:var(--bad-soft);border-color:rgba(191,51,73,.3);}
.t-warn{color:var(--warn);} .b-warn{background:var(--warn-soft);border-color:rgba(167,100,16,.32);}
.t-extra{color:var(--extra);} .b-extra{background:var(--extra-soft);border-color:rgba(87,80,173,.3);}
.t-muted{color:var(--muted);} .b-muted{background:var(--surface-2);border-color:var(--line);}

/* two-score */
.ts-head{font-size:13px;color:var(--muted);margin-bottom:14px;}
.ts-head b{color:var(--ink);}
.ts-row{display:grid;grid-template-columns:54px 1fr 44px;align-items:center;gap:10px;margin-bottom:6px;}
.ts-name{font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.ts-bar{height:12px;border-radius:6px;background:var(--surface-2);border:1px solid var(--line);overflow:hidden;}
.ts-fill{height:100%;border-radius:6px;}
.ts-ok{background:var(--ok);} .ts-warn{background:var(--warn);} .ts-bad{background:var(--bad);} .ts-muted{background:var(--faint);}
.ts-val{font-family:var(--mono);font-size:12px;font-weight:700;text-align:right;}
.ts-note{grid-column:2 / span 2;font-size:12px;color:var(--muted);margin:-2px 0 8px;}
.ts-empty{color:var(--muted);font-size:13px;}

/* raw vs decision */
.rd-line{display:grid;grid-template-columns:96px 1fr;gap:10px;align-items:baseline;padding:7px 0;}
.rd-tag{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;}
.rd-val{font-family:var(--mono);font-size:18px;}
.rd-note{margin-top:10px;font-size:12.5px;color:var(--muted);line-height:1.45;border-top:1px solid var(--line);padding-top:10px;}

/* drill-down */
.drill{margin-top:16px;border:1px solid var(--line);border-radius:10px;background:var(--surface);box-shadow:var(--shadow);overflow:hidden;}
.drill>summary{cursor:pointer;list-style:none;padding:14px 18px;font-family:var(--mono);font-size:12px;letter-spacing:.05em;color:var(--brand);display:flex;align-items:center;gap:8px;}
.drill>summary::-webkit-details-marker{display:none;}
.drill>summary::before{content:"▸";transition:transform .15s;}
.drill[open]>summary::before{transform:rotate(90deg);}
.drill[open]>summary{border-bottom:1px solid var(--line);background:var(--surface-2);}
.drill-body{padding:16px 18px;}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-bottom:14px;}
.metric{border:1px solid var(--line);border-radius:8px;background:var(--canvas);padding:8px 10px;}
.metric span{display:block;font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;}
.metric b{display:block;margin-top:3px;font-family:var(--mono);font-size:13px;}
.tablewrap{overflow:auto;border:1px solid var(--line);border-radius:8px;}
table{width:100%;border-collapse:collapse;min-width:560px;}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left;font-size:12px;}
th{background:var(--surface-2);font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;}
td.mono{font-family:var(--mono);}
.muted{color:var(--muted);}
.verdict{font-weight:800;}
.verdict.ok{color:var(--ok);} .verdict.bad{color:var(--bad);} .verdict.warn{color:var(--warn);} .verdict.muted{color:var(--muted);}
.metaline{margin-top:12px;font-size:12px;color:var(--muted);line-height:1.5;}
.xls-summary{font-size:13px;margin-bottom:10px;}
.empty{color:var(--muted);font-family:var(--mono);}

/* overview */
.overview{animation:fade .18s ease-out both;}
.ov-head{margin:6px 0 18px;}
.ov-head h1{font-family:var(--serif);font-size:30px;margin:0 0 4px;}
.ov-head p{margin:0;color:var(--muted);font-size:14px;}
.ov-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px;}
.ov-tile{text-align:left;border:1px solid var(--line);border-radius:10px;background:var(--surface);box-shadow:var(--shadow);padding:14px;cursor:pointer;transition:transform .12s,border-color .12s;}
.ov-tile:hover{transform:translateY(-2px);border-color:rgba(45,75,142,.45);}
.ov-top{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px;}
.ov-word{font-family:var(--serif);font-size:22px;font-weight:700;}
.ov-bottom{display:flex;align-items:center;justify-content:space-between;gap:8px;}
.ov-score{font-family:var(--mono);font-size:22px;font-weight:700;}
.ov-result{font-size:12px;font-weight:700;}
.hint{margin-top:22px;text-align:center;font-size:12px;color:var(--faint);font-family:var(--mono);letter-spacing:.04em;}

@media (max-width:920px){
  .slidegrid{grid-template-columns:1fr;}
  .hero{grid-template-columns:88px 1fr;}
  .ringwrap{width:84px;height:84px;}
  .hero h1{font-size:34px;}
}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="navgroup">
      <button class="iconbtn" id="ovBtn" title="Genel bakış (G)">⊞ Genel</button>
      <button class="iconbtn" id="prev" aria-label="Önceki">←</button>
    </div>
    <div class="meta">
      <div class="counter"><span id="ix">1</span> / <span id="total">0</span></div>
      <div class="titleline"><span class="word" id="navWord">—</span><span id="kind" class="pill brand">—</span></div>
    </div>
    <div class="navgroup"><button class="iconbtn" id="next" aria-label="Sonraki">→</button></div>
  </header>
  <main id="main"></main>
  <div class="hint">←/→ gez · boşluk: dinle · G: genel bakış · Esc: geri</div>
</div>
<script>
const SLIDES = __PAYLOAD__;
const PHONEME_LABELS = __PHONEME_LABELS__;
const PRIMARY = "mms-1b";
const SECONDARY = "xls-r-300m";
let cur = 0, sel = 0, mode = "overview";

const main = document.getElementById("main");
const ix = document.getElementById("ix");
const navWord = document.getElementById("navWord");
const kind = document.getElementById("kind");
const prev = document.getElementById("prev");
const next = document.getElementById("next");
const ovBtn = document.getElementById("ovBtn");
document.getElementById("total").textContent = String(SLIDES.length);

function esc(v){ return String(v==null?"":v).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmt(v,d){ d=d==null?2:d; return typeof v==="number"&&Number.isFinite(v)?v.toFixed(d).replace(/\.00$/,""):"—"; }
function pct(v){ return typeof v==="number"&&Number.isFinite(v)?Math.round(v*100)+"%":"—"; }
function phName(p){ if(!p) return ""; const l=PHONEME_LABELS[p]; return l||p; }
function withLabel(p){ const l=PHONEME_LABELS[p]; return l&&l!==p?p+" ("+l+")":p; }
function label(p){ if(!p) return "—"; return withLabel(p); }

/* ---- new contract readers (target → wav2vec → analysis → result/score) ---- */
function respOf(s,m){ return s.assessments?.[m] || null; }
function isUsable(r){ return r?.status==="ok" && !!r?.result; }
function resultOf(s,m){ const r=respOf(s,m); return isUsable(r)?r.result:null; }
function resultPhones(s,m){ const r=resultOf(s,m); return Array.isArray(r?.phones)?r.phones:[]; }
function primaryResult(s){ return resultOf(s,PRIMARY); }
function scoreBlock(s,m){ const r=respOf(s,m); return (r?.result?.score)||r?.score||null; }
function summaryOf(s,m){ const r=resultOf(s,m); return r?.summary||null; }

/* item_score is 0–100; band derived from the percent (no backend band anymore) */
function scoreBand(pct){ if(pct==null) return null; if(pct>=85) return "strong"; if(pct>=70) return "pass"; if(pct>=50) return "warning"; return "needs_work"; }
function bandTone(b){ return ({strong:"ok",pass:"ok",warning:"warn",needs_work:"bad"})[b]||"brand"; }
function bandTr(b){ return ({strong:"güçlü",pass:"geçer",warning:"uyarı",needs_work:"zayıf",unavailable:"yok"})[b]||"—"; }
function scoreInfo(s){
  const sc=scoreBlock(s,PRIMARY);
  if(sc){ const v=Math.round(sc.item_score); return {score:v, band:scoreBand(v), model:PRIMARY}; }
  const sc2=scoreBlock(s,SECONDARY);
  if(sc2){ const v=Math.round(sc2.item_score); return {score:v, band:scoreBand(v), model:SECONDARY}; }
  return {score:null, band:null, model:null};
}
/* deviations = phones that are not strictly correct */
function errorCount(s,m){ return resultPhones(s,m).filter(p=>p.strict_status!=="correct").length; }

function defaultSel(s){
  const phones=resultPhones(s,PRIMARY); let found=-1;
  for(let i=0;i<phones.length;i++){ if(found<0 && phones[i].strict_status!=="correct"){ found=i; } }
  return found>=0?found:0;
}
function selectedPhone(s){ return resultPhones(s,PRIMARY)[sel]||null; }

/* ---- strict_status helpers (correct/incorrect/missing/extra) ---- */
function statusTr(st){ return ({correct:"doğru",incorrect:"sapma",missing:"eksik",extra:"fazla"})[st]||st||"—"; }
function statusTone(st){ return ({correct:"ok",incorrect:"bad",missing:"warn",extra:"extra"})[st]||"muted"; }
function statusTokCls(st){ return ({correct:"tok-equal",incorrect:"tok-substitution",missing:"tok-deletion",extra:"tok-insertion"})[st]||"tok-masked"; }
/* humanize the raw score_reason into Turkish */
function reasonTr(r){
  return ({
    exact:"tam eşleşme",
    phonetic_distance:"fonetik mesafe cezası",
    analysis_corrected:"akustik analiz düzeltmesi",
    duration_override:"süre kararı",
    missing:"hedef ses duyulmadı",
    extra:"fazladan ses",
  })[r]||r||"—";
}
/* describe an acoustic override: changed_by_analysis = {from,to,source,change_id} */
function overrideNote(cba){
  if(!cba || typeof cba!=="object") return "";
  const from=cba.from, to=cba.to, src=cba.source;
  if(from==null && to==null) return "";
  const srcTr=({duration:"süre",pairwise:"ikili dedektör",nearest_atlas:"en yakın atlas",curated_stop_place:"küratörlü stop-yeri"})[src]||src||"akustik analiz";
  return `model /${from??"—"}/ duydu → analiz /${to??"—"}/ olarak düzeltti (kaynak: ${srcTr})`;
}

function ring(score,tone){
  const s=typeof score==="number"?Math.max(0,Math.min(100,score)):0;
  const r=50, c=2*Math.PI*r, off=c*(1-s/100);
  return `<div class="ringwrap"><svg width="112" height="112" viewBox="0 0 112 112"><circle cx="56" cy="56" r="${r}" fill="none" stroke="#e1e6ef" stroke-width="9"/><circle cx="56" cy="56" r="${r}" fill="none" stroke="var(--${tone})" stroke-width="9" stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}" transform="rotate(-90 56 56)"/></svg><div class="ringtext"><div><div class="score" style="color:var(--${tone})">${typeof score==="number"?score:"—"}</div><small>${typeof score==="number"?"/ 100":""}</small></div></div></div>`;
}
function pipe(){
  const stages=["G2P","wav2vec","koç","geri bildirim"];
  return `<div class="pipe">`+stages.map((st,i)=>{
    const cls=i<2?"done":i===2?"active":"";
    const arrow=i<stages.length-1?`<span class="parrow">→</span>`:"";
    return `<span class="pstage ${cls}">${esc(st)}</span>${arrow}`;
  }).join("")+`</div>`;
}

/* per-phone alignment strip from result.phones[] (new contract).
   Each row already carries expected/observed/strict_status/score. */
function phoneStrip(phones,clickable){
  if(!Array.isArray(phones)||!phones.length) return `<span class="empty">—</span>`;
  return `<div class="diff-strip">`+phones.map((p,i)=>{
    const cls=statusTokCls(p.strict_status);
    const selc=(clickable&&i===sel)?" sel":"";
    const top=p.expected??"∅", bot=p.observed??"∅";
    const same=p.expected===p.observed;
    const ov=p.changed_by_analysis?'<span class="tok-flag" title="akustik analiz düzeltti">⟳</span>':"";
    const tag=clickable?`data-eidx="${i}"`:"disabled";
    const scorePct=Math.round((typeof p.score==="number"?p.score:0)*100);
    const scoreTone=p.score>=0.85?"ok":p.score>=0.6?"warn":"bad";
    const bottom=same?"":`<span class="tok-line"><span class="tok-label">sen</span>${esc(bot)}${ov}</span>`;
    return `<button class="tok ${cls}${selc}" ${tag}><span class="tok-line"><span class="tok-label">bek</span>${esc(top)}</span>${bottom}<span class="tok-line tok-scoreline"><span class="tok-fill tok-fill-${scoreTone}" style="width:${scorePct}%"></span><span class="tok-pct">${scorePct}%</span></span></button>`;
  }).join("")+`</div>`;
}
/* free-decode vs target comparison strip (raw IPA list aligned to target) */
function ipaSeqStrip(targetSeq,observedSeq){
  const expected=Array.isArray(targetSeq)?targetSeq:[];
  const predicted=Array.isArray(observedSeq)?observedSeq:[];
  const n=expected.length, m=predicted.length;
  if(!n&&!m) return `<span class="empty">—</span>`;
  const dp=Array.from({length:n+1},()=>Array(m+1).fill(0));
  const back=Array.from({length:n+1},()=>Array(m+1).fill(null));
  for(let i=1;i<=n;i++){dp[i][0]=i;back[i][0]="missing";}
  for(let j=1;j<=m;j++){dp[0][j]=j;back[0][j]="extra";}
  for(let i=1;i<=n;i++)for(let j=1;j<=m;j++){
    const sc=expected[i-1]===predicted[j-1]?0:1;
    const cand=[[dp[i-1][j-1]+sc,sc?"incorrect":"correct"],[dp[i-1][j]+1,"missing"],[dp[i][j-1]+1,"extra"]];
    cand.sort((a,b)=>a[0]-b[0]); dp[i][j]=cand[0][0]; back[i][j]=cand[0][1];
  }
  const ops=[]; let i=n,j=m;
  while(i>0||j>0){
    const op=back[i][j]||"extra";
    if(op==="correct"||op==="incorrect"){ ops.push({st:op,expected:expected[i-1],observed:predicted[j-1]}); i--; j--; }
    else if(op==="missing"){ ops.push({st:"missing",expected:expected[i-1],observed:null}); i--; }
    else { ops.push({st:"extra",expected:null,observed:predicted[j-1]}); j--; }
  }
  ops.reverse();
  return `<div class="diff-strip">`+ops.map(op=>{
    const cls=statusTokCls(op.st);
    const top=op.expected??"∅", bot=op.observed??"∅";
    const same=op.expected===op.observed;
    const bottom=same?"":`<span class="tok-line"><span class="tok-label">duy</span>${esc(bot)}</span>`;
    return `<button class="tok ${cls}" disabled><span class="tok-line"><span class="tok-label">bek</span>${esc(top)}</span>${bottom}</button>`;
  }).join("")+`</div>`;
}

/* feedback sentences (ported from frontend Learner.tsx PhonemeRowView) */
function describePhone(p){
  const exp=p.expected, obs=p.observed;
  const ov=overrideNote(p.changed_by_analysis);
  if(p.strict_status==="correct"){
    const txt=ov?`${withLabel(exp||obs)} hedef sesle eşleşti. ${ov}.`:`${withLabel(exp||obs)} — doğru söyledin.`;
    return {tone:"ok",badge:"Doğru",glyph:exp||obs,text:txt};
  }
  if(p.strict_status==="missing")
    return {tone:"warn",badge:"Eksik",glyph:exp||"∅",text:`${withLabel(exp)} sesini söylemedin — bu sesi eklemelisin.`};
  if(p.strict_status==="extra")
    return {tone:"extra",badge:"Fazla",glyph:obs||"+",text:`Fazladan ${withLabel(obs)} sesi var — bunu çıkarmalısın.`};
  /* incorrect (sapma) */
  const base=`Burada ${withLabel(exp)} bekleniyordu; sen ${withLabel(obs)} söyledin.`;
  return {tone:"bad",badge:"Yanlış ses",glyph:exp||obs,text:ov?`${base} (${ov})`:base};
}
function headline(s){
  const r=primaryResult(s);
  if(!r){ const resp=respOf(s,PRIMARY); return `Değerlendirilemedi: ${esc(resp?.error_code||resp?.status||"bilinmiyor")}`; }
  const sm=summaryOf(s,PRIMARY)||{};
  const inc=sm.incorrect_count||0, mis=sm.missing_count||0, ext=sm.extra_count||0;
  const dev=inc+mis+ext;
  if(dev===0) return "Tüm sesleri doğru söyledin.";
  if(mis>0) return `${mis} hedef ses eksik ölçüldü.`;
  if(ext>0) return `${ext} fazladan ses skoru düşürdü.`;
  return `${inc} seste sapma var.`;
}
function feedbackList(s){
  const phones=resultPhones(s,PRIMARY); if(!phones.length) return "";
  return phones.map(p=>{
    const d=describePhone(p);
    const sc=typeof p.score==="number"?Math.round(p.score*100)+"%":"";
    return `<div class="fb-row"><span class="fb-glyph b-${d.tone} t-${d.tone}">${esc(d.glyph)}</span>`+
      `<div><span class="fb-badge t-${d.tone}">${esc(d.badge)}</span><span class="fb-text">${esc(d.text)}</span></div>`+
      `${sc?`<span class="fb-pct b-${d.tone} t-${d.tone}">${sc}</span>`:`<span></span>`}</div>`;
  }).join("");
}

/* per-phone score breakdown (new contract: fonetik-mesafe skoru + kalite) */
function bar(name,value,note){
  const has=typeof value==="number"&&Number.isFinite(value);
  const w=has?Math.round(value*100):0;
  const tone=!has?"muted":value>0.85?"ok":value>0.6?"warn":"bad";
  return `<div class="ts-row"><div class="ts-name">${name}</div><div class="ts-bar"><div class="ts-fill ts-${tone}" style="width:${w}%"></div></div><div class="ts-val">${has?w+"%":"—"}</div></div><div class="ts-note">${esc(note)}</div>`;
}
/* summarize distance_features (place/manner/voicing/height/backness… match flags) */
function featureNote(df){
  if(!df||typeof df!=="object") return "özellik ayrıntısı yok";
  if(df.exact===true) return "tam eşleşme";
  const diffs=[];
  for(const k of Object.keys(df)){
    if(k==="exact") continue;
    const v=df[k];
    if(v&&typeof v==="object"&&"match" in v&&v.match===false){
      const kTr=({place:"yer",manner:"biçim",voicing:"ötümlülük",class:"sınıf",height:"yükseklik",backness:"artlık",rounding:"yuvarlaklık",length:"uzunluk"})[k]||k;
      diffs.push(kTr);
    }
  }
  return diffs.length?`farklı: ${diffs.join(", ")}`:"özellikler eşleşti";
}
/* length is an independent, non-gating channel — it never folds into quality */
function lengthNoteTr(n){ return ({short:"kısa",long:"uzun",expected:"normal"})[n]||n||"—"; }
function lengthSummary(L){ if(!L) return null; const parts=[]; if(L.note) parts.push(lengthNoteTr(L.note)); if(typeof L.duration_ms==="number") parts.push(Math.round(L.duration_ms)+" ms"); if(typeof L.score==="number") parts.push("tipiklik "+pct(L.score)); if(typeof L.z==="number") parts.push("z="+L.z.toFixed(2)); return parts.length?parts.join(" · "):null; }

function twoScore(s){
  const p=selectedPhone(s);
  if(!p) return `<div class="ts-empty">Bir fonem seç (üstteki kutucuklar).</div>`;
  const sym=p.expected||p.observed;
  const idNote=p.observed&&p.observed!==p.expected?`hedef /${p.expected}/ yerine /${p.observed}/ duyuldu`:`hedef /${sym}/ duyuldu`;
  const ov=overrideNote(p.changed_by_analysis);
  const head=`<div class="ts-head">Seçili: <b>${esc(label(sym))}</b> · durum: <b>${esc(statusTr(p.strict_status))}</b> · neden: <b>${esc(reasonTr(p.score_reason))}</b></div>`;
  const Lsum=lengthSummary(p.length);
  const rows=bar("fonem skoru",p.score,idNote)+
    bar("kalite (atlas/GMM)",p.quality,p.quality==null?"kalite yalnızca hedef ses üretilince ölçülür":"yerel-konuşmacı yakınlığı · süreden bağımsız")+
    bar("süre tipikliği",p.length?.score,Lsum?("süre: "+Lsum+" · kaliteyi etkilemez"):"süre referansı yok")+
    `<div class="ts-row"><div class="ts-name">özellik</div><div class="ts-bar"><div class="ts-fill ts-${p.score>0.85?"ok":p.score>0.6?"warn":"bad"}" style="width:0%"></div></div><div class="ts-val">—</div></div><div class="ts-note">${esc(featureNote(p.distance_features))}</div>`;
  const ovRow=ov?`<div class="ts-note" style="margin-top:6px;color:var(--brand)">⟳ ${esc(ov)}</div>`:"";
  return head+rows+ovRow;
}

/* model'in ham duyduğu (wav2vec) vs analiz sonrası (override'lar) */
function rawDecodeBlock(s){
  const resp=respOf(s,PRIMARY);
  const ham=resp?.wav2vec?.ipa || "—";
  const analiz=resp?.analysis?.ipa || "—";
  const changes=Array.isArray(resp?.analysis?.changes)?resp.analysis.changes:[];
  const same=changes.length===0 && ham===analiz;
  const note=same
    ? "Akustik analiz modelin duyduğunu olduğu gibi bıraktı (override yok)."
    : `Akustik dedektörler ${changes.length||"bir"} konumda modelin ham çıktısını düzeltti.`;
  return `<div class="rd-line"><span class="rd-tag">ham model</span><span class="rd-val">${esc(ham)}</span></div>`+
    `<div class="rd-line"><span class="rd-tag">analiz sonrası</span><span class="rd-val">${esc(analiz)}</span></div>`+
    `<div class="rd-note">${esc(note)}</div>`;
}

/* technical drill-down */
function techTable(s,model){
  const resp=respOf(s,model); const phones=resultPhones(s,model); const r=resp?.result;
  if(!isUsable(resp)) return `<div class="empty">Model çıktısı kullanılamıyor: ${esc(resp?.error_code||resp?.status||resp?.recording_quality?.reason||"bilinmiyor")}</div>`;
  const sc=r?.score||{}; const sm=r?.summary||{};
  const metrics=`<div class="metrics">
    <div class="metric"><span>Skor</span><b>${fmt(sc.item_score,0)} / ${fmt(sc.max_item_score,0)}</b></div>
    <div class="metric"><span>Kelime</span><b>${pct(sc.word_score)}</b></div>
    <div class="metric"><span>Fonem ort.</span><b>${pct(sc.phoneme_core)}</b></div>
    <div class="metric"><span>Doğruluk</span><b>${pct(sm.accuracy)}</b></div>
    <div class="metric"><span>Fazla cezası</span><b>${pct(sc.extra_penalty)}</b></div>
  </div>`;
  const rows=phones.map((p,i)=>{
    const sym=p.expected||p.observed;
    const t=p.timing||{};
    const ms=(t.start_ms!=null&&t.end_ms!=null)?`${t.start_ms}–${t.end_ms}`:"—";
    const ov=overrideNote(p.changed_by_analysis);
    const gozlem=`ham ${esc(p.raw_observed||"—")}${p.analysis_observed&&p.analysis_observed!==p.raw_observed?` → analiz ${esc(p.analysis_observed)}`:""}${ov?` <span class="muted">(${esc(ov)})</span>`:""}`;
    const L=p.length||{}; const Lcell=L.note?`${esc(lengthNoteTr(L.note))} <span class="muted">${pct(L.score)}</span>`:"—";
    return `<tr><td class="mono">${i+1}</td><td class="mono"><b>${esc(p.expected??"∅")}</b> <span class="muted">${esc(sym?phName(sym):"")}</span></td><td class="mono">${esc(p.observed??"∅")}</td><td class="mono">${ms}</td><td class="mono">${pct(p.score)}</td><td class="mono">${pct(p.quality)}</td><td class="mono">${Lcell}</td><td><span class="verdict ${statusTone(p.strict_status)}">${esc(statusTr(p.strict_status))}</span></td><td class="mono">${gozlem}</td></tr>`;
  }).join("");
  const q=resp?.recording_quality||{}; const dbg=resp?.debug||{};
  const ss=dbg.speech_start_ms_original, se=dbg.speech_end_ms_original;
  return metrics+`<div class="tablewrap"><table><thead><tr><th>#</th><th>Beklenen</th><th>Gözlem</th><th>ms</th><th>Skor</th><th>Kalite</th><th>Süre</th><th>Durum</th><th>Ham → Analiz</th></tr></thead><tbody>${rows}</tbody></table></div>`+
    `<div class="metaline">Konuşma bölgesi: ${ss??"—"}–${se??"—"} ms · Kayıt: ${esc(q.status||"—")}, ${fmt(q.duration_ms,0)} ms · Ham IPA: ${esc(resp?.wav2vec?.ipa||"—")} · Analiz IPA: ${esc(resp?.analysis?.ipa||"—")} · Skor sürümü: ${esc(sc.score_version||"—")}</div>`;
}
function xlsBlock(s){
  const resp=respOf(s,SECONDARY); const target=s.g2p?.phonemes||[];
  if(!isUsable(resp)) return `<div class="empty">XLS-R çıktısı kullanılamıyor: ${esc(resp?.error_code||resp?.status||"bilinmiyor")}</div>`;
  const ham=resp?.wav2vec?.ipa||"—";
  const observed=Array.isArray(resp?.wav2vec?.phonemes)?resp.wav2vec.phonemes:[];
  const mmsErr=errorCount(s,PRIMARY), xlsErr=errorCount(s,SECONDARY);
  const cmp = xlsErr>mmsErr?"XLS-R bu kayıtta MMS-1B'den daha fazla sapma üretti — model seçimini destekler.":xlsErr<mmsErr?"XLS-R bu kayıtta daha az sapma gördü.":"İki model bu kayıtta benzer sonuç verdi.";
  return `<div class="xls-summary"><b>XLS-R ham:</b> <span class="mono">${esc(ham)}</span> · ${xlsErr} sapma</div>`+
    ipaSeqStrip(target,observed)+
    `<div class="metaline">${esc(cmp)}</div>`+techTable(s,SECONDARY);
}

/* ---- full API payload view: every block from every model on screen ---- */
function adSub(t){ return `<div style="margin:12px 0 4px;font:600 10px/1.2 var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.08em">${esc(t)}</div>`; }
function kvgrid(pairs){ return `<div class="metrics">`+pairs.map(([k,v])=>`<div class="metric"><span>${esc(k)}</span><b>${esc(v==null||v===""?"—":v)}</b></div>`).join("")+`</div>`; }
function jsonBox(label,obj){ return `<details class="drill"><summary>${esc(label)}</summary><pre style="max-height:420px;overflow:auto;white-space:pre-wrap;word-break:break-word;font:11px/1.5 var(--mono);color:var(--muted);background:var(--surface-2);border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:8px">${esc(JSON.stringify(obj==null?null:obj,null,2))}</pre></details>`; }
function wav2vecLine(s,model){ const w=respOf(s,model)?.wav2vec; const ph=Array.isArray(w?.phonemes)?w.phonemes:[]; return `<div class="metaline">ham IPA: <span class="mono">${esc(w?.ipa||"—")}</span> · fonemler: <span class="mono">${esc(ph.join(" ")||"—")}</span> · ${ph.length} fonem</div>`; }
function analysisTable(s,model){
  const ph=respOf(s,model)?.analysis?.phones; const list=Array.isArray(ph)?ph:[];
  if(!list.length) return `<div class="empty">analiz fonemi yok</div>`;
  const rows=list.map((p,i)=>{
    const ms=(p.start_ms!=null&&p.end_ms!=null)?`${p.start_ms}–${p.end_ms}`:"—";
    const raw=p.raw_ipa&&p.raw_ipa!==p.ipa?`${esc(p.raw_ipa)} → ${esc(p.ipa)}`:esc(p.ipa);
    const Ls=lengthSummary({score:p.length_score,note:p.length_note,z:p.length_z,duration_ms:p.duration_ms});
    return `<tr><td class="mono">${i+1}</td><td class="mono">${raw}</td><td class="mono">${ms}</td><td class="mono">${pct(p.confidence)}</td><td class="mono">${pct(p.quality)}</td><td class="mono">${Ls?esc(Ls):"—"}</td><td class="mono">${p.changed_by_analysis?"⟳ "+esc(p.analysis_source||""):"—"}</td></tr>`;
  }).join("");
  return `<div class="tablewrap"><table><thead><tr><th>#</th><th>IPA (ham→analiz)</th><th>ms</th><th>Güven</th><th>Kalite</th><th>Süre</th><th>Analiz</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}
function modelAllData(s,model){
  const r=respOf(s,model);
  if(!r) return `<div class="empty">${esc(model)}: yanıt yok</div>`;
  const t=r.target||{}, sm=r.result?.summary||{}, dbg=r.debug||{}, q=r.recording_quality||{};
  const top=kvgrid([["model",r.model_id||model],["durum",r.status],["hata",r.error_code],["kayıt",q.status],["kayıt ms",fmt(q.duration_ms,0)]]);
  const tgt=`<div class="metaline"><b>Hedef (G2P):</b> <span class="mono">${esc(t.ipa||"—")}</span> · puanlanabilir: <span class="mono">${esc((t.scoreable_phonemes||[]).join(" ")||"—")}</span></div>`;
  const sumKv=kvgrid([["hedef",sm.target_count],["doğru",sm.correct_count],["sapma",sm.incorrect_count],["eksik",sm.missing_count],["fazla",sm.extra_count],["doğruluk",pct(sm.accuracy)],["fonem ort.",pct(sm.mean_phone_score)],["kalite ort.",pct(sm.quality_score)]]);
  const ac=Array.isArray(dbg.analysis_candidates)?dbg.analysis_candidates.length:0;
  const pc=Array.isArray(dbg.posterior_candidates)?dbg.posterior_candidates.length:0;
  const dbgKv=kvgrid([["analiz adayları",ac],["posterior adayları",pc],["konuşma ms",`${dbg.speech_start_ms_original??"—"}–${dbg.speech_end_ms_original??"—"}`]]);
  return `<div style="margin-bottom:6px"><div class="eyebrow">${esc(model)} — tüm bloklar</div>${top}${tgt}`+
    adSub("wav2vec → analiz fonemleri")+wav2vecLine(s,model)+analysisTable(s,model)+
    adSub("result — fonem skor tablosu")+techTable(s,model)+
    adSub("summary")+sumKv+adSub("debug")+dbgKv+
    jsonBox(model+" — ham JSON (her alan)",r)+`</div>`;
}
function allDataBlock(s){
  return [PRIMARY,SECONDARY].map(m=>modelAllData(s,m)).join('<hr style="border:none;border-top:1px solid var(--line);margin:16px 0">')+
    jsonBox("Slot ham JSON (g2p + her iki model)",s);
}

/* slide */
function paintSlide(){
  const s=SLIDES[cur];
  const info=scoreInfo(s); const tone=bandTone(info.band);
  const fileNo=Math.floor(cur/2)+1;
  const sib=cur^1; const sibKind=SLIDES[sib]?.kind==="correct"?"doğru":"hatalı";
  main.innerHTML=`<section class="slide">
    ${pipe()}
    <div class="slidegrid">
      <div class="col">
        <div class="card hero-card">
          <div class="hero">
            ${ring(info.score,tone)}
            <div>
              <h1>${esc(s.word)}</h1>
              <div class="ipa">${esc(s.g2p?.ipa||s.ipa)}</div>
              <div class="pairline"><span>Çift ${fileNo}/${Math.ceil(SLIDES.length/2)}</span><span class="pill ${s.kind==="correct"?"ok":"bad"}">${s.kind==="correct"?"doğru kayıt":"hatalı kayıt"}</span>${info.band?`<span class="pill ${tone}">${esc(bandTr(info.band))}</span>`:""}<button class="pairbtn" id="sibBtn">${sibKind} kayda geç →</button></div>
              <div class="audiorow"><button class="audiobtn" id="audioBtn" aria-label="Dinle">▶</button><audio id="player" preload="metadata" src="${esc(s.audio)}"></audio><span class="audiocap">${fmt(s.duration_ms,0)} ms · dinle</span></div>
            </div>
          </div>
        </div>
        ${s.annotation?`<div class="card"><div class="anno-box"><div class="anno-label">Açıklama</div>${esc(s.annotation)}</div></div>`:""}
        <div class="card">
          <div class="eyebrow">Hizalama — beklenen ↔ duyulan</div>
          ${phoneStrip(resultPhones(s,PRIMARY),true)}
          <div class="hintline">Bir kutucuğa tıkla → sağda o sesin skor ayrıntısı. ⟳ = akustik analiz düzeltti.</div>
        </div>
        <div class="card">
          <div class="eyebrow">Geri bildirim</div>
          <p class="headline">${esc(headline(s))}</p>
          ${feedbackList(s)}
        </div>
      </div>
      <div class="col">
        <div class="card">
          <div class="eyebrow">Seçili fonem — fonetik-mesafe skoru</div>
          ${twoScore(s)}
        </div>
        <div class="card">
          <div class="eyebrow">Modelin ham duyduğu vs akustik analiz</div>
          ${rawDecodeBlock(s)}
        </div>
      </div>
    </div>
    <details class="drill"><summary>Teknik detay (MMS-1B fonem tablosu)</summary><div class="drill-body">${techTable(s,PRIMARY)}</div></details>
    <details class="drill"><summary>XLS-R karşılaştırması</summary><div class="drill-body">${xlsBlock(s)}</div></details>
    <details class="drill"><summary>Tüm API verisi (her blok, her model)</summary><div class="drill-body">${allDataBlock(s)}</div></details>
  </section>`;
  main.querySelectorAll(".tok[data-eidx]").forEach(b=>b.addEventListener("click",()=>{ sel=parseInt(b.dataset.eidx,10); paintSlide(); }));
  const sb=document.getElementById("sibBtn"); if(sb) sb.addEventListener("click",()=>{ cur=sib; sel=defaultSel(SLIDES[cur]); render(); window.scrollTo({top:0,behavior:"instant"}); });
  setupAudio();
}
function setupAudio(){
  const audio=document.getElementById("player"); const btn=document.getElementById("audioBtn");
  if(!audio||!btn) return;
  const PLAY='<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.14v13.72a1 1 0 0 0 1.55.83l9.43-6.86a1 1 0 0 0 0-1.66L9.55 4.31A1 1 0 0 0 8 5.14z"/></svg>';
  const PAUSE='<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
  const sync=()=>{ const p=!audio.paused&&!audio.ended; btn.innerHTML=p?PAUSE:PLAY; btn.classList.toggle("playing",p); };
  btn.onclick=()=>{ audio.paused?audio.play():audio.pause(); };
  audio.onplay=sync; audio.onpause=sync; audio.onended=sync; sync();
}

/* overview */
function tile(s,i){
  const info=scoreInfo(s); const r=primaryResult(s); const wrong=s.kind!=="correct";
  let result="değerlendirilemedi", tone="muted";
  if(r){ const e=errorCount(s,PRIMARY);
    if(wrong){ if(e>0){ result=`${e} hata yakalandı`; tone="ok"; } else { result="hata kaçırıldı"; tone="warn"; } }
    else { if(e===0){ result="doğru kabul"; tone="ok"; } else { result=`${e} yanlış alarm`; tone="bad"; } }
  }
  return `<button class="ov-tile" data-go="${i}"><div class="ov-top"><span class="ov-word">${esc(s.word)}</span><span class="pill ${s.kind==="correct"?"ok":"bad"}">${s.kind==="correct"?"doğru":"hatalı"}</span></div><div class="ov-bottom"><span class="ov-score" style="color:var(--${bandTone(info.band)})">${info.score==null?"—":info.score}</span><span class="ov-result t-${tone}">${esc(result)}</span></div></button>`;
}
function renderOverview(){
  main.innerHTML=`<section class="overview"><div class="ov-head"><h1>Türkçe Telaffuz Koçu — ${SLIDES.length} kayıt</h1><p>Her kelimenin doğru ve hatalı kaydı. Bir karoya tıkla → ayrıntılı değerlendirme.</p></div><div class="ov-grid">${SLIDES.map(tile).join("")}</div></section>`;
  main.querySelectorAll(".ov-tile").forEach(b=>b.addEventListener("click",()=>{ cur=parseInt(b.dataset.go,10); sel=defaultSel(SLIDES[cur]); mode="slide"; render(); window.scrollTo({top:0,behavior:"instant"}); }));
}

function updateTopbar(){
  const ov=mode==="overview";
  ovBtn.classList.toggle("active",ov);
  prev.disabled=ov||cur===0; next.disabled=ov||cur===SLIDES.length-1;
  if(ov){ ix.textContent="—"; navWord.textContent="Genel bakış"; kind.style.display="none"; }
  else {
    const s=SLIDES[cur]; ix.textContent=String(cur+1);
    navWord.textContent=s.word; kind.style.display="";
    kind.textContent=s.kind==="correct"?"doğru":"hatalı"; kind.className=`pill ${s.kind==="correct"?"ok":"bad"}`;
    document.title=`Demo ${cur+1}/${SLIDES.length} — ${s.word}`;
  }
}
function render(){ if(mode==="overview") renderOverview(); else paintSlide(); updateTopbar(); }
function go(d){ if(mode!=="slide") return; const n=Math.max(0,Math.min(SLIDES.length-1,cur+d)); if(n!==cur){ cur=n; sel=defaultSel(SLIDES[cur]); render(); window.scrollTo({top:0,behavior:"instant"}); } }

prev.addEventListener("click",()=>go(-1));
next.addEventListener("click",()=>go(1));
ovBtn.addEventListener("click",()=>{ mode=mode==="overview"?"slide":"overview"; render(); });
window.addEventListener("keydown",e=>{
  if(e.key==="g"||e.key==="G"){ e.preventDefault(); mode=mode==="overview"?"slide":"overview"; render(); return; }
  if(mode==="overview") return;
  if(e.key==="ArrowLeft"){ e.preventDefault(); go(-1); }
  else if(e.key==="ArrowRight"){ e.preventDefault(); go(1); }
  else if(e.key==="Escape"){ e.preventDefault(); mode="overview"; render(); }
  else if(e.key===" "&&!["INPUT","TEXTAREA","BUTTON"].includes(e.target.tagName)){ const a=document.getElementById("player"); if(a){ e.preventDefault(); a.paused?a.play():a.pause(); } }
});
render();
</script>
</body>
</html>
"""


def render_html(slides: list[dict[str, object]]) -> str:
    payload = js_json(slides)
    labels = js_json(PHONEME_LABELS_TR)
    return _TEMPLATE.replace("__PAYLOAD__", payload).replace(
        "__PHONEME_LABELS__", labels
    )


def assemble_from_capture(session: str) -> None:
    """Copy a demo-capture session's takes into ``demo_static/audio`` slots.

    Maps ``demo_capture/recordings/<session>/demoNN_{ok,err}-a*.wav`` (latest
    attempt) onto the ``audio/NN-{correct,wrong}.wav`` slots this builder reads.
    Missing takes are reported, not fatal, so a partial set still rebuilds.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from api.services.studio import recording_path  # noqa: PLC0415

    capture_root = ROOT / "data" / "demo_capture" / "recordings"
    audio_dir = DEMO_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    for slot in FALLBACK_SLOTS:
        file_no = int(slot["idx"]) // 2 + 1
        kind = str(slot["kind"])  # "correct" | "wrong"
        prompt_id = f"demo{file_no:02d}_{'ok' if kind == 'correct' else 'err'}"
        src = recording_path(capture_root, session, prompt_id, None)
        if src is None:
            missing.append(prompt_id)
            continue
        dst = audio_dir / f"{file_no:02d}-{kind}.wav"
        shutil.copyfile(src, dst)
        sys.stderr.write(f"  {prompt_id} -> {dst.name}\n")
    if missing:
        sys.stderr.write(
            f"UYARI: '{session}' oturumunda eksik kayıt: {', '.join(missing)}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true", help="Use fallback slots without API fetch.")
    parser.add_argument("--local", action="store_true", help="Run local G2P and assessment pipeline.")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse the current embedded SLIDES payload and only rerender the HTML shell.",
    )
    parser.add_argument(
        "--from-capture",
        metavar="SESSION",
        help="Assemble audio/ slots from a demo-capture session before building "
        "(use together with --local for an offline rebuild).",
    )
    args = parser.parse_args()

    if args.from_capture:
        sys.stderr.write(f"[from-capture] '{args.from_capture}' oturumu toplanıyor\n")
        assemble_from_capture(args.from_capture)

    base_slots = load_slots()
    annotations = load_annotations()

    def _attach_annotation(slide: dict[str, object]) -> dict[str, object]:
        key = f"{slide['idx']}-{slide['kind']}"
        slide["annotation"] = annotations.get(key)
        return slide
    slides: list[dict[str, object]] = []
    if args.reuse_existing:
        slides = [_attach_annotation(s) for s in load_existing_slides()]
    elif args.skip_fetch:
        slides = [
            _attach_annotation({
                **slot,
                "audio": f"audio/{int(slot['idx']) // 2 + 1:02d}-{slot['kind']}.wav",
                "duration_ms": 0,
                "g2p": {"ipa": slot["ipa"], "phonemes": [], "syllables": [], "source": "offline", "warnings": []},
                "assessments": {},
            })
            for slot in base_slots
        ]
    elif args.local:
        runner = build_runner()
        for i, slot in enumerate(base_slots, start=1):
            sys.stderr.write(f"[{i}/{len(base_slots)} local] {slot['word']} {slot['kind']}\n")
            slides.append(_attach_annotation(local_slot(slot, runner)))
    else:
        for i, slot in enumerate(base_slots, start=1):
            sys.stderr.write(f"[{i}/{len(base_slots)}] {slot['word']} {slot['kind']}\n")
            slides.append(_attach_annotation(fetch_slot(slot)))
            time.sleep(0.15)

    INDEX.write_text(render_html(slides), encoding="utf-8")
    sys.stderr.write(f"wrote {INDEX}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
