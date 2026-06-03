import clsx from "clsx";
import {
  Activity,
  AlertTriangle,
  Loader2,
  Mic,
  Play,
  Search,
  Square,
  Upload,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchG2P,
  fetchModels,
  postAssess,
  type AssessResponse,
  type G2PResult,
  type ModelItem,
} from "../api";
import { AssessResultView } from "../components/AssessResult";
import { useSettings } from "../context/SettingsContext";

type Phase = "idle" | "recording" | "analyzing" | "done" | "error";

export function Sandbox() {
  const { settings, update } = useSettings();
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelId, setModelId] = useState(settings.modelId);

  const [word, setWord] = useState("merhaba");
  const [g2p, setG2p] = useState<G2PResult | null>(null);
  const [expectedPhonemes, setExpectedPhonemes] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [recSeconds, setRecSeconds] = useState(0);
  const [level, setLevel] = useState(0);
  const [resp, setResp] = useState<AssessResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [g2pLoading, setG2pLoading] = useState(false);

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const audioCtx = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetchModels()
      .then((r) => {
        setModels(r.items);
        if (!modelId) {
          const def = r.items.find((m) => m.is_default && m.available) ?? r.items.find((m) => m.available);
          if (def) {
            setModelId(def.id);
            update({ modelId: def.id });
          }
        }
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const stopMeter = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (timerRef.current !== null) clearInterval(timerRef.current);
    timerRef.current = null;
    audioCtx.current?.close().catch(() => undefined);
    audioCtx.current = null;
    setLevel(0);
  }, []);

  async function runG2p() {
    if (!word.trim()) return;
    setG2pLoading(true);
    setErrorMsg("");
    try {
      const r = await fetchG2P(word.trim(), {
        pedagogical: settings.pedagogical,
        use_reference: settings.useReference,
      });
      setG2p(r);
      setExpectedPhonemes(r.phonemes.join(" "));
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "G2P başarısız");
    } finally {
      setG2pLoading(false);
    }
  }

  function attachMeter(stream: MediaStream) {
    try {
      const ctx = new AudioContext();
      audioCtx.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      const buf = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sum += v * v;
        }
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 3));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();
    } catch {
      /* meter is optional */
    }
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunks.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        stopMeter();
        const blob = new Blob(audioChunks.current, { type: "audio/webm" });
        setAudioBlob(blob);
        if (audioUrl) URL.revokeObjectURL(audioUrl);
        setAudioUrl(URL.createObjectURL(blob));
        setPhase("idle");
      };
      mediaRecorder.current = recorder;
      recorder.start();
      setRecSeconds(0);
      timerRef.current = setInterval(() => setRecSeconds((s) => s + 1), 1000);
      attachMeter(stream);
      setPhase("recording");
    } catch (err) {
      setErrorMsg("Mikrofon erişimi reddedildi: " + (err as Error).message);
      setPhase("error");
    }
  }

  function stopRecording() {
    if (mediaRecorder.current?.state === "recording") mediaRecorder.current.stop();
  }

  function onUpload(file: File | null) {
    if (!file) return;
    setAudioBlob(file);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(file));
  }

  async function analyze() {
    if (!modelId || !expectedPhonemes.trim() || !audioBlob) return;
    setPhase("analyzing");
    setErrorMsg("");
    try {
      const r = await postAssess({
        audio: audioBlob,
        model_id: modelId,
        word: word.trim() || undefined,
        expected_phonemes: expectedPhonemes.trim().split(/\s+/),
        duration_ms: Math.max(1000, recSeconds * 1000),
        source: "sandbox",
        exercise_mode: "manual",
      });
      setResp(r);
      setPhase("done");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Analiz başarısız");
      setPhase("error");
    }
  }

  const canAnalyze = Boolean(modelId && expectedPhonemes.trim() && audioBlob) && phase !== "analyzing";

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[400px_1fr]">
      {/* Left panel */}
      <div className="space-y-4">
        {/* Model picker */}
        <div className="card">
          <div className="eyebrow mb-3">Model</div>
          {models.length === 0 ? (
            <p className="text-[13px] text-muted">Modeller yükleniyor…</p>
          ) : (
            <div className="space-y-2">
              {models.map((m) => (
                <button
                  key={m.id}
                  onClick={() => {
                    if (!m.available) return;
                    setModelId(m.id);
                    update({ modelId: m.id });
                  }}
                  disabled={!m.available}
                  className={clsx(
                    "set-card",
                    m.id === modelId && "border-brand bg-brand-soft/40",
                    !m.available && "opacity-40 cursor-not-allowed"
                  )}
                >
                  <span>
                    <strong className="block text-[14px] text-ink">{m.label}</strong>
                    <small className="block font-mono text-[11px] text-muted">{m.id}</small>
                  </span>
                  <span className="flex items-center gap-1.5">
                    {m.frozen && (
                      <span className="rounded-full border border-line bg-canvas px-2 py-0.5 font-mono text-[10px] text-muted">
                        frozen
                      </span>
                    )}
                    {!m.available && (
                      <span className="rounded-full border border-bad/25 bg-bad-soft px-2 py-0.5 font-mono text-[10px] text-bad">
                        yok
                      </span>
                    )}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* G2P */}
        <div className="card space-y-3">
          <div className="eyebrow">1 · Kelime → G2P</div>
          <div className="flex gap-2">
            <input
              className="field flex-1 text-[15px]"
              value={word}
              onChange={(e) => setWord(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && runG2p()}
              placeholder="örn. merhaba"
            />
            <button className="btn-sm shrink-0" onClick={runG2p} disabled={g2pLoading || !word.trim()}>
              {g2pLoading ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
              G2P
            </button>
          </div>
          {g2p && (
            <div className="space-y-2">
              <div className="flex flex-wrap gap-1.5 font-mono text-[14px]">
                {g2p.phonemes.map((p, i) => (
                  <span key={`${p}-${i}`} className="rounded-md border border-brand/25 bg-brand-soft px-2 py-0.5 text-brand">
                    {p}
                  </span>
                ))}
              </div>
              <div className="text-[12px] text-muted">
                IPA: <span className="font-mono">{g2p.ipa}</span> · {g2p.source}
              </div>
              <div>
                <label className="eyebrow mb-1 block">Beklenen fonemler (düzenlenebilir)</label>
                <input
                  className="field font-mono text-[13px]"
                  value={expectedPhonemes}
                  onChange={(e) => setExpectedPhonemes(e.target.value)}
                />
              </div>
            </div>
          )}
          {!g2p && (
            <div>
              <label className="eyebrow mb-1 block">Fonemler (boşlukla ayrılmış)</label>
              <input
                className="field font-mono text-[13px]"
                value={expectedPhonemes}
                onChange={(e) => setExpectedPhonemes(e.target.value)}
                placeholder="örn. m æ ɾ̞̊ h a b a"
              />
            </div>
          )}
        </div>

        {/* Recorder */}
        <div className="card space-y-3">
          <div className="eyebrow">2 · Kayıt</div>
          <div className="flex flex-col items-center gap-4 py-2">
            <button
              className={clsx(
                "grid h-24 w-24 place-items-center rounded-full shadow-card transition-all",
                phase === "recording"
                  ? "scale-110 bg-bad text-white"
                  : "border-2 border-line bg-surface text-ink hover:border-brand/50"
              )}
              onClick={() => (phase === "recording" ? stopRecording() : startRecording())}
              disabled={phase === "analyzing"}
            >
              {phase === "recording" ? <Square size={30} /> : <Mic size={34} />}
            </button>

            {phase === "recording" ? (
              <div className="w-full space-y-1">
                <div className="text-center font-mono text-[12px] text-bad">● kayıt {recSeconds}s</div>
                <div className="h-2 w-full overflow-hidden rounded-full bg-line">
                  <div
                    className="h-full bg-bad transition-[width] duration-100"
                    style={{ width: `${Math.round(level * 100)}%` }}
                  />
                </div>
              </div>
            ) : (
              <p className="text-[12px] text-muted">Mikrofona bas veya dosya yükle</p>
            )}

            <label className="btn-ghost cursor-pointer">
              <Upload size={15} /> Dosya yükle
              <input
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => onUpload(e.target.files?.[0] ?? null)}
              />
            </label>

            {audioUrl && (
              <audio className="w-full" controls src={audioUrl}>
                <track kind="captions" />
              </audio>
            )}
          </div>

          <button className="btn-primary" onClick={analyze} disabled={!canAnalyze}>
            {phase === "analyzing" ? (
              <><Activity className="animate-spin" size={15} /> Analiz ediliyor…</>
            ) : (
              <><Play size={15} /> Analiz et</>
            )}
          </button>

          {phase === "analyzing" && (
            <p className="text-center text-[12px] text-muted">
              İlk çağrı MMS-FA hizalama modelini indirebilir — bu biraz sürebilir.
            </p>
          )}

          {errorMsg && (
            <div className="flex items-center gap-2 rounded-xl border border-bad/25 bg-bad-soft px-3 py-2.5 text-[13px] text-bad">
              <AlertTriangle size={15} className="shrink-0" />
              {errorMsg}
            </div>
          )}
        </div>
      </div>

      {/* Right: result */}
      <div>
        <AssessResultView resp={resp} showTechnical />
      </div>
    </div>
  );
}
