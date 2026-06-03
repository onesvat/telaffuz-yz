import clsx from "clsx";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronRight,
  Loader2,
  Mic,
  RotateCcw,
  Search,
  ShieldCheck,
} from "lucide-react";
import type { KeyboardEvent, PointerEvent, ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { fetchG2P, postAssess, type AssessResponse, type CoachResultPhone } from "../api";
import { useSettings } from "../context/SettingsContext";
import { phonemeLabel } from "../phonemeInfo";
import { measureNoiseFloor } from "../vad";

// ---- test set definitions ----

type TestItem = { word: string; ipa: string };
type TestSet = { id: string; name: string; items: TestItem[] };

export const TEST_SETS: TestSet[] = [
  {
    id: "demo",
    name: "Demo Seti",
    items: [
      { word: "dede", ipa: "/de.ˈde/" },
      { word: "baba", ipa: "/ba.ˈba/" },
      { word: "kapı", ipa: "/kʰa.ˈpʰɯ/" },
      { word: "cam", ipa: "/d͡ʒam/" },
      { word: "kas", ipa: "/kʰas/" },
      { word: "göz", ipa: "/ɟœz/" },
      { word: "kâr", ipa: "/cʰaːɾ̞̊/" },
      { word: "çiçek", ipa: "/t͡ʃi.ˈt͡ʃec/" },
      { word: "tencere", ipa: "/tʰeɲ.ˈd͡ʒe.ɾe/" },
      { word: "ışık", ipa: "/ɯ.ˈʃɯk/" },
    ],
  },
];

// ---- types ----

type Mode = "consent" | "entry" | "free-setup" | "test-calibrate" | "test-item" | "test-feedback" | "free-record" | "free-feedback";

type ResolvedItem = TestItem & { phonemes: string[] };
type AudioContextCtor = typeof AudioContext;

// ---- main component ----

const WAVE_BARS = 36;

export function Learner() {
  const { settings } = useSettings();
  const [mode, setMode] = useState<Mode>("consent");
  const [consentAudio, setConsentAudio] = useState(false);
  const [participantName, setParticipantName] = useState("");
  const [clientId] = useState(() => getOrCreateClientId());

  // free mode state
  const [freeWord, setFreeWord] = useState("");
  const [freeG2pLoading, setFreeG2pLoading] = useState(false);
  const [freeItem, setFreeItem] = useState<ResolvedItem | null>(null);

  // test mode state
  const [selectedSet, setSelectedSet] = useState<TestSet | null>(null);
  const [resolvedItems, setResolvedItems] = useState<ResolvedItem[]>([]);
  const [resolving, setResolving] = useState(false);
  const [testIndex, setTestIndex] = useState(0);

  // shared recording state
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [lastResp, setLastResp] = useState<AssessResponse | null>(null);
  const [currentItem, setCurrentItem] = useState<ResolvedItem | null>(null);
  const [recorderKey, setRecorderKey] = useState(0);

  // ---- resolve G2P for free mode ----

  async function startFree() {
    const w = freeWord.trim();
    if (!w) return;
    setFreeG2pLoading(true);
    try {
      const r = await fetchG2P(w, { pedagogical: settings.pedagogical, use_reference: settings.useReference });
      setFreeItem({ word: r.word, ipa: r.ipa, phonemes: r.phonemes });
      setMode("test-calibrate");
    } catch {
      setFreeItem(null);
    } finally {
      setFreeG2pLoading(false);
    }
  }

  // ---- resolve G2P for test set ----

  async function startTestSet(set: TestSet) {
    setSelectedSet(set);
    setResolving(true);
    const resolved = await Promise.all(
      set.items.map(async (item) => {
        try {
          const r = await fetchG2P(item.word, {
            pedagogical: settings.pedagogical,
            use_reference: settings.useReference,
          });
          return { word: item.word, ipa: item.ipa, phonemes: r.phonemes };
        } catch {
          return { word: item.word, ipa: item.ipa, phonemes: [] };
        }
      })
    );
    setResolvedItems(resolved);
    setResolving(false);
    setTestIndex(0);
    setMode("test-calibrate");
  }

  // ---- calibration done ----

  function onCalibrated(media: MediaStream, floor: number) {
    setStream(media);
    void floor;
    if (freeItem && mode === "test-calibrate" && !selectedSet) {
      setCurrentItem(freeItem);
      setMode("free-record");
    } else {
      setCurrentItem(resolvedItems[0] ?? null);
      setMode("test-item");
    }
  }

  // ---- after assess ----

  function onAttempt(resp: AssessResponse) {
    setLastResp(resp);
    if (freeItem && !selectedSet) {
      setMode("free-feedback");
    } else {
      setMode("test-feedback");
    }
  }

  function nextTestItem() {
    const next = testIndex + 1;
    if (next >= resolvedItems.length) {
      reset();
    } else {
      setTestIndex(next);
      setCurrentItem(resolvedItems[next]);
      setLastResp(null);
      setRecorderKey((k) => k + 1);
      setMode("test-item");
    }
  }

  function reset() {
    setMode("entry");
    setSelectedSet(null);
    setResolvedItems([]);
    setTestIndex(0);
    setFreeItem(null);
    setFreeWord("");
    setCurrentItem(null);
    setLastResp(null);
    setRecorderKey(0);
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
  }

  const progress = resolvedItems.length
    ? (testIndex + (mode === "test-feedback" ? 1 : 0)) / resolvedItems.length
    : 0;

  return (
    <div className="mx-auto w-full max-w-[520px]">
      {/* Progress bar for test mode */}
      {(mode === "test-item" || mode === "test-feedback") && resolvedItems.length > 0 && (
        <div className="mb-4 space-y-1">
          <div className="flex justify-between font-mono text-[11px] text-muted">
            <span>İlerleme</span>
            <span>{testIndex + (mode === "test-feedback" ? 1 : 0)} / {resolvedItems.length}</span>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </div>
      )}

      <div className="stack pb-4">
        {/* Entry */}
        {mode === "consent" && (
          <div className="space-y-4 animate-fade-up">
            <Card>
              <div className="flex items-start gap-3">
                <ShieldCheck className="mt-1 shrink-0 text-brand" size={22} />
                <div className="min-w-0 flex-1">
                  <div className="eyebrow text-brand">Kayıt izni</div>
                  <h2 className="title mt-3">Başlamadan önce</h2>
                  <label className="mt-4 flex cursor-pointer items-start gap-3 text-[14px] leading-6 text-ink">
                    <input
                      type="checkbox"
                      className="mt-1 h-4 w-4 shrink-0 accent-brand"
                      checked={consentAudio}
                      onChange={(e) => setConsentAudio(e.target.checked)}
                    />
                    <span>
                      Ses kayıtlarımın telaffuz değerlendirmesi ve araştırma kayıt arşivi için alınmasına izin veriyorum.
                    </span>
                  </label>
                  <div className="mt-4">
                    <label className="eyebrow mb-1 block">İsim (opsiyonel)</label>
                    <input
                      className="field"
                      value={participantName}
                      onChange={(e) => setParticipantName(e.target.value)}
                      placeholder="Boş bırakabilirsin"
                    />
                  </div>
                  <button
                    className="btn-primary mt-5"
                    disabled={!consentAudio}
                    onClick={() => setMode("entry")}
                  >
                    <Check size={16} /> Devam et
                  </button>
                </div>
              </div>
            </Card>
          </div>
        )}

        {mode === "entry" && (
          <div className="space-y-4 animate-fade-up">
            <Card>
              <div className="eyebrow text-brand">Serbest Mod</div>
              <h2 className="title mt-2">İstediğin kelimeyi söyle</h2>
              <p className="lede">Kelimeyi yaz, G2P fonemlerini çek, kayıt yap ve geri bildirim al.</p>
              <form
                className="mt-5 flex gap-2"
                onSubmit={(e) => { e.preventDefault(); setMode("free-setup"); }}
              >
                <input
                  className="field flex-1"
                  value={freeWord}
                  onChange={(e) => setFreeWord(e.target.value)}
                  placeholder="örn. çiçek"
                />
                <button
                  className="btn-sm shrink-0 px-4"
                  type="submit"
                  disabled={!freeWord.trim()}
                >
                  <Search size={14} /> Başla
                </button>
              </form>
            </Card>

            {/* Separator */}
            <div className="flex items-center gap-3">
              <div className="flex-1 border-t border-line" />
              <span className="text-[12px] text-muted">veya test seti seç</span>
              <div className="flex-1 border-t border-line" />
            </div>

            {TEST_SETS.map((set) => (
              <button
                key={set.id}
                className="set-card"
                onClick={() => startTestSet(set)}
                disabled={resolving}
              >
                <span>
                  <strong className="block text-[15px] text-ink">{set.name}</strong>
                  <small className="block text-[13px] text-muted">{set.items.length} kelime</small>
                </span>
                {resolving ? (
                  <Loader2 size={16} className="animate-spin text-muted" />
                ) : (
                  <ChevronRight size={18} className="text-muted" />
                )}
              </button>
            ))}
          </div>
        )}

        {/* Free mode: confirm word before calibrate */}
        {mode === "free-setup" && (
          <Card>
            <div className="eyebrow">Kelime</div>
            <h2 className="prompt-word mt-3">{freeWord}</h2>
            <div className="mt-6 space-y-3">
              <button
                className="btn-primary"
                onClick={startFree}
                disabled={freeG2pLoading}
              >
                {freeG2pLoading ? (
                  <><Loader2 size={16} className="animate-spin" /> G2P alınıyor…</>
                ) : (
                  <><Mic size={16} /> Mikrofonu aç ve devam et</>
                )}
              </button>
              <button className="btn-ghost" onClick={() => setMode("entry")}>
                Geri
              </button>
            </div>
          </Card>
        )}

        {/* Calibration */}
        {mode === "test-calibrate" && (
          <Calibrate onReady={onCalibrated} />
        )}

        {/* Test item: recorder */}
        {(mode === "test-item" || mode === "free-record") && currentItem && (
          <Recorder
            key={`${currentItem.word}-${recorderKey}`}
            stream={stream}
            item={currentItem}
            modelId={settings.modelId}
            participantName={participantName}
            consentAudio={consentAudio}
            clientId={clientId}
            exerciseMode={freeItem && !selectedSet ? "free" : "test"}
            testSet={selectedSet?.id}
            testIndex={selectedSet ? testIndex : undefined}
            onAttempt={onAttempt}
          />
        )}

        {/* Feedback: test mode */}
        {mode === "test-feedback" && currentItem && (
          <FeedbackView
            item={currentItem}
            resp={lastResp}
            isLast={testIndex + 1 >= resolvedItems.length}
            onRetry={() => {
              setLastResp(null);
              setRecorderKey((k) => k + 1);
              setMode("test-item");
            }}
            onNext={nextTestItem}
          />
        )}

        {/* Feedback: free mode */}
        {mode === "free-feedback" && currentItem && (
          <FeedbackView
            item={currentItem}
            resp={lastResp}
            isLast
            onRetry={() => {
              setLastResp(null);
              setRecorderKey((k) => k + 1);
              setMode("free-record");
            }}
            onNext={reset}
            nextLabel="Yeni kelime"
          />
        )}
      </div>
    </div>
  );
}

function getOrCreateClientId(): string {
  const key = "telaffuz_client_id";
  const existing = window.localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID ? crypto.randomUUID() : `client-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  window.localStorage.setItem(key, id);
  return id;
}

// ---- Calibrate ----

function Calibrate({ onReady }: { onReady: (stream: MediaStream, floor: number) => void }) {
  const [phase, setPhase] = useState<"idle" | "requesting" | "measuring" | "error">("idle");
  const [level, setLevel] = useState(0);

  async function begin() {
    setPhase("requesting");
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      setPhase("measuring");
      const ctx = new AudioContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      const src = ctx.createMediaStreamSource(media);
      src.connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      let raf = 0;
      const t0 = performance.now();
      const loop = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128;
          sum += v * v;
        }
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 6));
        if (performance.now() - t0 < 1400) {
          raf = requestAnimationFrame(loop);
        } else {
          cancelAnimationFrame(raf);
          src.disconnect();
          analyser.disconnect();
          void ctx.close().catch(() => undefined);
        }
      };
      raf = requestAnimationFrame(loop);
      const floor = await measureNoiseFloor(media, 900);
      onReady(media, floor);
    } catch {
      setPhase("error");
    }
  }

  return (
    <Card>
      <Mic className="text-brand" size={24} />
      <h2 className="title mt-4">Mikrofon kontrolü</h2>
      <p className="lede">
        Mikrofon şimdi açılır ve açık kalır. Kelime ekranında basılı tutarak kayıt yapacaksın.
      </p>

      {phase === "measuring" && (
        <div className="mt-8 flex items-end justify-center gap-1.5" aria-hidden>
          {Array.from({ length: 24 }).map((_, i) => {
            const h = 10 + Math.max(0, level - i / 60) * 90;
            return (
              <span
                key={i}
                className="wave-bar wave-bar-recording"
                style={{ height: `${Math.min(64, h)}px` }}
              />
            );
          })}
        </div>
      )}

      {phase === "error" && (
        <p className="note mt-6">
          Mikrofona erişilemedi. Tarayıcı izinlerini kontrol edip tekrar dene.
        </p>
      )}

      <div className="mt-auto pt-8">
        <button
          className="btn-primary"
          onClick={begin}
          disabled={phase === "requesting" || phase === "measuring"}
        >
          {phase === "requesting" || phase === "measuring" ? (
            <><Loader2 size={16} className="animate-spin" /> Hazırlanıyor</>
          ) : phase === "error" ? (
            <><RotateCcw size={16} /> Tekrar dene</>
          ) : (
            <><Mic size={16} /> Mikrofonu aç ve başla</>
          )}
        </button>
      </div>
    </Card>
  );
}

// ---- Recorder ----

function Recorder({
  stream,
  item,
  modelId,
  participantName,
  consentAudio,
  clientId,
  exerciseMode,
  testSet,
  testIndex,
  onAttempt,
}: {
  stream: MediaStream | null;
  item: ResolvedItem;
  modelId: string;
  participantName: string;
  consentAudio: boolean;
  clientId: string;
  exerciseMode: string;
  testSet?: string;
  testIndex?: number;
  onAttempt: (r: AssessResponse) => void;
}) {
  const [phase, setPhase] = useState<"ready" | "recording" | "finalizing">("ready");
  const [levels, setLevels] = useState<number[]>(() => new Array(WAVE_BARS).fill(0));
  const [uploading, setUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const frameRef = useRef(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startedAtRef = useRef(0);
  const stopTimerRef = useRef<number | null>(null);
  const recordingRef = useRef(false);

  useEffect(() => {
    if (!stream) return;
    const Ctx = (window as unknown as {
      AudioContext?: AudioContextCtor;
      webkitAudioContext?: AudioContextCtor;
    }).AudioContext ?? (window as unknown as { webkitAudioContext?: AudioContextCtor }).webkitAudioContext;
    const audioCtx = Ctx ? new Ctx() : null;
    const analyser = audioCtx ? audioCtx.createAnalyser() : null;
    let source: MediaStreamAudioSourceNode | null = null;
    let raf = 0;

    if (audioCtx && analyser) {
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.65;
      source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      const buffer = new Uint8Array(analyser.fftSize);
      let smoothed = 0;

      const tick = () => {
        analyser.getByteTimeDomainData(buffer);
        let sum = 0;
        for (let i = 0; i < buffer.length; i += 1) {
          const v = (buffer[i] - 128) / 128;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / buffer.length);
        smoothed = smoothed * 0.72 + rms * 0.28;
        const lvl = Math.min(1, smoothed * 4.5);
        frameRef.current += 1;
        if (frameRef.current % 2 === 0) {
          setLevels((prev) => {
            const next = prev.slice(1);
            next.push(lvl);
            return next;
          });
        }
        raf = requestAnimationFrame(tick);
      };

      if (audioCtx.state === "suspended") void audioCtx.resume().catch(() => undefined);
      raf = requestAnimationFrame(tick);
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      try {
        source?.disconnect();
        analyser?.disconnect();
      } catch {
        /* ignore */
      }
      if (audioCtx && audioCtx.state !== "closed") {
        void audioCtx.close().catch(() => undefined);
      }
      if (stopTimerRef.current !== null) window.clearTimeout(stopTimerRef.current);
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.ondataavailable = null;
        recorderRef.current.onstop = null;
        recorderRef.current.stop();
      }
    };
  }, [stream]);

  async function uploadRecording(blob: Blob, durationMs: number) {
    setUploading(true);
    setErrorMsg("");
    setPhase("finalizing");
    try {
      const r = await postAssess({
        audio: blob,
        model_id: modelId,
        word: item.word,
        expected_phonemes: item.phonemes.length ? item.phonemes : undefined,
        duration_ms: durationMs,
        source: "learner",
        participant_name: participantName.trim() || undefined,
        consent_audio: consentAudio,
        client_id: clientId,
        exercise_mode: exerciseMode,
        test_set: testSet,
        test_index: testIndex,
      });
      onAttempt(r);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Analiz başarısız");
      setUploading(false);
      setPhase("ready");
    }
  }

  function startRecording() {
    if (!stream || recordingRef.current || uploading) return;
    if (typeof MediaRecorder === "undefined") {
      setErrorMsg("Bu tarayıcı ses kaydını desteklemiyor.");
      return;
    }

    try {
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recordingRef.current = true;
      startedAtRef.current = performance.now();
      setErrorMsg("");
      setPhase("recording");

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        recordingRef.current = false;
        if (stopTimerRef.current !== null) {
          window.clearTimeout(stopTimerRef.current);
          stopTimerRef.current = null;
        }
        const mime = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: mime });
        chunksRef.current = [];
        const durationMs = Math.max(1, Math.round(performance.now() - startedAtRef.current));
        if (blob.size === 0 || durationMs < 200) {
          setErrorMsg("Kayıt çok kısa. Butona basılı tutarak kelimeyi oku.");
          setPhase("ready");
          return;
        }
        void uploadRecording(blob, durationMs);
      };

      recorder.start();
      stopTimerRef.current = window.setTimeout(() => stopRecording(), 9000);
    } catch (err) {
      recordingRef.current = false;
      setPhase("ready");
      setErrorMsg(err instanceof Error ? err.message : "Kayıt başlatılamadı");
    }
  }

  function stopRecording() {
    if (!recordingRef.current) return;
    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== "recording") return;
    setPhase("finalizing");
    recorder.stop();
  }

  function handlePressStart(event: PointerEvent<HTMLElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    startRecording();
  }

  function handlePressEnd(event: PointerEvent<HTMLElement>) {
    event.preventDefault();
    stopRecording();
  }

  function handleKeyStart(event: KeyboardEvent<HTMLElement>) {
    if (event.repeat) return;
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      startRecording();
    }
  }

  function handleKeyEnd(event: KeyboardEvent<HTMLElement>) {
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      stopRecording();
    }
  }

  const recording = phase === "recording";
  const finalizing = phase === "finalizing" || uploading;

  return (
    <Card>
      <div className="eyebrow text-center">Kelimeyi oku</div>
      <h2 className="prompt-word mt-3">{item.word}</h2>
      <div className="ipa-chip">
        {item.phonemes.length
          ? item.phonemes.map((p, i) => <span key={i}>{p}</span>)
          : <span>{item.ipa}</span>}
      </div>

      {errorMsg && (
        <div className="mt-4 rounded-lg bg-bad/10 px-4 py-3 text-sm text-bad text-center">
          {errorMsg}
        </div>
      )}

      <div className="mt-10 flex flex-col items-center">
        <button
          type="button"
          className={clsx(
            "mic-orb select-none border-0 outline-none disabled:cursor-not-allowed disabled:opacity-80",
            recording && "mic-orb-recording"
          )}
          onPointerDown={handlePressStart}
          onPointerUp={handlePressEnd}
          onPointerCancel={stopRecording}
          onKeyDown={handleKeyStart}
          onKeyUp={handleKeyEnd}
          disabled={!stream || finalizing}
          aria-label={recording ? "Kaydı bırakınca yükle" : "Basılı tut ve konuş"}
        >
          {(recording || phase === "ready") && (
            <span
              className={clsx(
                "absolute inset-0 rounded-full",
                recording ? "bg-bad/30" : "bg-brand/25",
                "animate-pulse-ring"
              )}
            />
          )}
          {finalizing ? (
            <Loader2 size={32} className="animate-spin" />
          ) : (
            <Mic size={34} />
          )}
        </button>

        <div className="mt-7 flex h-16 items-end justify-center gap-[3px]" aria-hidden>
          {levels.map((lvl, i) => (
            <span
              key={i}
              className={clsx("wave-bar", recording && "wave-bar-recording")}
              style={{ height: `${6 + lvl * 56}px` }}
            />
          ))}
        </div>

        <div
          className={clsx(
            "status-pill mt-7",
            recording ? "status-recording" : finalizing ? "status-analyzing" : "status-armed"
          )}
        >
          {finalizing ? (
            "Yükleniyor"
          ) : recording ? (
            <><span className="h-2 w-2 rounded-full bg-bad" /> Kaydediliyor</>
          ) : (
            <><span className="h-2 w-2 rounded-full bg-brand" /> Mikrofon hazır</>
          )}
        </div>
      </div>

      <div className="mt-auto pt-8">
        {finalizing ? (
          <p className="text-center text-[13px] text-muted">Analiz ediliyor…</p>
        ) : (
          <button
            className={clsx("btn-primary select-none", recording && "bg-bad")}
            onPointerDown={handlePressStart}
            onPointerUp={handlePressEnd}
            onPointerCancel={stopRecording}
            onKeyDown={handleKeyStart}
            onKeyUp={handleKeyEnd}
            disabled={!stream || uploading}
          >
            {recording ? (
              <><Check size={16} /> Bırakınca yüklenecek</>
            ) : (
              <><Mic size={16} /> Basılı tut ve konuş</>
            )}
          </button>
        )}
        {!finalizing && (
          <p className="mt-3 text-center text-[13px] leading-6 text-muted">
            Mikrofon açık kalır; sadece butona basılı tuttuğun bölüm kaydedilir.
          </p>
        )}
      </div>
    </Card>
  );
}

// ---- Feedback ----

function FeedbackView({
  item,
  resp,
  isLast,
  onRetry,
  onNext,
  nextLabel = "Sonraki",
}: {
  item: ResolvedItem;
  resp: AssessResponse | null;
  isLast: boolean;
  onRetry: () => void;
  onNext: () => void;
  nextLabel?: string;
}) {
  const result = resp?.result ?? null;
  const score = resp?.score ?? result?.score ?? null;
  const phones = result?.phones ?? [];
  const pct = score ? Math.round((score.item_score / (score.max_item_score || 100)) * 100) : 0;
  const weakCount = phones.filter((phone) => phone.score < 0.6 && phone.strict_status !== "extra").length;
  const missingCount = phones.filter((phone) => phone.strict_status === "missing").length;
  const extraCount = result?.summary.extra_count ?? 0;
  const feedbackText = learnerFeedbackText(pct, weakCount, missingCount, extraCount);
  const resultModel = resp?.wav2vec?.model || resp?.model_id || "—";

  if (!resp || resp.status === "invalid" || !result) {
    const modelError = resp?.status === "model_unavailable" || resp?.status === "unknown_model";
    const runtimeError =
      resp?.status === "failed" ||
      resp?.status === "alignment_failed" ||
      resp?.status === "no_intervals" ||
      resp?.status === "no_speech";
    const note = modelError
      ? "Seçili telaffuz modeli bu ortamda hazır değil. Ayarlardan kullanılabilir bir model seç veya model dosyalarının yüklendiğini kontrol et."
      : runtimeError
        ? "Kayıt işlendi, ancak değerlendirme modeli bu denemede sonuç üretemedi. Model ve bağlantı durumunu kontrol edip tekrar dene."
        : "Kayıt çok kısa, sessiz veya anlaşılamadı. Sessiz bir ortamda, mikrofona yakın ve net söyle.";
    return (
      <div className="space-y-3 animate-fade-up">
        <Card>
          <h2 className="title">{item.word}</h2>
          <p className="note mt-4">{note}</p>
        </Card>
        <button className="btn-primary" onClick={onRetry}>
          <RotateCcw size={16} /> Tekrar dene
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fade-up">
      <Card>
        <div className="flex items-center gap-4">
          <ScoreRing value={pct} />
          <div className="min-w-0">
            <div className="eyebrow">Kelime</div>
            <h2 className="mt-1 truncate font-serif text-[26px] font-semibold">{item.word}</h2>
            <p className="mt-1 text-[13px] text-muted">
              {feedbackText}
            </p>
            <ResultSourceChips model={resultModel} />
          </div>
        </div>
      </Card>

      {phones.length > 0 && (
        <Card>
          <div className="eyebrow">Hizalama — beklenen / söylenen</div>
          <div className="diff-strip mt-3">
            {phones.map((phone) => (
              <PhoneScoreToken key={phone.index} phone={phone} />
            ))}
          </div>
        </Card>
      )}

      {phones.length > 0 && (
        <Card>
          <div className="eyebrow">Ses ses analiz</div>
          <div className="mt-1">
            {phones.map((phone) => (
              <PhonemeRowView
                key={phone.index}
                phone={phone}
              />
            ))}
          </div>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-3">
        <button className="btn-ghost" onClick={onRetry}>
          <RotateCcw size={16} /> Tekrar oku
        </button>
        <button className="btn-primary" onClick={onNext}>
          {isLast ? nextLabel : "Sonraki"} <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}

// ---- Diff widgets ----

const STATUS_STYLE: Record<CoachResultPhone["strict_status"], { glyph: string; badge: string }> = {
  correct: { glyph: "border-ok/25 bg-ok-soft text-ok", badge: "bg-ok-soft text-ok" },
  incorrect: { glyph: "border-bad/25 bg-bad-soft text-bad", badge: "bg-bad-soft text-bad" },
  missing: { glyph: "border-warn/25 bg-warn-soft text-warn", badge: "bg-warn-soft text-warn" },
  extra: { glyph: "border-extra/25 bg-extra-soft text-extra", badge: "bg-extra-soft text-extra" },
};

const STATUS_LABEL: Record<CoachResultPhone["strict_status"], string> = {
  correct: "doğru",
  incorrect: "sapma",
  missing: "eksik",
  extra: "fazla",
};

// Student-facing Turkish labels for raw score_reason codes.
const SCORE_REASON_LABEL_TR: Record<string, string> = {
  exact: "tam",
  analysis_corrected: "analizle düzeltildi",
  phonetic_distance: "fonetik mesafe",
  duration_override: "süre düzeltmesi",
  duration_verified: "süre doğrulandı",
  missing: "eksik",
  extra_penalty: "fazla",
  extra: "fazla",
};

function scoreReasonLabelTr(reason: string | null | undefined): string {
  if (!reason) return "";
  return SCORE_REASON_LABEL_TR[reason] ?? reason;
}

type AnalysisOverride = {
  from: string | null;
  to: string | null;
  source: string | null;
};

function analysisOverride(raw: unknown): AnalysisOverride | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const asStr = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
  const from = asStr(obj.from);
  const to = asStr(obj.to);
  const source = asStr(obj.source);
  if (!from && !to && !source) return null;
  return { from, to, source };
}

function overrideStudentText(ov: AnalysisOverride): string {
  const from = ov.from ?? "?";
  const to = ov.to ?? "?";
  const src = ov.source ? ` (kaynak: ${ov.source})` : "";
  return `Model /${from}/ duydu → akustik analiz /${to}/'ya düzeltti${src}.`;
}

function hasDurationEvidence(features: Record<string, unknown> | undefined): boolean {
  if (!features) return false;
  const mods = features.modifiers;
  return !!mods && typeof mods === "object" && Object.keys(mods as object).length > 0;
}

function ResultSourceChips({ model }: { model: string }) {
  return (
    <div className="mt-3 flex flex-wrap gap-1.5">
      <span
        className="rounded-full border border-line bg-canvas px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.14em] text-muted"
        title={`Bu sonuç ${model} modeliyle üretildi.`}
      >
        Model {model}
      </span>
    </div>
  );
}

function learnerFeedbackText(
  score: number,
  weakCount: number,
  missingCount: number,
  extraCount: number
): string {
  if (score >= 85 && weakCount === 0 && missingCount === 0 && extraCount === 0) {
    return "Tüm sesler güçlü ölçüldü.";
  }
  if (missingCount > 0) {
    return `${missingCount} hedef ses eksik ölçüldü.`;
  }
  if (extraCount > 0) {
    return `${extraCount} fazla ses skoru düşürdü.`;
  }
  if (weakCount > 0) {
    return `${weakCount} seste çalışılacak sapma var.`;
  }
  return "Genel telaffuz skoru hazır.";
}

function PhoneScoreToken({ phone }: { phone: CoachResultPhone }) {
  const same = phone.expected === phone.observed;
  const override = analysisOverride(phone.changed_by_analysis);
  const cls =
    phone.strict_status === "correct"
      ? "tok-equal"
      : phone.strict_status === "extra"
      ? "tok-insertion"
      : phone.strict_status === "missing"
      ? "tok-deletion"
      : "tok-substitution";
  return (
    <span className={clsx("tok relative", cls)} title={override ? overrideStudentText(override) : undefined}>
      {override && (
        <span
          className="absolute right-0.5 top-0.5 z-10 inline-flex items-center rounded-full bg-extra-soft px-1 font-mono text-[8px] uppercase tracking-wider text-extra"
          aria-label="akustik analizle düzeltildi"
        >
          A
        </span>
      )}
      <span className="tok-line">
        {!same && <span className="tok-label">bek</span>}
        <span className="tok-glyph">{phone.expected ?? "∅"}</span>
      </span>
      {!same && (
        <span className="tok-line">
          <span className="tok-label">sen</span>
          <span className="tok-glyph">{phone.observed ?? "∅"}</span>
        </span>
      )}
      <span className="tok-score">
        <span className="tok-score-track" aria-hidden>
          <span
            className={clsx(
              "tok-score-fill",
              phone.score >= 0.85 ? "bg-ok" : phone.score >= 0.6 ? "bg-warn" : "bg-bad"
            )}
            style={{ width: `${Math.max(0, Math.min(1, phone.score)) * 100}%` }}
          />
        </span>
        <span className="tok-score-label">{Math.round(phone.score * 100)}%</span>
      </span>
    </span>
  );
}

function PhonemeRowView({
  phone,
}: {
  phone: CoachResultPhone;
}) {
  const style = STATUS_STYLE[phone.strict_status];
  const shownScore = phone.score;
  const label = STATUS_LABEL[phone.strict_status];
  const symbol = phone.expected ?? phone.observed ?? "—";
  const glyph = phone.strict_status === "correct"
    ? symbol
    : phone.strict_status === "missing"
    ? "∅"
    : phone.strict_status === "extra"
    ? "+"
    : symbol;
  const observed = phone.observed && phone.observed !== phone.expected
    ? ` Gözlem: /${phone.observed}/.`
    : "";
  const baseExplanation =
    phone.strict_status === "correct"
      ? "Hedef sesle eşleşti."
      : phone.strict_status === "missing"
      ? "Bu hedef ses duyulmadı."
      : phone.strict_status === "extra"
      ? "Hedefte olmayan ek bir ses duyuldu."
      : `Hedef ses tam eşleşmedi.${observed}`;
  const override = analysisOverride(phone.changed_by_analysis);
  const durationEvidence = hasDurationEvidence(phone.distance_features);
  const reasonTr = scoreReasonLabelTr(phone.score_reason);
  const explanation = override
    ? `${baseExplanation} ${overrideStudentText(override)}`
    : baseExplanation;
  return (
    <div className="pho-row">
      <span className={clsx("pho-glyph", style.glyph)}>{glyph}</span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className={clsx("pho-badge", style.badge)}>{label}</span>
          <span className="font-mono text-[11px] text-muted">
            {phonemeLabel(symbol)}
          </span>
          {override && (
            <span
              className="rounded-full border border-extra/25 bg-extra-soft px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-extra"
              title={overrideStudentText(override)}
            >
              akustik analiz
            </span>
          )}
          {durationEvidence && (
            <span
              className="rounded-full border border-line bg-canvas px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-muted"
              title="Süre/uzunluk kanıtı skoru etkiledi (kâr/kar ayrımı)."
            >
              süre kanıtı
            </span>
          )}
        </div>
        <p className="mt-1 text-[13px] leading-6 text-ink">{explanation}</p>
        {reasonTr && reasonTr !== "tam" && (
          <p className="mt-0.5 font-mono text-[11px] text-muted">Neden: {reasonTr}</p>
        )}
      </div>
      <span
        className={clsx(
          "mt-1 rounded-md border px-2 py-1 font-mono text-[12px] font-semibold",
          shownScore > 0.5
            ? "border-ok/25 bg-ok-soft text-ok"
            : "border-bad/25 bg-bad-soft text-bad"
        )}
        title={reasonTr || phone.score_reason}
      >
        {Math.round(shownScore * 100)}%
      </span>
      {phone.strict_status === "correct" && <Check size={15} className="mt-1.5 shrink-0 text-ok" />}
      {phone.strict_status !== "correct" && (
        <AlertTriangle size={15} className="mt-1.5 shrink-0 text-warn" />
      )}
    </div>
  );
}

// ---- ScoreRing ----

function ScoreRing({ value, size = 88 }: { value: number; size?: number }) {
  const clamped = Math.max(0, Math.min(100, value));
  const stroke = 7;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const tone = clamped >= 80 ? "#1a7a44" : clamped >= 55 ? "#a76410" : "#bf3349";
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e2e6ee" strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={tone} strokeWidth={stroke} strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c - (c * clamped) / 100}
          style={{ transition: "stroke-dashoffset 0.7s ease-out" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-serif text-[20px] font-semibold leading-none" style={{ color: tone }}>
          {clamped}
        </span>
        <span className="mt-0.5 font-mono text-[9px] uppercase tracking-widest text-muted">/ 100</span>
      </div>
    </div>
  );
}

// ---- Card ----

function Card({ children }: { children: ReactNode }) {
  return <div className="card stack">{children}</div>;
}
