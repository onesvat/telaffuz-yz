import clsx from "clsx";
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  Loader2,
  Mic,
  Play,
  RotateCcw,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteStudioAttempt,
  fetchStudioPrompts,
  fetchStudioRecordings,
  studioAudioUrl,
  uploadStudioRecording,
  type StudioAttempt,
  type StudioCollection,
  type StudioPrompt,
} from "../api";
import { useSettings } from "../context/SettingsContext";
import { isErrorPrompt, speakerInstructionForRule } from "../errorInstructions";
import {
  measureNoiseFloor,
  startPushToTalk,
  startVad,
  type PushRecorderHandle,
  type VadHandle,
  type VadPhase,
} from "../vad";

type Stage = "setup" | "loading" | "calibrate" | "ready" | "error";

const WAVE_BARS = 36;

export function Studio({
  collection = "validation",
  title,
  intro,
  manual = false,
}: {
  collection?: StudioCollection;
  title?: string;
  intro?: string;
  manual?: boolean;
} = {}) {
  const { settings } = useSettings();
  const [stage, setStage] = useState<Stage>("setup");
  const [sessionName, setSessionName] = useState("");
  const [activeSession, setActiveSession] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  const [prompts, setPrompts] = useState<StudioPrompt[]>([]);
  const [attemptsByPrompt, setAttemptsByPrompt] = useState<
    Record<string, StudioAttempt[]>
  >({});
  const [index, setIndex] = useState(0);
  const [notice, setNotice] = useState("");
  const [recorderKey, setRecorderKey] = useState(0);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [noiseFloor, setNoiseFloor] = useState(0.008);

  const playback = useRef<HTMLAudioElement | null>(null);
  const playbackToken = useRef(0);
  const [playingKey, setPlayingKey] = useState("");

  const current = prompts[index];
  const recordedCount = Object.keys(attemptsByPrompt).length;

  function stopPlayback() {
    playbackToken.current += 1;
    if (playback.current) {
      playback.current.pause();
      playback.current = null;
    }
    setPlayingKey("");
  }

  useEffect(() => stopPlayback, []);

  async function startSession(e: React.FormEvent) {
    e.preventDefault();
    const sid = sessionName.trim();
    if (!sid) return;
    setStage("loading");
    setActiveSession(sid);
    setErrorMsg("");
    try {
      const [p, rec] = await Promise.all([
        fetchStudioPrompts(collection),
        fetchStudioRecordings(sid, collection),
      ]);
      setPrompts(p.items);
      setAttemptsByPrompt(rec.attempts ?? {});
      const firstUnrecorded = p.items.findIndex(
        (it) => !(rec.attempts ?? {})[it.prompt_id]?.length
      );
      setIndex(firstUnrecorded >= 0 ? firstUnrecorded : 0);
      setStage("calibrate");
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Yükleme başarısız");
      setStage("error");
    }
  }

  function onCalibrated(media: MediaStream, floor: number) {
    setStream(media);
    setNoiseFloor(floor);
    setStage("ready");
  }

  const goto = useCallback(
    (next: number) => {
      stopPlayback();
      setNotice("");
      setIndex((prev) => {
        if (next < 0 || next >= prompts.length) return prev;
        return next;
      });
      setRecorderKey((k) => k + 1);
    },
    [prompts.length]
  );

  function reset() {
    stopPlayback();
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    setStage("setup");
    setPrompts([]);
    setAttemptsByPrompt({});
    setIndex(0);
    setNotice("");
    setRecorderKey(0);
  }

  function onSaved(promptId: string, attempt: StudioAttempt) {
    setAttemptsByPrompt((prev) => ({
      ...prev,
      [promptId]: [...(prev[promptId] ?? []), attempt],
    }));
    setNotice(`Deneme #${attempt.attempt} kaydedildi.`);
  }

  function onRejected(reason: string) {
    setNotice(`Kayıt kabul edilmedi (${reason}) — tekrar deneyin.`);
    setRecorderKey((k) => k + 1);
  }

  async function playAttempt(promptId: string, attempt: number) {
    stopPlayback();
    const token = ++playbackToken.current;
    setPlayingKey(`${promptId}:${attempt}`);
    const audio = new Audio(
      studioAudioUrl(activeSession, promptId, attempt, collection)
    );
    playback.current = audio;
    await new Promise<void>((resolve) => {
      audio.onended = () => resolve();
      audio.onerror = () => resolve();
      audio.play().catch(() => resolve());
    });
    if (playbackToken.current === token) {
      setPlayingKey("");
      playback.current = null;
    }
  }

  async function removeAttempt(promptId: string, attempt: number) {
    stopPlayback();
    try {
      await deleteStudioAttempt(activeSession, promptId, attempt, collection);
    } catch (err) {
      setNotice(`Silinemedi: ${err instanceof Error ? err.message : "hata"}`);
      return;
    }
    setAttemptsByPrompt((prev) => {
      const remaining = (prev[promptId] ?? []).filter(
        (a) => a.attempt !== attempt
      );
      const next = { ...prev };
      if (remaining.length) next[promptId] = remaining;
      else delete next[promptId];
      return next;
    });
    setNotice(`a${String(attempt).padStart(2, "0")} silindi.`);
  }

  // ---- render ----

  if (stage === "setup") {
    return (
      <div className="mx-auto w-full max-w-[460px] pt-10">
        <div className="card stack">
          <Mic className="text-brand" size={24} />
          <h2 className="title mt-4">{title ?? "Doğrulama Kaydı"}</h2>
          <p className="lede">
            {intro ??
              "Prompt'lar sırayla gösterilir. Kırmızı olanlar kasıtlı hata ister. Her kayıttan sonra ne kaydettiğini dinlersin. Değerlendirme yapılmaz — yalnızca kayıt toplanır."}
          </p>
          <form onSubmit={startSession} className="mt-5 space-y-3">
            <div>
              <label className="eyebrow">Oturum adı</label>
              <input
                className="field mt-1 w-full"
                value={sessionName}
                onChange={(e) => setSessionName(e.target.value)}
                placeholder="örn. onur_pilot_1"
                autoFocus
              />
            </div>
            <button
              className="btn-primary"
              type="submit"
              disabled={!sessionName.trim()}
            >
              <ArrowRight size={16} /> Başla
            </button>
          </form>
        </div>
      </div>
    );
  }

  if (stage === "loading") {
    return (
      <div className="flex h-[50vh] items-center justify-center text-muted">
        <Loader2 className="animate-spin" size={28} />
      </div>
    );
  }

  if (stage === "error") {
    return (
      <div className="mx-auto w-full max-w-[460px] pt-10">
        <div className="card stack">
          <AlertTriangle className="text-bad" size={24} />
          <h2 className="title mt-4">Hata</h2>
          <p className="note mt-2">{errorMsg}</p>
          <button className="btn-primary mt-4" onClick={() => setStage("setup")}>
            <RotateCcw size={16} /> Başa dön
          </button>
        </div>
      </div>
    );
  }

  if (stage === "calibrate") {
    return (
      <div className="mx-auto w-full max-w-[460px] pt-6">
        <Calibrate onReady={onCalibrated} manual={manual} />
      </div>
    );
  }

  if (!current) {
    return (
      <div className="flex h-[50vh] items-center justify-center text-muted">
        Prompt bulunamadı.
      </div>
    );
  }

  const isErr = isErrorPrompt(current);
  const curAttempts = attemptsByPrompt[current.prompt_id] ?? [];
  const progress = prompts.length
    ? Math.round((recordedCount / prompts.length) * 100)
    : 0;

  return (
    <div className="mx-auto w-full max-w-5xl">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="font-mono text-[12px] text-muted">
          Oturum <span className="text-ink">{activeSession}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[12px] text-muted">
            {recordedCount} / {prompts.length} kayıtlı
          </span>
          <button className="btn-sm" onClick={reset}>
            Bitir
          </button>
        </div>
      </div>
      <div className="progress-track mb-5">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* Prompt list */}
        <div className="order-2 flex max-h-[64vh] w-full shrink-0 flex-col gap-1.5 overflow-y-auto lg:order-1 lg:w-64">
          {prompts.map((p, i) => {
            const done = (attemptsByPrompt[p.prompt_id] ?? []).length;
            const err = isErrorPrompt(p);
            return (
              <button
                key={p.prompt_id}
                onClick={() => goto(i)}
                className={clsx(
                  "flex items-center justify-between gap-2 rounded-lg border px-3 py-2 text-left text-[13px] transition",
                  i === index
                    ? "border-brand bg-brand-soft text-brand"
                    : done
                      ? "border-ok/25 bg-ok-soft text-ok hover:border-ok/40"
                      : "border-line bg-surface text-muted hover:border-brand/40 hover:text-ink"
                )}
              >
                <span className="min-w-0">
                  <span className="block truncate font-medium">
                    <span className="mr-1.5 font-mono text-[10px] opacity-50">
                      {i + 1}
                    </span>
                    {p.display_reading_text || p.target_text}
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-1">
                  {done > 0 && (
                    <span className="font-mono text-[10px]">×{done}</span>
                  )}
                  <span
                    className={clsx(
                      "rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-widest",
                      err
                        ? "bg-bad-soft text-bad"
                        : "bg-canvas text-muted"
                    )}
                  >
                    {err ? "ERR" : "NAT"}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        {/* Main */}
        <div className="order-1 min-w-0 flex-1 lg:order-2">
          <div className="card stack">
            <div className="flex items-center justify-between">
              <span className="eyebrow">
                {current.prompt_type} · {current.difficulty || "—"}
              </span>
              <span
                className={clsx(
                  "rounded-full px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-widest",
                  isErr ? "bg-bad-soft text-bad" : "bg-ok-soft text-ok"
                )}
              >
                {isErr ? "Kasıtlı hata" : "Doğal okuma"}
              </span>
            </div>

            <h2
              className={clsx(
                "prompt-word mt-3",
                isErr && "text-bad"
              )}
            >
              {current.display_reading_text || current.target_text}
            </h2>

            <div className="ipa-chip">
              <span>
                {isErr && current.intended_wrong_ipa
                  ? current.intended_wrong_ipa
                  : current.canonical_ipa}
              </span>
            </div>

            {isErr && (
              <div className="mt-4 rounded-lg border border-bad/25 bg-bad-soft px-4 py-3 text-[14px] text-bad">
                <strong>Kasıtlı hata isteniyor:</strong>{" "}
                {current.speaker_instruction_tr &&
                current.speaker_instruction_tr !== "none"
                  ? current.speaker_instruction_tr
                  : speakerInstructionForRule(current.error_rule_id) ||
                    current.intended_error}
              </div>
            )}

            {!isErr &&
              current.speaker_instruction_tr &&
              current.speaker_instruction_tr !== "none" && (
                <div className="mt-4 rounded-lg border border-line bg-canvas px-4 py-3 text-[14px] text-muted">
                  <strong className="text-ink">Not:</strong>{" "}
                  {current.speaker_instruction_tr}
                </div>
              )}

            {current.error_explanation_tr &&
              current.error_explanation_tr !== "none" && (
                <p className="mt-3 text-[13px] text-muted">
                  <strong>Açıklama:</strong> {current.error_explanation_tr}
                </p>
              )}

            {notice && (
              <div className="mt-4 rounded-lg border border-warn/30 bg-warn-soft px-4 py-2 text-[13px] text-warn">
                {notice}
              </div>
            )}

            {curAttempts.length > 0 && (
              <div className="mt-4 flex flex-wrap gap-1.5">
                {curAttempts.map((a) => {
                  const key = `${current.prompt_id}:${a.attempt}`;
                  return (
                    <AttemptChip
                      key={a.attempt}
                      attempt={a}
                      playing={playingKey === key}
                      onPlay={() => playAttempt(current.prompt_id, a.attempt)}
                      onDelete={() =>
                        removeAttempt(current.prompt_id, a.attempt)
                      }
                    />
                  );
                })}
              </div>
            )}

            {manual ? (
              <PushRecorder
                key={`${current.prompt_id}-${recorderKey}`}
                stream={stream}
                sessionId={activeSession}
                promptId={current.prompt_id}
                collection={collection}
                onSaved={(a) => onSaved(current.prompt_id, a)}
                onRejected={onRejected}
                onPlaybackDone={() => {
                  if (index + 1 < prompts.length) goto(index + 1);
                }}
              />
            ) : (
              <Recorder
                key={`${current.prompt_id}-${recorderKey}`}
                stream={stream}
                noiseFloor={noiseFloor}
                silenceMs={settings.silenceMs}
                sessionId={activeSession}
                promptId={current.prompt_id}
                collection={collection}
                onSaved={(a) => onSaved(current.prompt_id, a)}
                onRejected={onRejected}
                onPlaybackDone={() => {
                  if (index + 1 < prompts.length) goto(index + 1);
                }}
              />
            )}

            <div className="mt-5 flex items-center justify-between">
              <button
                className="btn-ghost"
                onClick={() => goto(index - 1)}
                disabled={index === 0}
              >
                <ArrowLeft size={16} /> Önceki
              </button>
              <button
                className="btn-ghost"
                onClick={() => goto(index + 1)}
                disabled={index + 1 >= prompts.length}
              >
                Sonraki <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---- Recorder ----

function Recorder({
  stream,
  noiseFloor,
  silenceMs,
  sessionId,
  promptId,
  collection,
  onSaved,
  onRejected,
  onPlaybackDone,
}: {
  stream: MediaStream | null;
  noiseFloor: number;
  silenceMs: number;
  sessionId: string;
  promptId: string;
  collection: StudioCollection;
  onSaved: (attempt: StudioAttempt) => void;
  onRejected: (reason: string) => void;
  onPlaybackDone: () => void;
}) {
  const [phase, setPhase] = useState<VadPhase | "uploading" | "review">(
    "armed"
  );
  const [levels, setLevels] = useState<number[]>(() =>
    new Array(WAVE_BARS).fill(0)
  );
  const [errMsg, setErrMsg] = useState("");
  const handleRef = useRef<VadHandle | null>(null);
  const frame = useRef(0);
  const reviewAudio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (!stream) return;
    const handle = startVad({
      stream,
      noiseFloor,
      silenceMs,
      minSpeechMs: 300,
      maxRecordMs: 12000,
      onPhase: (p) => setPhase(p),
      onLevel: (lvl) => {
        frame.current += 1;
        if (frame.current % 2 !== 0) return;
        setLevels((prev) => {
          const next = prev.slice(1);
          next.push(lvl);
          return next;
        });
      },
      onResult: async (blob, durationMs) => {
        setPhase("uploading");
        setErrMsg("");
        const res = await uploadStudioRecording({
          sessionId,
          promptId,
          audio: blob,
          duration_ms: durationMs,
          collection,
        });
        if (!res.ok) {
          if (res.quality) {
            onRejected(res.quality.reason ?? "geçersiz");
          } else {
            setErrMsg(res.error ?? "Kayıt yüklenemedi");
            setPhase("armed");
          }
          return;
        }
        onSaved({
          attempt: res.data.attempt ?? 0,
          filename: res.data.recording_filename,
          duration_ms: res.data.quality.duration_ms,
        });
        // Play back what was just recorded, then advance.
        setPhase("review");
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        reviewAudio.current = audio;
        audio.onended = () => {
          URL.revokeObjectURL(url);
          onPlaybackDone();
        };
        audio.onerror = () => {
          URL.revokeObjectURL(url);
          onPlaybackDone();
        };
        audio.play().catch(() => {
          URL.revokeObjectURL(url);
          onPlaybackDone();
        });
      },
    });
    handleRef.current = handle;
    return () => {
      handle.cancel();
      if (reviewAudio.current) {
        reviewAudio.current.pause();
        reviewAudio.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const recording = phase === "recording";
  const busy = phase === "finalizing" || phase === "uploading";
  const reviewing = phase === "review";

  return (
    <div className="mt-6 flex flex-col items-center">
      {errMsg && (
        <div className="mb-4 w-full rounded-lg bg-bad/10 px-4 py-3 text-center text-sm text-bad">
          {errMsg}
        </div>
      )}

      <div className={clsx("mic-orb", recording && "mic-orb-recording")}>
        {(recording || phase === "armed") && (
          <span
            className={clsx(
              "absolute inset-0 rounded-full animate-pulse-ring",
              recording ? "bg-bad/30" : "bg-brand/25"
            )}
          />
        )}
        {busy ? (
          <Loader2 size={32} className="animate-spin" />
        ) : reviewing ? (
          <Play size={32} />
        ) : (
          <Mic size={34} />
        )}
      </div>

      <div
        className="mt-7 flex h-16 items-end justify-center gap-[3px]"
        aria-hidden
      >
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
          recording
            ? "status-recording"
            : busy || reviewing
              ? "status-analyzing"
              : "status-armed"
        )}
      >
        {busy ? (
          "Kaydediliyor"
        ) : reviewing ? (
          <>
            <Play size={12} /> Dinle — kaydını kontrol et
          </>
        ) : recording ? (
          <>
            <span className="h-2 w-2 rounded-full bg-bad" /> Kaydediliyor
          </>
        ) : (
          <>
            <span className="h-2 w-2 rounded-full bg-brand" /> Dinleniyor —
            konuşmaya başla
          </>
        )}
      </div>

      <div className="mt-6 h-9">
        {recording && (
          <button
            className="btn-ghost"
            onClick={() => handleRef.current?.finishNow()}
          >
            <Check size={16} /> Bitirdim
          </button>
        )}
      </div>
    </div>
  );
}

// ---- PushRecorder (press-and-hold, no silence detection) ----

function PushRecorder({
  stream,
  sessionId,
  promptId,
  collection,
  onSaved,
  onRejected,
  onPlaybackDone,
}: {
  stream: MediaStream | null;
  sessionId: string;
  promptId: string;
  collection: StudioCollection;
  onSaved: (attempt: StudioAttempt) => void;
  onRejected: (reason: string) => void;
  onPlaybackDone: () => void;
}) {
  const [phase, setPhase] = useState<
    "idle" | "recording" | "uploading" | "review"
  >("idle");
  const [levels, setLevels] = useState<number[]>(() =>
    new Array(WAVE_BARS).fill(0)
  );
  const [errMsg, setErrMsg] = useState("");
  const handleRef = useRef<PushRecorderHandle | null>(null);
  const holding = useRef(false);
  const frame = useRef(0);
  const reviewAudio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    return () => {
      handleRef.current?.cancel();
      if (reviewAudio.current) {
        reviewAudio.current.pause();
        reviewAudio.current = null;
      }
    };
  }, []);

  async function submit(blob: Blob, durationMs: number) {
    setPhase("uploading");
    setErrMsg("");
    const res = await uploadStudioRecording({
      sessionId,
      promptId,
      audio: blob,
      duration_ms: durationMs,
      collection,
    });
    if (!res.ok) {
      if (res.quality) {
        onRejected(res.quality.reason ?? "geçersiz");
      } else {
        setErrMsg(res.error ?? "Kayıt yüklenemedi");
        setPhase("idle");
      }
      return;
    }
    onSaved({
      attempt: res.data.attempt ?? 0,
      filename: res.data.recording_filename,
      duration_ms: res.data.quality.duration_ms,
    });
    setPhase("review");
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    reviewAudio.current = audio;
    const finish = () => {
      URL.revokeObjectURL(url);
      onPlaybackDone();
    };
    audio.onended = finish;
    audio.onerror = finish;
    audio.play().catch(finish);
  }

  function startHold() {
    if (!stream || holding.current || phase === "uploading" || phase === "review")
      return;
    holding.current = true;
    setLevels(new Array(WAVE_BARS).fill(0));
    setPhase("recording");
    handleRef.current = startPushToTalk({
      stream,
      maxRecordMs: 15000,
      onLevel: (lvl) => {
        frame.current += 1;
        if (frame.current % 2 !== 0) return;
        setLevels((prev) => {
          const next = prev.slice(1);
          next.push(lvl);
          return next;
        });
      },
      onResult: (blob, durationMs) => {
        void submit(blob, durationMs);
      },
    });
  }

  function endHold() {
    if (!holding.current) return;
    holding.current = false;
    handleRef.current?.stop();
    handleRef.current = null;
  }

  const recording = phase === "recording";
  const busy = phase === "uploading";
  const reviewing = phase === "review";

  return (
    <div className="mt-6 flex flex-col items-center">
      {errMsg && (
        <div className="mb-4 w-full rounded-lg bg-bad/10 px-4 py-3 text-center text-sm text-bad">
          {errMsg}
        </div>
      )}

      <button
        type="button"
        className={clsx(
          "mic-orb touch-none select-none",
          recording && "mic-orb-recording"
        )}
        disabled={busy || reviewing || !stream}
        onPointerDown={(e) => {
          e.preventDefault();
          e.currentTarget.setPointerCapture(e.pointerId);
          startHold();
        }}
        onPointerUp={endHold}
        onPointerCancel={endHold}
        onKeyDown={(e) => {
          if ((e.key === " " || e.key === "Enter") && !e.repeat) {
            e.preventDefault();
            startHold();
          }
        }}
        onKeyUp={(e) => {
          if (e.key === " " || e.key === "Enter") {
            e.preventDefault();
            endHold();
          }
        }}
        onContextMenu={(e) => e.preventDefault()}
        aria-label="Bas ve konuş"
      >
        {(recording || phase === "idle") && (
          <span
            className={clsx(
              "absolute inset-0 rounded-full",
              recording ? "animate-pulse-ring bg-bad/30" : "bg-brand/15"
            )}
          />
        )}
        {busy ? (
          <Loader2 size={32} className="animate-spin" />
        ) : reviewing ? (
          <Play size={32} />
        ) : (
          <Mic size={34} />
        )}
      </button>

      <div
        className="mt-7 flex h-16 items-end justify-center gap-[3px]"
        aria-hidden
      >
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
          recording
            ? "status-recording"
            : busy || reviewing
              ? "status-analyzing"
              : "status-armed"
        )}
      >
        {busy ? (
          "Kaydediliyor"
        ) : reviewing ? (
          <>
            <Play size={12} /> Dinle — kaydını kontrol et
          </>
        ) : recording ? (
          <>
            <span className="h-2 w-2 rounded-full bg-bad" /> Bırakınca biter
          </>
        ) : (
          <>
            <Mic size={12} /> Bas ve konuş — basılı tut
          </>
        )}
      </div>
    </div>
  );
}

// ---- AttemptChip ----

function AttemptChip({
  attempt,
  playing,
  onPlay,
  onDelete,
}: {
  attempt: StudioAttempt;
  playing: boolean;
  onPlay: () => void;
  onDelete: () => void;
}) {
  const label = `a${String(attempt.attempt).padStart(2, "0")}`;
  const dur =
    attempt.duration_ms && attempt.duration_ms > 0
      ? `${(attempt.duration_ms / 1000).toFixed(1)}s`
      : "--";
  return (
    <span
      className={clsx(
        "group/chip flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] transition",
        playing
          ? "border-brand bg-brand-soft text-brand"
          : "border-line bg-surface text-muted hover:border-brand/40"
      )}
    >
      <button
        type="button"
        className="flex items-center gap-1.5"
        onClick={onPlay}
        title="Çal"
      >
        <Play size={12} />
        {label}
        <span className="opacity-60">{dur}</span>
      </button>
      <button
        type="button"
        className="text-muted opacity-0 transition group-hover/chip:opacity-100 hover:text-bad"
        onClick={onDelete}
        title="Sil"
      >
        <Trash2 size={12} />
      </button>
    </span>
  );
}

// ---- Calibrate (mirrors Learner mic check) ----

function Calibrate({
  onReady,
  manual = false,
}: {
  onReady: (stream: MediaStream, floor: number) => void;
  manual?: boolean;
}) {
  const [phase, setPhase] = useState<
    "idle" | "requesting" | "measuring" | "error"
  >("idle");
  const [level, setLevel] = useState(0);

  async function begin() {
    setPhase("requesting");
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
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
    <div className="card stack">
      <Mic className="text-brand" size={24} />
      <h2 className="title mt-4">Mikrofon kontrolü</h2>
      <p className="lede">
        {manual
          ? "Her prompt'ta mikrofon tuşuna basılı tutarak konuş; bıraktığında kayıt biter ve sana dinletilir."
          : "Kayıt otomatik: prompt ekrana gelince konuşmaya başla, sustuğunda kendiliğinden durur ve kaydını sana dinletir."}
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
            <>
              <Loader2 size={16} className="animate-spin" /> Hazırlanıyor
            </>
          ) : phase === "error" ? (
            <>
              <RotateCcw size={16} /> Tekrar dene
            </>
          ) : (
            <>
              <Mic size={16} /> Mikrofonu aç ve başla
            </>
          )}
        </button>
      </div>
    </div>
  );
}
