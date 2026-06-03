// Voice activity detection — pre-roll recorder with automatic silence cutoff.
// Adapted from user-app/src/vad.ts (no logic changes).

export type VadPhase = "armed" | "recording" | "finalizing";

export interface VadOptions {
  stream: MediaStream;
  silenceMs?: number;
  minSpeechMs?: number;
  maxRecordMs: number;
  startDebounceMs?: number;
  noiseFloor?: number;
  onPhase: (phase: VadPhase) => void;
  onLevel: (level: number) => void;
  onResult: (blob: Blob, durationMs: number, mime: string) => void;
}

export interface VadHandle {
  finishNow: () => void;
  cancel: () => void;
}

type AudioContextCtor = typeof AudioContext;

function resolveAudioContext(): AudioContextCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    AudioContext?: AudioContextCtor;
    webkitAudioContext?: AudioContextCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

export function startVad(opts: VadOptions): VadHandle {
  const silenceMs = opts.silenceMs ?? 1200;
  const minSpeechMs = opts.minSpeechMs ?? 350;
  const startDebounceMs = opts.startDebounceMs ?? 130;
  const threshold = Math.max(0.018, (opts.noiseFloor ?? 0.008) * 2.4);

  const Ctx = resolveAudioContext();
  const audioCtx = Ctx ? new Ctx() : null;
  const analyser = audioCtx ? audioCtx.createAnalyser() : null;
  let source: MediaStreamAudioSourceNode | null = null;
  if (audioCtx && analyser) {
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.6;
    source = audioCtx.createMediaStreamSource(opts.stream);
    source.connect(analyser);
  }
  const buffer = analyser ? new Uint8Array(analyser.fftSize) : new Uint8Array(0);

  let recorder: MediaRecorder | null = null;
  const chunks: Blob[] = [];
  let raf = 0;
  let done = false;
  let speechDetected = false;
  let aboveSince = 0;
  const recordStartedAt = performance.now();
  let speechSeenAt = 0;
  let lastVoiceAt = 0;
  let smoothed = 0;

  function setPhase(next: VadPhase) {
    opts.onPhase(next);
  }

  function rms(): number {
    if (!analyser) return 0;
    analyser.getByteTimeDomainData(buffer);
    let sum = 0;
    for (let i = 0; i < buffer.length; i += 1) {
      const v = (buffer[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / buffer.length);
  }

  function startPreRollRecorder() {
    chunks.length = 0;
    recorder = new MediaRecorder(opts.stream);
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data);
    };
    recorder.onstop = () => {
      const mime = recorder?.mimeType || "audio/webm";
      const blob = new Blob(chunks, { type: mime });
      const duration = Math.max(1, Math.round(performance.now() - recordStartedAt));
      cleanup();
      setPhase("finalizing");
      opts.onResult(blob, duration, blob.type || mime);
    };
    recorder.start();
  }

  function stopRecording() {
    if (done) return;
    done = true;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    } else {
      cleanup();
    }
  }

  function cleanup() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    try {
      source?.disconnect();
      analyser?.disconnect();
    } catch {
      /* ignore */
    }
    if (audioCtx && audioCtx.state !== "closed") {
      void audioCtx.close().catch(() => undefined);
    }
  }

  function tick() {
    if (done) return;
    const now = performance.now();
    const level = rms();
    smoothed = smoothed * 0.7 + level * 0.3;
    opts.onLevel(Math.min(1, smoothed * 4.5));
    const voiced = level > threshold;

    if (!speechDetected) {
      if (voiced) {
        if (aboveSince === 0) aboveSince = now;
        if (now - aboveSince >= startDebounceMs) {
          speechDetected = true;
          speechSeenAt = now;
          lastVoiceAt = now;
          setPhase("recording");
        }
      } else {
        aboveSince = 0;
      }
    } else {
      if (voiced) lastVoiceAt = now;
      const longEnough = now - speechSeenAt >= minSpeechMs;
      const silentEnough = now - lastVoiceAt >= silenceMs;
      if (longEnough && silentEnough) {
        stopRecording();
        return;
      }
    }
    if (now - recordStartedAt >= opts.maxRecordMs) {
      stopRecording();
      return;
    }
    raf = requestAnimationFrame(tick);
  }

  if (audioCtx && audioCtx.state === "suspended") {
    void audioCtx.resume().catch(() => undefined);
  }
  setPhase("armed");
  startPreRollRecorder();
  raf = requestAnimationFrame(tick);

  return {
    finishNow() {
      if (done) return;
      if (speechDetected) {
        stopRecording();
        return;
      }
      done = true;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      try {
        if (recorder && recorder.state === "recording") {
          recorder.ondataavailable = null;
          recorder.onstop = null;
          recorder.stop();
        }
      } catch {
        /* ignore */
      }
      cleanup();
    },
    cancel() {
      if (done) return;
      done = true;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      try {
        if (recorder && recorder.state === "recording") {
          recorder.ondataavailable = null;
          recorder.onstop = null;
          recorder.stop();
        }
      } catch {
        /* ignore */
      }
      cleanup();
    },
  };
}

// Manual push-to-talk recorder: records from start() until stop()/cancel(),
// with no silence detection. Used by the demo capture page where the speaker
// holds a button to record and releases to finish.
export interface PushRecorderOptions {
  stream: MediaStream;
  maxRecordMs?: number;
  onLevel?: (level: number) => void;
  onResult: (blob: Blob, durationMs: number, mime: string) => void;
}

export interface PushRecorderHandle {
  stop: () => void;
  cancel: () => void;
}

export function startPushToTalk(opts: PushRecorderOptions): PushRecorderHandle {
  const maxRecordMs = opts.maxRecordMs ?? 15000;
  const Ctx = resolveAudioContext();
  const audioCtx = Ctx ? new Ctx() : null;
  const analyser = audioCtx ? audioCtx.createAnalyser() : null;
  let source: MediaStreamAudioSourceNode | null = null;
  if (audioCtx && analyser) {
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.6;
    source = audioCtx.createMediaStreamSource(opts.stream);
    source.connect(analyser);
  }
  const buffer = analyser ? new Uint8Array(analyser.fftSize) : new Uint8Array(0);

  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(opts.stream);
  const startedAt = performance.now();
  let raf = 0;
  let done = false;
  let emit = true;
  let smoothed = 0;

  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  recorder.onstop = () => {
    const mime = recorder.mimeType || "audio/webm";
    cleanup();
    if (!emit) return;
    const blob = new Blob(chunks, { type: mime });
    const duration = Math.max(1, Math.round(performance.now() - startedAt));
    opts.onResult(blob, duration, blob.type || mime);
  };

  function cleanup() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    try {
      source?.disconnect();
      analyser?.disconnect();
    } catch {
      /* ignore */
    }
    if (audioCtx && audioCtx.state !== "closed") {
      void audioCtx.close().catch(() => undefined);
    }
  }

  function finish() {
    if (done) return;
    done = true;
    if (recorder.state === "recording") recorder.stop();
    else cleanup();
  }

  function tick() {
    if (done) return;
    if (analyser) {
      analyser.getByteTimeDomainData(buffer);
      let sum = 0;
      for (let i = 0; i < buffer.length; i += 1) {
        const v = (buffer[i] - 128) / 128;
        sum += v * v;
      }
      smoothed = smoothed * 0.7 + Math.sqrt(sum / buffer.length) * 0.3;
      opts.onLevel?.(Math.min(1, smoothed * 4.5));
    }
    if (performance.now() - startedAt >= maxRecordMs) {
      finish();
      return;
    }
    raf = requestAnimationFrame(tick);
  }

  if (audioCtx && audioCtx.state === "suspended") {
    void audioCtx.resume().catch(() => undefined);
  }
  recorder.start();
  raf = requestAnimationFrame(tick);

  return {
    stop() {
      finish();
    },
    cancel() {
      if (done) return;
      done = true;
      emit = false;
      try {
        if (recorder.state === "recording") recorder.stop();
        else cleanup();
      } catch {
        cleanup();
      }
    },
  };
}

export function measureNoiseFloor(stream: MediaStream, durationMs = 800): Promise<number> {
  return new Promise((resolve) => {
    const Ctx = resolveAudioContext();
    if (!Ctx) {
      resolve(0.008);
      return;
    }
    const ctx = new Ctx();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyser);
    const buf = new Uint8Array(analyser.fftSize);
    const samples: number[] = [];
    const startedAt = performance.now();
    let raf = 0;

    const finish = () => {
      if (raf) cancelAnimationFrame(raf);
      try {
        source.disconnect();
        analyser.disconnect();
      } catch {
        /* ignore */
      }
      if (ctx.state !== "closed") void ctx.close().catch(() => undefined);
      samples.sort((a, b) => a - b);
      const median = samples.length ? samples[Math.floor(samples.length / 2)] : 0.008;
      resolve(Math.min(0.05, Math.max(0.002, median)));
    };

    const loop = () => {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i += 1) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      samples.push(Math.sqrt(sum / buf.length));
      if (performance.now() - startedAt >= durationMs) {
        finish();
        return;
      }
      raf = requestAnimationFrame(loop);
    };

    if (ctx.state === "suspended") void ctx.resume().catch(() => undefined);
    raf = requestAnimationFrame(loop);
  });
}
