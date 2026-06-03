import clsx from "clsx";
import { AlertTriangle } from "lucide-react";
import type {
  AssessResponse,
  CoachAnalysis,
  CoachResult,
  CoachResultPhone,
  CoachScore,
  CoachWav2Vec,
  LengthEvidence,
} from "../api";

// ---- helpers ----

export function fmt(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function scoreTone(v: number | null | undefined) {
  if (v == null) return "conf-fill-ok";
  if (v >= 0.75) return "conf-fill-ok";
  if (v >= 0.5) return "conf-fill-warn";
  return "conf-fill-bad";
}

function ConfBar({ value }: { value: number | null | undefined }) {
  const width = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="conf-track">
        <div className={scoreTone(value)} style={{ width: `${width}%` }} />
      </div>
      <span className="font-mono text-[11px] text-muted">{fmt(value)}</span>
    </div>
  );
}

function Banner({ tone, children }: { tone: "bad" | "warn"; children: React.ReactNode }) {
  const cls =
    tone === "bad"
      ? "border-bad/25 bg-bad-soft text-bad"
      : "border-warn/25 bg-warn-soft text-warn";
  return (
    <div className={clsx("flex items-center gap-2 rounded-xl border px-3 py-2.5 text-[13px]", cls)}>
      <AlertTriangle size={15} className="shrink-0" />
      {children}
    </div>
  );
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-line bg-canvas px-3 py-2">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 font-mono text-[13px] font-medium text-ink">{value}</div>
    </div>
  );
}

// ---- status tone ----

const RESULT_STATUS_TONE: Record<CoachResultPhone["strict_status"], string> = {
  correct: "border-ok/25 bg-ok-soft text-ok",
  incorrect: "border-bad/25 bg-bad-soft text-bad",
  missing: "border-warn/25 bg-warn-soft text-warn",
  extra: "border-extra/25 bg-extra-soft text-extra",
};

// ---- score_reason humanization ----

const SCORE_REASON_LABEL: Record<string, string> = {
  exact: "tam",
  analysis_corrected: "analizle düzeltildi",
  phonetic_distance: "fonetik mesafe",
  duration_override: "süre düzeltmesi",
  duration_verified: "süre doğrulandı",
  missing: "eksik",
  extra_penalty: "fazla",
  extra: "fazla",
};

function scoreReasonLabel(reason: string | null | undefined): string {
  if (!reason) return "—";
  return SCORE_REASON_LABEL[reason] ?? reason;
}

// ---- acoustic analysis override ----

type AnalysisOverride = {
  from: string | null;
  to: string | null;
  source: string | null;
  changeId: string | null;
};

function analysisOverride(raw: unknown): AnalysisOverride | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const asStr = (v: unknown): string | null => (typeof v === "string" && v ? v : null);
  const from = asStr(obj.from);
  const to = asStr(obj.to);
  const source = asStr(obj.source);
  const changeId = asStr(obj.change_id);
  if (!from && !to && !source && !changeId) return null;
  return { from, to, source, changeId };
}

function overrideText(ov: AnalysisOverride): string {
  const from = ov.from ?? "?";
  const to = ov.to ?? "?";
  const src = ov.source ? ` (kaynak: ${ov.source})` : "";
  return `model /${from}/ duydu → akustik analiz /${to}/'ya düzeltti${src}`;
}

// ---- duration / length modifier evidence ----

function durationModifierText(features: Record<string, unknown> | undefined): string | null {
  if (!features) return null;
  const mods = features.modifiers;
  if (!mods || typeof mods !== "object") return null;
  const parts: string[] = [];
  for (const [name, payload] of Object.entries(mods as Record<string, unknown>)) {
    if (payload == null) continue;
    if (typeof payload === "number") {
      parts.push(`${name}: ${payload.toFixed(2)}`);
    } else if (typeof payload === "object") {
      const obj = payload as Record<string, unknown>;
      const nudge = obj.nudge ?? obj.delta ?? obj.value ?? obj.score;
      if (typeof nudge === "number") parts.push(`${name}: ${nudge.toFixed(2)}`);
      else parts.push(name);
    } else {
      parts.push(`${name}: ${String(payload)}`);
    }
  }
  return parts.length ? `Süre kanıtı — ${parts.join(" · ")}` : null;
}

// Length is an independent, non-gating channel (it never folds into quality).
const LENGTH_NOTE_LABEL: Record<string, string> = {
  short: "kısa",
  long: "uzun",
  expected: "normal",
};

function lengthNoteLabel(length: LengthEvidence | null | undefined): string | null {
  if (!length || !length.note) return null;
  return LENGTH_NOTE_LABEL[length.note] ?? length.note;
}

function lengthTooltip(length: LengthEvidence | null | undefined): string | undefined {
  if (!length) return undefined;
  const parts: string[] = [];
  const label = lengthNoteLabel(length);
  if (label) parts.push(`süre: ${label}`);
  if (typeof length.duration_ms === "number") parts.push(`${Math.round(length.duration_ms)} ms`);
  if (typeof length.score === "number") parts.push(`tipiklik ${pct(length.score)}`);
  if (typeof length.z === "number") parts.push(`z=${length.z.toFixed(2)}`);
  return parts.length ? parts.join(" · ") : undefined;
}

// ---- main component ----

export function AssessResultView({
  resp,
  showTechnical = false,
}: {
  resp: AssessResponse | null;
  showTechnical?: boolean;
}) {
  if (!resp) {
    return (
      <div className="card">
        <div className="eyebrow">Sonuç</div>
        <p className="mt-4 text-center text-[13px] text-muted">
          Model seç, G2P çalıştır, kaydet ve analiz et.
        </p>
      </div>
    );
  }

  const rq = resp.recording_quality;

  if (resp.status === "invalid") {
    return (
      <div className="card space-y-3">
        <div className="eyebrow">Sonuç</div>
        <Banner tone="bad">Kayıt geçersiz: {rq?.reason ?? "bilinmiyor"}</Banner>
        <QualitySection rq={rq} />
      </div>
    );
  }

  if (resp.status === "unknown_model") {
    return (
      <div className="card space-y-3">
        <div className="eyebrow">Sonuç</div>
        <Banner tone="bad">Bilinmeyen model kimliği.</Banner>
      </div>
    );
  }

  if (resp.status === "model_unavailable") {
    return (
      <div className="card space-y-3">
        <div className="eyebrow">Sonuç</div>
        <Banner tone="bad">Model artefaktı bulunamadı (checkpoint mevcut değil).</Banner>
      </div>
    );
  }

  if (resp.status === "failed" || !resp.result) {
    return (
      <div className="card space-y-3">
        <div className="eyebrow">Sonuç</div>
        <Banner tone="bad">Analiz başarısız: {resp.error_code ?? "bilinmiyor"}</Banner>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fade-up">
      <ModelOutputSection wav2vec={resp.wav2vec} analysis={resp.analysis} />
      <PhoneScoreSection result={resp.result} showTechnical={showTechnical} />
      <ScoringSection score={resp.score ?? resp.result.score} summary={resp.result.summary} />
      {showTechnical && (
        <DiagnosticsSection
          debug={resp.debug}
          wav2vec={resp.wav2vec}
          analysis={resp.analysis}
        />
      )}
      <QualitySection rq={rq} />
    </div>
  );
}

// ---- sections ----

function ModelOutputSection({
  wav2vec,
  analysis,
}: {
  wav2vec: CoachWav2Vec;
  analysis: CoachAnalysis;
}) {
  const confValues = wav2vec.phones
    .map((p) => p.confidence)
    .filter((c): c is number => c !== null);
  const meanConf = confValues.length
    ? confValues.reduce((a, b) => a + b, 0) / confValues.length
    : null;

  return (
    <div className="card">
      <div className="eyebrow mb-3">Model çıktısı</div>
      <div className="grid grid-cols-2 gap-2 mb-4 sm:grid-cols-3">
        <MetricChip label="Model" value={wav2vec.model} />
        <MetricChip label="Ham IPA" value={wav2vec.ipa || "—"} />
        <MetricChip label="Analiz IPA" value={analysis.ipa || "—"} />
        <MetricChip label="Ort. güven" value={fmt(meanConf)} />
      </div>
      <div className="rounded-xl border border-line overflow-x-auto">
        <div className="min-w-[400px]">
          <div className="data-row grid-cols-[1fr_80px_80px_120px] bg-canvas font-mono text-[10px] uppercase tracking-widest text-muted">
            <span>fonem</span><span>başla</span><span>bitiş</span><span>güven</span>
          </div>
          {wav2vec.phones.map((p, i) => (
            <div key={`${p.ipa}-${i}`} className="data-row grid-cols-[1fr_80px_80px_120px]">
              <span className="font-mono text-[13px] font-medium text-ink">{p.ipa}</span>
              <span className="font-mono text-[11px] text-muted">{p.start_ms}ms</span>
              <span className="font-mono text-[11px] text-muted">{p.end_ms}ms</span>
              <ConfBar value={p.confidence} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PhoneScoreSection({
  result,
  showTechnical,
}: {
  result: CoachResult;
  showTechnical: boolean;
}) {
  const cols = showTechnical
    ? "grid-cols-[64px_64px_96px_110px_72px_140px_120px_1fr]"
    : "grid-cols-[72px_72px_100px_110px_72px_140px_1fr]";
  return (
    <div className="card">
      <div className="eyebrow mb-3">Fonem skoru</div>
      <div className="rounded-xl border border-line overflow-x-auto">
        <div className={showTechnical ? "min-w-[840px]" : "min-w-[700px]"}>
          <div className={clsx("data-row bg-canvas font-mono text-[10px] uppercase tracking-widest text-muted", cols)}>
            <span>beklenen</span><span>gözlem</span><span>durum</span><span>skor</span><span>kalite</span><span>neden</span>
            {showTechnical && <span>ham → analiz</span>}
            <span>özellik</span>
          </div>
          {result.phones.map((phone) => {
            const override = analysisOverride(phone.changed_by_analysis);
            const durationNote = durationModifierText(phone.distance_features);
            const reasonTitle = [
              durationNote,
              override ? overrideText(override) : null,
            ]
              .filter(Boolean)
              .join("\n") || undefined;
            return (
              <div
                key={`${phone.index}-${phone.expected ?? "extra"}-${phone.observed ?? "missing"}`}
                className={clsx("data-row", cols)}
              >
                <span className="font-mono text-[13px] font-medium text-ink">
                  {phone.expected ?? "—"}
                </span>
                <span
                  className={clsx(
                    "font-mono text-[13px]",
                    phone.observed && phone.observed !== phone.expected ? "text-bad" : "text-muted"
                  )}
                >
                  {phone.observed ?? "—"}
                </span>
                <span className="flex min-w-0 items-center gap-1">
                  <span
                    className={clsx(
                      "inline-flex items-center rounded-full border px-2 py-0.5 font-mono text-[10px]",
                      RESULT_STATUS_TONE[phone.strict_status]
                    )}
                  >
                    {phone.strict_status}
                  </span>
                  {override && (
                    <span
                      className="inline-flex shrink-0 items-center rounded-full border border-extra/25 bg-extra-soft px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-extra"
                      title={overrideText(override)}
                    >
                      analiz
                    </span>
                  )}
                </span>
                <ConfBar value={phone.score} />
                <span className="flex min-w-0 flex-col items-start gap-0.5 font-mono text-[11px] text-muted">
                  <span title="Atlas/GMM telaffuz kalitesi (süre kanalından bağımsız)">
                    {phone.quality == null ? "—" : pct(phone.quality)}
                  </span>
                  {phone.length?.note && phone.length.note !== "expected" && (
                    <span
                      className="inline-flex items-center rounded-full border border-warn/25 bg-warn-soft px-1.5 py-0 text-[9px] text-warn"
                      title={lengthTooltip(phone.length)}
                    >
                      {lengthNoteLabel(phone.length)}
                    </span>
                  )}
                </span>
                <span
                  className="min-w-0 truncate font-mono text-[11px] text-muted"
                  title={reasonTitle}
                >
                  {scoreReasonLabel(phone.score_reason)}
                  {durationNote && <span className="text-extra"> · süre</span>}
                </span>
                {showTechnical && (
                  <span className="min-w-0 truncate font-mono text-[10px] text-muted">
                    {phone.raw_observed ?? "—"} → {phone.analysis_observed ?? "—"}
                  </span>
                )}
                <span
                  className="min-w-0 truncate font-mono text-[10px] text-muted"
                  title={JSON.stringify(phone.distance_features ?? {})}
                >
                  {featureSummary(phone.distance_features)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-2">
        <MetricChip label="Hedef fonem" value={String(result.summary.target_count)} />
        <MetricChip label="Eksik" value={String(result.summary.missing_count)} />
        <MetricChip label="Fazla" value={String(result.summary.extra_count)} />
      </div>

      {showTechnical && (
        <details className="mt-3 rounded-xl border border-line bg-canvas px-3 py-2">
          <summary className="cursor-pointer font-mono text-[11px] text-muted">raw phone JSON</summary>
          <pre className="mt-2 max-h-[360px] overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-relaxed text-muted">
            {JSON.stringify(result.phones, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

function featureSummary(features: Record<string, unknown> | undefined): string {
  if (!features) return "—";
  if (features.exact === true) return "exact";
  const parts: string[] = [];
  for (const [name, payload] of Object.entries(features)) {
    if (name === "exact") continue;
    if (payload && typeof payload === "object" && "match" in payload) {
      const match = (payload as { match?: unknown }).match;
      parts.push(`${name}:${match ? "ok" : "diff"}`);
    }
  }
  return parts.length ? parts.join(" · ") : "—";
}

function ScoringSection({
  score,
  summary,
}: {
  score: CoachScore;
  summary: CoachResult["summary"];
}) {
  return (
    <div className="card space-y-4">
      <div className="eyebrow">Skorlama</div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <MetricChip label="Skor" value={`${fmt(score.item_score)} / ${fmt(score.max_item_score)}`} />
        <MetricChip label="Kelime" value={pct(score.word_score)} />
        <MetricChip label="Fonem ort." value={pct(score.phoneme_core)} />
        <MetricChip label="Kalite" value={pct(summary.quality_score)} />
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <MetricChip label="Doğru" value={String(summary.correct_count)} />
        <MetricChip label="Yanlış" value={String(summary.incorrect_count)} />
        <MetricChip label="Eksik" value={String(summary.missing_count)} />
        <MetricChip label="Fazla cezası" value={pct(score.extra_penalty)} />
      </div>
    </div>
  );
}

function QualitySection({ rq }: { rq: AssessResponse["recording_quality"] }) {
  if (!rq) return null;
  return (
    <div className="card">
      <div className="eyebrow mb-3">Kayıt kalitesi</div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <MetricChip label="Durum" value={rq.status} />
        <MetricChip label="Süre" value={`${rq.duration_ms} ms`} />
        <MetricChip label="Örnek hızı" value={rq.sample_rate ? `${rq.sample_rate} Hz` : "—"} />
        <MetricChip label="RMS" value={fmt(rq.rms, 1)} />
      </div>
      {rq.reason && (
        <p className="mt-2 text-[12px] text-muted">{rq.reason}</p>
      )}
    </div>
  );
}

function DiagnosticsSection({
  debug,
  wav2vec,
  analysis,
}: {
  debug: AssessResponse["debug"];
  wav2vec: CoachWav2Vec;
  analysis: CoachAnalysis;
}) {
  const posteriorCount = Array.isArray(wav2vec.posterior_candidates)
    ? wav2vec.posterior_candidates.length
    : 0;
  const analysisCandidateCount =
    debug && Array.isArray(debug.analysis_candidates)
      ? debug.analysis_candidates.length
      : 0;
  return (
    <div className="card space-y-4">
      <div className="eyebrow">Teknik döküm</div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <MetricChip label="Ham fonem" value={String(wav2vec.phones.length)} />
        <MetricChip label="Analiz fonem" value={String(analysis.phones.length)} />
        <MetricChip label="Posterior" value={String(posteriorCount)} />
        <MetricChip label="Aday" value={String(analysisCandidateCount)} />
      </div>

      {(wav2vec.posterior_candidates?.length ?? 0) > 0 && (
        <div className="rounded-xl border border-line overflow-x-auto">
          <div className="min-w-[500px]">
            <div className="data-row grid-cols-[52px_72px_88px_1fr] bg-canvas font-mono text-[10px] uppercase tracking-widest text-muted">
              <span>#</span><span>fonem</span><span>frame</span><span>adaylar</span>
            </div>
            {wav2vec.posterior_candidates?.map((row) => (
              <div key={`posterior-${row.index}`} className="data-row grid-cols-[52px_72px_88px_1fr]">
                <span className="font-mono text-[11px] text-muted">{row.index}</span>
                <span className="font-mono text-[13px] font-medium text-ink">{row.raw_ipa ?? "—"}</span>
                <span className="font-mono text-[11px] text-muted">{row.frame_count}</span>
                <span
                  className="min-w-0 truncate font-mono text-[10px] text-muted"
                  title={JSON.stringify(row.candidates)}
                >
                  {row.candidates.map((item) => `${item.phone}:${fmt(item.probability)}`).join(" · ") || "—"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <details className="rounded-xl border border-line bg-canvas px-3 py-2">
        <summary className="cursor-pointer font-mono text-[11px] text-muted">raw debug JSON</summary>
        <pre className="mt-2 max-h-[420px] overflow-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-relaxed text-muted">
          {JSON.stringify({ wav2vec, analysis, debug }, null, 2)}
        </pre>
      </details>
    </div>
  );
}
