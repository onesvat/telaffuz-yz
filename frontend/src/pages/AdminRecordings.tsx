import clsx from "clsx";
import { FileJson, Loader2, RefreshCw, Search, Volume2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchAdminRecordingAudio,
  fetchAdminRecordingResult,
  fetchAdminRecordings,
  postAssess,
  type AdminRecording,
  type AssessResponse,
} from "../api";
import { AssessResultView } from "../components/AssessResult";

type AdminRecordingPayload = {
  recording_id: string;
  created_at: string;
  metadata?: Record<string, unknown>;
  result?: AssessResponse;
};

const PAGE_SIZE = 100;

export function AdminRecordings({ password }: { password: string }) {
  const [items, setItems] = useState<AdminRecording[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [limit, setLimit] = useState(PAGE_SIZE);
  const [hasMore, setHasMore] = useState(true);
  const [error, setError] = useState("");
  const [wordQuery, setWordQuery] = useState("");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<AdminRecordingPayload | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [showJson, setShowJson] = useState(false);
  const [reassessing, setReassessing] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);

  async function load(nextLimit = limit, mode: "reset" | "more" = "reset") {
    if (mode === "more") {
      setLoadingMore(true);
    } else {
      setLoading(true);
    }
    setError("");
    try {
      const res = await fetchAdminRecordings(password, {
        limit: nextLimit,
        word: wordQuery,
        source: sourceFilter,
        status: statusFilter,
      });
      setItems(res.items);
      setLimit(nextLimit);
      setHasMore(res.items.length >= nextLimit);
      if (selectedId && !res.items.some((item) => item.recording_id === selectedId)) {
        setSelectedId("");
        setDetail(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Kayıtlar alınamadı");
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }

  useEffect(() => {
    setLimit(PAGE_SIZE);
    void load(PAGE_SIZE);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [password]);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      listRef.current?.scrollTo({ top: 0 });
      void load(PAGE_SIZE);
    }, 250);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wordQuery, sourceFilter, statusFilter]);

  async function selectRecording(item: AdminRecording) {
    setSelectedId(item.recording_id);
    setDetail(null);
    setDetailError("");
    setShowJson(false);
    setDetailLoading(true);
    try {
      const json = await fetchAdminRecordingResult(item.recording_id, password);
      setDetail(json as AdminRecordingPayload);
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : "Sonuç alınamadı");
    } finally {
      setDetailLoading(false);
    }
  }

  async function reassessCurrentRecording() {
    if (!detail?.result) return;
    const metadata = detail.metadata ?? {};
    const expectedPhonemes = detail.result.expected_phonemes ?? [];
    if (!expectedPhonemes.length) {
      setDetailError("Beklenen fonemler bulunamadı; yeniden değerlendirme yapılamıyor.");
      return;
    }
    setReassessing(true);
    setDetailError("");
    try {
      const audio = await fetchAdminRecordingAudio(detail.recording_id, password);
      const result = await postAssess({
        audio,
        model_id: detail.result.model_id,
        word: stringMeta(metadata.word ?? metadata.prompt),
        expected_phonemes: expectedPhonemes,
        duration_ms: detail.result.recording_quality?.duration_ms ?? numberMeta(metadata.duration_ms) ?? 1000,
        source: stringMeta(metadata.source),
        participant_name: stringMeta(metadata.participant_name),
        consent_audio: boolMeta(metadata.consent_audio) ?? true,
        client_id: stringMeta(metadata.client_id),
        exercise_mode: stringMeta(metadata.exercise_mode),
        test_set: stringMeta(metadata.test_set),
        test_index: numberMeta(metadata.test_index),
        prompt_id: stringMeta(metadata.prompt_id),
        session_id: stringMeta(metadata.session_id),
        target_kind: targetKindMeta(metadata.target_kind),
        attempt: numberMeta(metadata.attempt),
      });
      const nextId = result.recording_id ?? detail.recording_id;
      setSelectedId(nextId);
      setDetail({
        ...detail,
        recording_id: nextId,
        created_at: new Date().toISOString(),
        result,
      });
      void load(PAGE_SIZE);
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : "Yeniden değerlendirme başarısız");
    } finally {
      setReassessing(false);
    }
  }

  const selectedItem = items.find((item) => item.recording_id === selectedId) ?? null;
  const hasActiveFilters = Boolean(wordQuery.trim() || sourceFilter !== "all" || statusFilter !== "all");
  const sourceOptions = useMemo(() => uniqueOptions(items.map((item) => item.source)), [items]);
  const statusOptions = useMemo(() => uniqueOptions(items.map((item) => item.quality_status ?? item.status)), [items]);

  function handleListScroll() {
    const el = listRef.current;
    if (!el || loading || loadingMore || !hasMore) return;
    const distanceToBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (distanceToBottom < 160) {
      void load(limit + PAGE_SIZE, "more");
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="title">Kayıtlar</h2>
          <p className="mt-1 text-[13px] text-muted">Arşiv kayıtlarını seç, sonucu incele ve sesi dinle.</p>
        </div>
        <button className="btn-sm" onClick={() => void load(PAGE_SIZE)} disabled={loading}>
          {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          Yenile
        </button>
      </div>

      {error && <div className="note">{error}</div>}

      <div className="grid min-h-[640px] grid-cols-1 gap-4 lg:grid-cols-[340px_minmax(0,1fr)]">
        <aside className="overflow-hidden rounded-2xl border border-line bg-surface shadow-card">
          <div className="border-b border-line bg-canvas px-4 py-3">
            <div className="eyebrow">Kayıt listesi</div>
            <div className="mt-1 text-[12px] text-muted">
              {items.length} sonuç yüklendi
            </div>
            <div className="mt-3 space-y-2">
              <label className="relative block">
                <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                <input
                  className="field h-10 rounded-lg pl-9 text-[13px]"
                  value={wordQuery}
                  onChange={(e) => setWordQuery(e.target.value)}
                  placeholder="Kelime ara"
                />
              </label>
              <div className="grid grid-cols-2 gap-2">
                <select
                  className="select h-10 rounded-lg px-3 py-0 text-[12px]"
                  value={sourceFilter}
                  onChange={(e) => setSourceFilter(e.target.value)}
                  aria-label="Kaynak filtresi"
                >
                  <option value="all">Tüm kaynaklar</option>
                  {sourceOptions.map((source) => (
                    <option key={source} value={source}>
                      {source}
                    </option>
                  ))}
                </select>
                <select
                  className="select h-10 rounded-lg px-3 py-0 text-[12px]"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  aria-label="Durum filtresi"
                >
                  <option value="all">Tüm durumlar</option>
                  {statusOptions.map((status) => (
                    <option key={status} value={status}>
                      {status}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {loading ? (
            <div className="flex h-48 items-center justify-center text-muted">
              <Loader2 className="animate-spin" size={24} />
            </div>
          ) : items.length === 0 && !hasActiveFilters ? (
            <div className="px-4 py-10 text-center text-[13px] text-muted">Henüz kayıt yok.</div>
          ) : items.length === 0 ? (
            <div className="px-4 py-10 text-center text-[13px] text-muted">Filtreye uyan kayıt yok.</div>
          ) : (
            <div
              ref={listRef}
              className="max-h-[calc(100vh-350px)] divide-y divide-line overflow-auto"
              onScroll={handleListScroll}
            >
              {items.map((item) => (
                <button
                  key={item.recording_id}
                  className={clsx(
                    "block w-full px-4 py-3 text-left transition hover:bg-brand-soft/35",
                    selectedId === item.recording_id && "bg-brand-soft/60"
                  )}
                  onClick={() => void selectRecording(item)}
                >
                  <div className="flex min-w-0 items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-[13px] font-semibold text-ink">
                        {formatDateTime(item.created_at)} - {recordingTitle(item)}
                      </div>
                      <div className="mt-1 truncate text-[12px] text-muted">
                        {participantLabel(item)}
                      </div>
                    </div>
                    <ScorePill score={item.score} />
                  </div>
                  <div className="mt-2 flex min-w-0 items-center gap-2 font-mono text-[10px] text-muted">
                    {chipText(item.source || "—")}
                    <span className="truncate">{item.prompt_id || item.recording_id}</span>
                  </div>
                </button>
              ))}
              {loadingMore && (
                <div className="flex items-center justify-center gap-2 px-4 py-3 text-[12px] text-muted">
                  <Loader2 size={14} className="animate-spin" />
                  Daha fazla kayıt yükleniyor…
                </div>
              )}
              {!hasMore && items.length > PAGE_SIZE && (
                <div className="px-4 py-3 text-center text-[11px] text-muted">Tüm kayıtlar yüklendi.</div>
              )}
            </div>
          )}
        </aside>

        <main className="min-w-0">
          {!selectedId ? (
            <div className="card flex min-h-[420px] items-center justify-center text-center">
              <div>
                <div className="eyebrow">Detay</div>
                <p className="mt-3 text-[13px] text-muted">Soldaki listeden bir kayıt seç.</p>
              </div>
            </div>
          ) : detailLoading ? (
            <div className="card flex min-h-[420px] items-center justify-center text-muted">
              <Loader2 className="animate-spin" size={26} />
            </div>
          ) : detailError ? (
            <div className="note">{detailError}</div>
          ) : detail?.result ? (
            <div className="space-y-4">
              <RecordingHeader
                item={selectedItem}
                detail={detail}
                password={password}
                showJson={showJson}
                reassessing={reassessing}
                onToggleJson={() => setShowJson((v) => !v)}
                onReassess={() => void reassessCurrentRecording()}
              />
              <AssessResultView resp={detail.result} showTechnical />
              {showJson && (
                <pre className="max-h-[420px] overflow-auto rounded-xl border border-line bg-ink p-4 text-[11px] leading-5 text-white">
                  {JSON.stringify(detail, null, 2)}
                </pre>
              )}
            </div>
          ) : (
            <div className="note">Bu kayıt için değerlendirme sonucu bulunamadı.</div>
          )}
        </main>
      </div>
    </div>
  );
}

function RecordingHeader({
  item,
  detail,
  password,
  showJson,
  reassessing,
  onToggleJson,
  onReassess,
}: {
  item: AdminRecording | null;
  detail: AdminRecordingPayload;
  password: string;
  showJson: boolean;
  reassessing: boolean;
  onToggleJson: () => void;
  onReassess: () => void;
}) {
  const metadata = detail.metadata ?? {};
  const title = item ? recordingTitle(item) : String(metadata.word ?? metadata.prompt ?? "—");
  const participant = item ? participantLabel(item) : String(metadata.participant_name ?? metadata.client_id ?? "Anonim");

  return (
    <div className="card space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="eyebrow">Seçili kayıt</div>
          <h3 className="mt-1 truncate font-serif text-[28px] font-semibold leading-tight text-ink">{title}</h3>
          <p className="mt-1 text-[13px] text-muted">
            {formatDateFull(detail.created_at)} · {participant}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="btn-sm" onClick={onReassess} disabled={reassessing}>
            {reassessing ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Yeniden değerlendir
          </button>
          <button className="btn-sm" onClick={onToggleJson} title="Ham JSON">
            <FileJson size={14} />
            {showJson ? "JSON gizle" : "JSON"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <HeaderMetric label="Kaynak" value={String(metadata.source ?? item?.source ?? "—")} />
        <HeaderMetric label="Model" value={detail.result?.model_id ?? item?.model_id ?? "—"} />
        <HeaderMetric label="Hedef" value={String(metadata.target_kind ?? item?.target_kind ?? "—")} />
        <HeaderMetric label="Deneme" value={String(metadata.attempt ?? item?.attempt ?? "—")} />
      </div>

      <div className="audio-row">
        <Volume2 size={16} className="shrink-0 text-muted" />
        <AdminAudio recordingId={detail.recording_id} password={password} />
      </div>
    </div>
  );
}

function AdminAudio({ recordingId, password }: { recordingId: string; password: string }) {
  const [url, setUrl] = useState("");
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl = "";
    setUrl("");
    setFailed(false);
    fetchAdminRecordingAudio(recordingId, password)
      .then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [recordingId, password]);

  if (failed) return <span className="text-[12px] text-muted">Ses alınamadı.</span>;
  if (!url) return <span className="font-mono text-[11px] text-muted">ses yükleniyor…</span>;

  return (
    <audio className="h-9 w-full" controls src={url}>
      <track kind="captions" />
    </audio>
  );
}

function HeaderMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-line bg-canvas px-3 py-2">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 min-w-0 truncate font-mono text-[12px] text-ink" title={value}>
        {value}
      </div>
    </div>
  );
}

function ScorePill({ score }: { score?: number | null }) {
  if (score == null) return null;
  const tone =
    score >= 75
      ? "border-ok/25 bg-ok-soft text-ok"
      : score >= 50
      ? "border-warn/25 bg-warn-soft text-warn"
      : "border-bad/25 bg-bad-soft text-bad";
  return (
    <span className={clsx("shrink-0 rounded-full border px-2 py-0.5 font-mono text-[10px]", tone)}>
      {score}%
    </span>
  );
}

function chipText(label: string) {
  return (
    <span className="shrink-0 rounded-full border border-line bg-canvas px-2 py-0.5 uppercase tracking-[0.12em]">
      {label}
    </span>
  );
}

function recordingTitle(item: AdminRecording): string {
  return item.word || item.prompt || item.prompt_id || "—";
}

function participantLabel(item: AdminRecording): string {
  return item.participant_name || item.client_id || "Anonim";
}

function stringMeta(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function numberMeta(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function boolMeta(value: unknown): boolean | undefined {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    if (value === "true") return true;
    if (value === "false") return false;
  }
  return undefined;
}

function targetKindMeta(value: unknown): "manual" | "canonical" | "intended_wrong" | undefined {
  if (value === "manual" || value === "canonical" || value === "intended_wrong") return value;
  return undefined;
}

function uniqueOptions(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.map((value) => value || "—"))].sort((a, b) =>
    a.localeCompare(b, "tr")
  );
}

function formatDateTime(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("tr-TR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDateFull(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("tr-TR", {
    year: "numeric",
    month: "long",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
