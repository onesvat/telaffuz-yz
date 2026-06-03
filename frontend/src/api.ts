const API_ORIGIN = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const BASE = `${API_ORIGIN}/api/v1`;

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) throw new Error(await res.text());
  return (await res.json()) as T;
}

// ---- phoneme catalog ----

export type PhonemeItem = {
  symbol: string;
  name_tr: string;
  example: string;
  hint_tr: string;
  category: string;
};

export type PhonemeGroup = {
  key: string;
  label: string;
  items: PhonemeItem[];
};

export type PhonemesResponse = {
  phonemes: Record<string, PhonemeItem>;
  groups: PhonemeGroup[];
  category_labels: Record<string, string>;
  syllable_boundary: string;
};

export function fetchPhonemes(): Promise<PhonemesResponse> {
  return apiFetch("/phonemes");
}

// ---- G2P ----

export type TraceStep = {
  name: string;
  input: string[];
  output: string[];
  fired: boolean;
  note: string;
};

export type G2PResult = {
  word: string;
  normalized: string;
  ipa: string;
  phonemes: string[];
  syllables: string[][];
  source: string;
  exception_type: string | null;
  warnings: string[];
  trace: TraceStep[];
};

export function fetchG2P(
  word: string,
  opts: { pedagogical?: boolean; use_reference?: boolean } = {}
): Promise<G2PResult> {
  const p = new URLSearchParams({ word });
  if (opts.pedagogical !== undefined) p.set("pedagogical", String(opts.pedagogical));
  if (opts.use_reference !== undefined) p.set("use_reference", String(opts.use_reference));
  return apiFetch(`/g2p/inspect?${p}`);
}

// ---- models ----

export type ModelItem = {
  id: string;
  label: string;
  available: boolean;
  frozen: boolean;
  is_default: boolean;
};

export function fetchModels(): Promise<{ items: ModelItem[] }> {
  return apiFetch("/models");
}

// ---- assess ----

export type PhoneTiming = {
  ipa: string;
  start_ms: number;
  end_ms: number;
  confidence: number | null;
};

export type RecordingQuality = {
  status: "valid" | "invalid";
  duration_ms: number;
  sample_rate: number | null;
  rms: number | null;
  reason: string | null;
};

export type CoachTarget = {
  text: string | null;
  ipa: string;
  phonemes: string[];
  scoreable_phonemes: string[];
};

export type PosteriorCandidate = {
  phone: string;
  probability: number;
};

export type PosteriorCandidateRow = {
  index: number;
  raw_ipa: string | null;
  start_ms: number;
  end_ms: number;
  frame_count: number;
  candidates: PosteriorCandidate[];
};

export type Wav2VecPhone = PhoneTiming & {
  raw_ipa?: string;
};

export type CoachWav2Vec = {
  model: string;
  ipa: string;
  phonemes: string[];
  phones: Wav2VecPhone[];
  timed_phones: Wav2VecPhone[];
  posterior_candidates?: PosteriorCandidateRow[];
};

// Length is a non-gating channel: it never folds into `quality`. `note` is
// "short" | "long" | "expected" | null; `z` is the robust-scaled deviation.
export type LengthEvidence = {
  score?: number | null;
  note?: "short" | "long" | "expected" | null;
  z?: number | null;
  duration_ms?: number | null;
};

export type CoachAnalysisPhone = Wav2VecPhone & {
  raw_ipa?: string;
  quality?: number | null;
  quality_score?: number | null;
  length_score?: number | null;
  length_note?: "short" | "long" | "expected" | null;
  length_z?: number | null;
  duration_ms?: number | null;
  quantity_span?: Record<string, unknown> | null;
  changed_by_analysis?: boolean;
  change_id?: string | null;
  analysis_source?: string | null;
};

export type CoachAnalysis = {
  ipa: string;
  phonemes: string[];
  phones: CoachAnalysisPhone[];
  timed_phones: CoachAnalysisPhone[];
  changes: Record<string, unknown>[];
};

export type CoachResultPhone = {
  index: number;
  index_expected: number | null;
  index_observed: number | null;
  expected: string | null;
  observed: string | null;
  status: "correct" | "incorrect" | "missing" | "extra";
  strict_status: "correct" | "incorrect" | "missing" | "extra";
  score: number;
  score_percent: number;
  reason?: string;
  score_reason: string;
  raw_observed?: string | null;
  analysis_observed?: string | null;
  distance_features?: Record<string, unknown>;
  quality?: number | null;
  length?: LengthEvidence | null;
  changed_by_analysis?: Record<string, unknown> | null;
  timing?: { start_ms?: number | null; end_ms?: number | null } | null;
};

export type CoachPhoneScore = {
  index: number;
  index_expected: number | null;
  expected: string | null;
  observed: string | null;
  score: number;
  score_percent: number;
  strict_status: "correct" | "incorrect" | "missing" | "extra";
  score_reason: string;
};

export type CoachScore = {
  item_score: number;
  max_item_score: number;
  word_score: number;
  phoneme_core: number;
  extra_penalty: number;
  score_version: string;
  phone_scores: CoachPhoneScore[];
};

export type CoachSummary = {
  target_count: number;
  correct_count: number;
  incorrect_count: number;
  missing_count: number;
  extra_count: number;
  accuracy: number;
  mean_phone_score: number;
  word_score: number;
  quality_score: number | null;
  extra_penalty: number;
  quality_factor: number;
  overall_score: number;
  score_version: string;
};

export type CoachResult = {
  phones: CoachResultPhone[];
  score: CoachScore;
  summary: CoachSummary;
};

export type AssessResponse = {
  recording_id?: string;
  status:
    | "ok"
    | "valid"
    | "invalid"
    | "no_speech"
    | "alignment_failed"
    | "no_intervals"
    | "unknown_model"
    | "model_unavailable"
    | "failed";
  recording_quality: RecordingQuality;
  expected_phonemes: string[];
  model_id: string;
  target: CoachTarget;
  wav2vec: CoachWav2Vec;
  analysis: CoachAnalysis;
  result: CoachResult | null;
  score: CoachScore | null;
  error_code: string | null;
  debug?: Record<string, unknown>;
};

export function postAssess(opts: {
  audio: Blob;
  model_id: string;
  word?: string;
  expected_phonemes?: string[];
  duration_ms: number;
  source?: string;
  participant_name?: string;
  consent_audio?: boolean;
  client_id?: string;
  exercise_mode?: string;
  test_set?: string;
  test_index?: number;
  prompt_id?: string;
  session_id?: string;
  target_kind?: "manual" | "canonical" | "intended_wrong";
  attempt?: number;
}): Promise<AssessResponse> {
  const form = new FormData();
  form.append("audio", opts.audio, "recording.webm");
  form.append("model_id", opts.model_id);
  if (opts.word) form.append("word", opts.word);
  if (opts.expected_phonemes?.length)
    form.append("expected_phonemes", opts.expected_phonemes.join(" "));
  form.append("duration_ms", String(opts.duration_ms));
  if (opts.source) form.append("source", opts.source);
  if (opts.participant_name) form.append("participant_name", opts.participant_name);
  if (opts.consent_audio !== undefined) form.append("consent_audio", String(opts.consent_audio));
  if (opts.client_id) form.append("client_id", opts.client_id);
  if (opts.exercise_mode) form.append("exercise_mode", opts.exercise_mode);
  if (opts.test_set) form.append("test_set", opts.test_set);
  if (opts.test_index !== undefined) form.append("test_index", String(opts.test_index));
  if (opts.prompt_id) form.append("prompt_id", opts.prompt_id);
  if (opts.session_id) form.append("session_id", opts.session_id);
  if (opts.target_kind) form.append("target_kind", opts.target_kind);
  if (opts.attempt !== undefined) form.append("attempt", String(opts.attempt));
  return apiFetch("/assess", { method: "POST", body: form });
}

// ---- studio (validation recording capture) ----

export type StudioPrompt = {
  prompt_id: string;
  prompt_type: string;
  target_text: string;
  display_reading_text: string;
  canonical_ipa: string;
  intended_wrong_ipa: string;
  error_rule_id: string;
  minimal_pair_id: string;
  focus_phonemes: string;
  read_mode: string;
  intended_error: string;
  focus_category: string;
  difficulty: string;
  speaker_instruction_tr: string;
  error_explanation_tr: string;
  why_selected_tr: string;
  source_refs: string;
  scoring_scope: string;
  is_scored: string;
  expected_human_note: string;
};

export type StudioAttempt = {
  attempt: number;
  filename: string;
  duration_ms?: number | null;
};

export type StudioRecordings = {
  session_id: string;
  recorded_prompts: string[];
  attempt_counts: Record<string, number>;
  attempts: Record<string, StudioAttempt[]>;
};

export type StudioUploadResponse = {
  ok: boolean;
  recording_filename: string;
  recording_id?: string | null;
  attempt: number | null;
  quality: RecordingQuality;
};

// Capture collections share one flow but isolated storage: "validation" is the
// thesis set (/studio), "demo" is the public showcase set (/demo-studio).
export type StudioCollection = "validation" | "demo";

function studioPrefix(collection: StudioCollection = "validation"): string {
  return collection === "demo" ? "/demo-studio" : "/studio";
}

export function fetchStudioPrompts(
  collection: StudioCollection = "validation"
): Promise<{ items: StudioPrompt[] }> {
  return apiFetch(`${studioPrefix(collection)}/prompts`);
}

export function fetchStudioRecordings(
  sessionId: string,
  collection: StudioCollection = "validation"
): Promise<StudioRecordings> {
  return apiFetch(
    `${studioPrefix(collection)}/recordings/${encodeURIComponent(sessionId)}`
  );
}

export async function uploadStudioRecording(opts: {
  sessionId: string;
  promptId: string;
  audio: Blob;
  duration_ms: number;
  collection?: StudioCollection;
}): Promise<{ ok: true; data: StudioUploadResponse } | { ok: false; quality?: RecordingQuality; error?: string }> {
  const form = new FormData();
  form.append("audio", opts.audio, "recording.webm");
  form.append("duration_ms", String(opts.duration_ms));
  const res = await fetch(
    `${BASE}${studioPrefix(opts.collection)}/recordings/${encodeURIComponent(opts.sessionId)}/${encodeURIComponent(opts.promptId)}`,
    { method: "POST", body: form }
  );
  if (res.status === 422) {
    const body = await res.json().catch(() => null);
    return { ok: false, quality: body?.detail?.quality };
  }
  if (!res.ok) return { ok: false, error: (await res.text()) || res.statusText };
  return { ok: true, data: (await res.json()) as StudioUploadResponse };
}

export function deleteStudioAttempt(
  sessionId: string,
  promptId: string,
  attempt: number,
  collection: StudioCollection = "validation"
): Promise<{ ok: boolean; remaining_attempts: number[] }> {
  return apiFetch(
    `${studioPrefix(collection)}/recordings/${encodeURIComponent(sessionId)}/${encodeURIComponent(promptId)}?attempt=${attempt}`,
    { method: "DELETE" }
  );
}

export function studioAudioUrl(
  sessionId: string,
  promptId: string,
  attempt: number,
  collection: StudioCollection = "validation"
): string {
  return `${BASE}${studioPrefix(collection)}/recordings/${encodeURIComponent(sessionId)}/${encodeURIComponent(promptId)}/audio?attempt=${attempt}`;
}

// ---- admin ----

export type AdminRecording = {
  recording_id: string;
  created_at: string;
  source?: string | null;
  participant_name?: string | null;
  client_id?: string | null;
  word?: string | null;
  prompt?: string | null;
  prompt_id?: string | null;
  session_id?: string | null;
  target_kind?: string | null;
  attempt?: number | null;
  model_id?: string | null;
  status?: string | null;
  quality_status?: string | null;
  quality_reason?: string | null;
  duration_ms?: number | null;
  score?: number | null;
  layer_summary?: {
    verdict_counts?: Record<string, number>;
    weakest?: { phone?: string | null; gop_score?: number | null; verdict?: string | null } | null;
  } | null;
};

function adminHeaders(password: string): HeadersInit {
  return { "X-Admin-Password": password };
}

export function adminLogin(password: string): Promise<{ ok: boolean }> {
  return apiFetch("/admin/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}

export function fetchAdminRecordings(
  password: string,
  opts: { limit?: number; word?: string; source?: string; status?: string } = {}
): Promise<{ items: AdminRecording[] }> {
  const p = new URLSearchParams();
  if (opts.limit !== undefined) p.set("limit", String(opts.limit));
  if (opts.word?.trim()) p.set("word", opts.word.trim());
  if (opts.source && opts.source !== "all") p.set("source", opts.source);
  if (opts.status && opts.status !== "all") p.set("status", opts.status);
  const suffix = p.toString() ? `?${p}` : "";
  return apiFetch(`/admin/recordings${suffix}`, { headers: adminHeaders(password) });
}

export async function fetchAdminRecordingAudio(recordingId: string, password: string): Promise<Blob> {
  const res = await fetch(`${BASE}/admin/recordings/${encodeURIComponent(recordingId)}/audio`, {
    headers: adminHeaders(password),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

export function fetchAdminRecordingResult(recordingId: string, password: string): Promise<unknown> {
  return apiFetch(`/admin/recordings/${encodeURIComponent(recordingId)}/result`, {
    headers: adminHeaders(password),
  });
}
