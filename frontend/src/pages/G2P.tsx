import clsx from "clsx";
import { Loader2, Search } from "lucide-react";
import { type FormEvent, useEffect, useState } from "react";
import { fetchG2P, fetchPhonemes, type G2PResult, type PhonemeItem, type TraceStep } from "../api";
import { useSettings } from "../context/SettingsContext";

export function G2P() {
  const { settings, update } = useSettings();
  const [phonemes, setPhonemes] = useState<Record<string, PhonemeItem>>({});
  const [word, setWord] = useState("ekmek");
  const [onlyFired, setOnlyFired] = useState(false);
  const [result, setResult] = useState<G2PResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchPhonemes()
      .then((r) => setPhonemes(r.phonemes))
      .catch(() => undefined);
  }, []);

  async function run(e?: FormEvent) {
    e?.preventDefault();
    if (!word.trim()) return;
    setLoading(true);
    setError("");
    try {
      const r = await fetchG2P(word.trim(), {
        pedagogical: settings.pedagogical,
        use_reference: settings.useReference,
      });
      setResult(r);
    } catch (err) {
      setError(err instanceof Error ? err.message : "G2P başarısız");
    } finally {
      setLoading(false);
    }
  }

  const trace = result?.trace.filter((s) => !onlyFired || s.fired) ?? [];

  return (
    <div className="space-y-4">
      {/* Input */}
      <div className="card space-y-4">
        <div className="eyebrow">G2P İnceleyici</div>
        <form className="flex gap-2" onSubmit={run}>
          <input
            className="field flex-1"
            value={word}
            onChange={(e) => setWord(e.target.value)}
            placeholder="örn. merhaba"
            autoFocus
          />
          <button
            className="btn-sm shrink-0 px-4"
            type="submit"
            disabled={loading || !word.trim()}
          >
            {loading ? <Loader2 size={15} className="animate-spin" /> : <Search size={15} />}
            <span className="hidden sm:inline">Çalıştır</span>
          </button>
        </form>

        <div className="flex flex-wrap gap-4">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={settings.pedagogical}
              onChange={(e) => update({ pedagogical: e.target.checked })}
            />
            Pedagojik allofonlar
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={settings.useReference}
              onChange={(e) => update({ useReference: e.target.checked })}
            />
            Referans sözlük
          </label>
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={onlyFired}
              onChange={(e) => setOnlyFired(e.target.checked)}
            />
            Sadece aktif adımlar
          </label>
        </div>

        {error && (
          <div className="rounded-xl border border-bad/25 bg-bad-soft px-4 py-3 text-[13px] text-bad">
            {error}
          </div>
        )}
      </div>

      {/* Result summary */}
      {result && (
        <div className="card animate-fade-up">
          <div className="flex flex-wrap items-center gap-3">
            <h2 className="font-serif text-[28px] font-semibold text-ink">
              {result.normalized || result.word}
            </h2>
            <div className="flex flex-wrap items-center gap-1">
              {splitTokens(result.ipa).map((tok, i) => (
                <span
                  key={`${tok}-${i}`}
                  className={clsx("trace-token", tokenColorClass(tok, phonemes))}
                >
                  {tok}
                </span>
              ))}
            </div>
            <SourceBadge source={result.source} />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="rounded-full border border-line bg-canvas px-2.5 py-1 font-mono text-[11px] text-muted">
              {formatSyllables(result.syllables)}
            </span>
            {result.exception_type && (
              <span className="rounded-full border border-warn/25 bg-warn-soft px-2.5 py-1 font-mono text-[11px] text-warn">
                {result.exception_type}
              </span>
            )}
            {result.warnings.map((w) => (
              <span
                key={w}
                className="rounded-full border border-warn/25 bg-warn-soft px-2.5 py-1 font-mono text-[11px] text-warn"
              >
                {w}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Trace */}
      {result && (
        <div className="card animate-fade-up space-y-2">
          <div className="flex items-center justify-between">
            <div className="eyebrow">Pipeline trace</div>
            <span className="font-mono text-[11px] text-muted">
              {trace.filter((s) => s.fired).length} / {trace.length} aktif
            </span>
          </div>
          <ol className="space-y-2">
            {trace.map((step, i) => (
              <TraceStepView key={`${step.name}-${i}`} step={step} index={i + 1} phonemes={phonemes} />
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

function TraceStepView({
  step,
  index,
  phonemes,
}: {
  step: TraceStep;
  index: number;
  phonemes: Record<string, PhonemeItem>;
}) {
  return (
    <li className={clsx("trace-row", step.fired ? "trace-row-fired" : "trace-row-idle")}>
      <div className="font-mono text-[11px] text-muted w-5 shrink-0">{String(index).padStart(2, "0")}</div>
      <div className="min-w-0 flex-1">
        <div className="font-mono text-[13px] font-semibold text-ink">{step.name}</div>
        {step.note && <div className="mt-0.5 text-[11px] text-muted">{step.note}</div>}
      </div>
      <div className="flex flex-wrap items-center gap-1 shrink-0">
        <TokenSeq tokens={step.input} changed={step.fired} mode="before" other={step.output} phonemes={phonemes} />
        <span className="font-mono text-[11px] text-muted">→</span>
        <TokenSeq tokens={step.output} changed={step.fired} mode="after" other={step.input} phonemes={phonemes} />
      </div>
    </li>
  );
}

function TokenSeq({
  tokens,
  changed,
  mode,
  other,
  phonemes,
}: {
  tokens: string[];
  changed: boolean;
  mode: "before" | "after";
  other: string[];
  phonemes: Record<string, PhonemeItem>;
}) {
  if (tokens.length === 0) {
    return <span className="font-mono text-[12px] text-muted">∅</span>;
  }
  const bounds = diffBounds(mode === "before" ? tokens : other, mode === "after" ? tokens : other);
  return (
    <span className="flex flex-wrap gap-0.5">
      {tokens.map((tok, i) => {
        const inRange =
          changed && i >= bounds.prefix && i < tokens.length - bounds.suffix;
        return (
          <span
            key={`${tok}-${i}`}
            className={clsx(
              "trace-token",
              tokenColorClass(tok, phonemes),
              inRange && mode === "before" && "trace-token-removed",
              inRange && mode === "after" && "trace-token-added"
            )}
          >
            {tok}
          </span>
        );
      })}
    </span>
  );
}

function diffBounds(before: string[], after: string[]) {
  let prefix = 0;
  const minLen = Math.min(before.length, after.length);
  while (prefix < minLen && before[prefix] === after[prefix]) prefix++;
  let suffix = 0;
  while (
    suffix < before.length - prefix &&
    suffix < after.length - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix++;
  }
  return { prefix, suffix };
}

function SourceBadge({ source }: { source: string }) {
  const cls =
    source === "reference"
      ? "border-brand/25 bg-brand-soft text-brand"
      : source === "exception"
      ? "border-extra/25 bg-extra-soft text-extra"
      : "border-line bg-canvas text-muted";
  return (
    <span className={clsx("rounded-full border px-2.5 py-1 font-mono text-[11px]", cls)}>
      {source}
    </span>
  );
}

function formatSyllables(syl: string[][]): string {
  return syl.length ? syl.map((s) => s.join("")).join(".") : "no syllables";
}

function splitTokens(ipa: string): string[] {
  const result: string[] = [];
  let i = 0;
  while (i < ipa.length) {
    const ch = ipa[i];
    // Multi-char affricates
    if (ch === "t" && ipa.slice(i, i + 3) === "t͡ʃ") {
      result.push("t͡ʃ");
      i += 3;
    } else if (ch === "d" && ipa.slice(i, i + 3) === "d͡ʒ") {
      result.push("d͡ʒ");
      i += 3;
    } else if (ch === "ɾ" && ipa.slice(i, i + 4) === "ɾ̞̊") {
      result.push("ɾ̞̊");
      i += 4;
    } else if (ch === "β" && ipa.slice(i, i + 2) === "β̞") {
      result.push("β̞");
      i += 2;
    } else if (ch === "ː") {
      // attach to previous
      if (result.length) result[result.length - 1] += "ː";
      i++;
    } else if (ch === "ʰ" && result.length) {
      result[result.length - 1] += "ʰ";
      i++;
    } else {
      result.push(ch);
      i++;
    }
  }
  return result.filter((t) => t !== " ");
}

function tokenColorClass(tok: string, phonemes: Record<string, PhonemeItem>): string {
  const info = phonemes[tok];
  if (!info) return "border-line bg-canvas text-muted";
  switch (info.category) {
    case "vowel":
      return "border-brand/25 bg-brand-soft text-brand";
    case "consonant":
      return "border-ink/15 bg-ink/5 text-ink";
    case "suprasegmental":
      return "border-muted/25 bg-canvas text-muted";
    default:
      return "border-line bg-canvas text-muted";
  }
}
