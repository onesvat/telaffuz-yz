import { Check, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchModels, type ModelItem } from "../api";
import { useSettings } from "../context/SettingsContext";
import { TEST_SETS } from "./Learner";

export function Settings() {
  const { settings, update } = useSettings();
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchModels()
      .then((r) => setModels(r.items))
      .catch(() => setModels([]))
      .finally(() => setLoading(false));
  }, []);

  function handleUpdate<K extends keyof typeof settings>(key: K, value: (typeof settings)[K]) {
    update({ [key]: value });
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="mx-auto max-w-[560px] space-y-6">
      {/* Model */}
      <section className="card space-y-4">
        <div className="flex items-center justify-between">
          <div className="eyebrow">Varsayılan Model</div>
          {loading && <Loader2 size={14} className="animate-spin text-muted" />}
        </div>

        {!loading && models.length === 0 && (
          <p className="text-[13px] text-muted">Model listesi alınamadı.</p>
        )}

        {models.map((m) => (
          <button
            key={m.id}
            onClick={() => m.available && handleUpdate("modelId", m.id)}
            disabled={!m.available}
            className={[
              "w-full rounded-xl border px-4 py-3 text-left transition",
              m.id === settings.modelId
                ? "border-brand bg-brand-soft/50"
                : "border-line bg-canvas hover:border-brand/40",
              !m.available && "cursor-not-allowed opacity-40",
            ].join(" ")}
          >
            <div className="flex items-center justify-between">
              <div>
                <div className="text-[14px] font-semibold text-ink">{m.label}</div>
                <div className="mt-0.5 font-mono text-[11px] text-muted">{m.id}</div>
              </div>
              <div className="flex items-center gap-2">
                {m.frozen && (
                  <span className="rounded-full border border-line bg-canvas px-2 py-0.5 font-mono text-[10px] text-muted">
                    frozen
                  </span>
                )}
                {m.is_default && (
                  <span className="rounded-full border border-brand/25 bg-brand-soft px-2 py-0.5 font-mono text-[10px] text-brand">
                    varsayılan
                  </span>
                )}
                {m.id === settings.modelId && (
                  <Check size={16} className="text-brand" />
                )}
              </div>
            </div>
            {!m.available && (
              <p className="mt-1 text-[12px] text-bad">Checkpoint bulunamadı.</p>
            )}
          </button>
        ))}

        <p className="text-[12px] text-muted">
          Bu model Sandbox ve Öğrenci Modu değerlendirmelerinde varsayılan olarak kullanılır.
        </p>
      </section>

      {/* G2P options */}
      <section className="card space-y-4">
        <div className="eyebrow">G2P Seçenekleri</div>

        <label className="toggle-label">
          <input
            type="checkbox"
            checked={settings.pedagogical}
            onChange={(e) => handleUpdate("pedagogical", e.target.checked)}
          />
          <div>
            <div className="font-medium text-ink">Pedagojik allofonlar</div>
            <div className="text-[12px] text-muted">Soluklu patlamalar, ötümsüz r gibi pedagojik eklere dahil et.</div>
          </div>
        </label>

        <label className="toggle-label">
          <input
            type="checkbox"
            checked={settings.useReference}
            onChange={(e) => handleUpdate("useReference", e.target.checked)}
          />
          <div>
            <div className="font-medium text-ink">Referans sözlük</div>
            <div className="text-[12px] text-muted">Önce 50K kelimelik manuel IPA sözlüğüne bak.</div>
          </div>
        </label>
      </section>

      {/* Learner / VAD */}
      <section className="card space-y-4">
        <div className="eyebrow">Öğrenci Modu — VAD Hassasiyeti</div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-[14px] font-medium text-ink">Sessizlik eşiği</label>
            <span className="font-mono text-[12px] text-muted">{settings.silenceMs} ms</span>
          </div>
          <input
            type="range"
            min={600}
            max={2000}
            step={100}
            value={settings.silenceMs}
            onChange={(e) => handleUpdate("silenceMs", Number(e.target.value))}
            className="w-full accent-brand"
          />
          <div className="mt-1 flex justify-between text-[11px] text-muted">
            <span>600ms — hızlı</span>
            <span>2000ms — yavaş</span>
          </div>
          <p className="mt-2 text-[12px] text-muted">
            Konuşma bittikten sonra kaydın durması için gereken sessizlik süresi.
            Düşük değer daha hızlı keser, yüksek değer cümle sonlarını daha iyi yakalar.
          </p>
        </div>
      </section>

      {/* Test sets info */}
      <section className="card space-y-3">
        <div className="eyebrow">Mevcut Test Setleri</div>
        {TEST_SETS.map((set) => (
          <div key={set.id} className="rounded-xl border border-line bg-canvas px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="text-[14px] font-semibold text-ink">{set.name}</div>
              <span className="font-mono text-[11px] text-muted">{set.items.length} kelime</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {set.items.map((item) => (
                <span
                  key={item.word}
                  className="rounded-md border border-line bg-surface px-2 py-0.5 text-[12px] text-ink"
                >
                  {item.word}
                </span>
              ))}
            </div>
          </div>
        ))}
        <p className="text-[12px] text-muted">
          Test seti kelimelerini değiştirmek için <code className="font-mono">src/pages/Learner.tsx</code> dosyasındaki <code className="font-mono">TEST_SETS</code> dizisini düzenle.
        </p>
      </section>

      {/* Save indicator */}
      {saved && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2 rounded-full border border-ok/25 bg-ok-soft px-4 py-2 shadow-card text-[13px] text-ok animate-fade-up">
          <Check size={14} /> Kaydedildi
        </div>
      )}
    </div>
  );
}
