import { BookOpen, Cpu, FlaskConical, Mic, Settings, Wand2 } from "lucide-react";
import type { Page } from "../App";

export function Landing({ onNavigate }: { onNavigate: (p: Page) => void }) {
  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="card">
        <div className="eyebrow text-brand">İstanbul Üniversitesi-Cerrahpaşa · 2026</div>
        <h1 className="title mt-2">Türkçe Telaffuz Değerlendirme</h1>
        <p className="lede">
          Grafem-fonem dönüşümü (G2P), wav2vec 2.0 ince-ayarı ve GOP skorlamasını birleştiren
          hibrit bir konuşma değerlendirme sistemi. Doğrudan tarayıcıdan deneyebilirsin.
        </p>

        <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <Stat label="Fonem envanteri" value="49" sub="Türkçe sesbirim" />
          <Stat label="Referans sözlük" value="50K+" sub="manuel IPA girişi" />
          <Stat label="Değerlendirme" value="GOP" sub="Goodness of Pronunciation" />
        </div>
      </div>

      {/* Tool cards */}
      <div>
        <div className="eyebrow mb-3">Araçlar</div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <button className="tool-card text-left" onClick={() => onNavigate("g2p")}>
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-brand-soft text-brand">
                <Wand2 size={20} />
              </div>
              <div>
                <div className="font-semibold text-ink">G2P İnceleyici</div>
                <div className="text-[13px] text-muted">Kelime → IPA + pipeline trace</div>
              </div>
            </div>
            <p className="mt-3 text-[13px] leading-6 text-muted">
              Herhangi bir Türkçe kelimeyi adım adım fonem dizisine çevir. Morpholoji, ünlü
              uyumu, allofon kuralları ve vurgu atamalarını canlı gör.
            </p>
          </button>

          <button className="tool-card text-left" onClick={() => onNavigate("sandbox")}>
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-ok-soft text-ok">
                <FlaskConical size={20} />
              </div>
              <div>
                <div className="font-semibold text-ink">Sandbox</div>
                <div className="text-[13px] text-muted">Kaydet → GOP değerlendirmesi</div>
              </div>
            </div>
            <p className="mt-3 text-[13px] leading-6 text-muted">
              Bir kelime söyle veya ses dosyası yükle; model fonem hizalaması ve GOP
              skorunu hesaplar. Farklı modeller ve profilleri karşılaştır.
            </p>
          </button>

          <button className="tool-card text-left" onClick={() => onNavigate("learner")}>
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-extra-soft text-extra">
                <Mic size={20} />
              </div>
              <div>
                <div className="font-semibold text-ink">Öğrenci Modu</div>
                <div className="text-[13px] text-muted">Serbest veya test listesiyle pratik</div>
              </div>
            </div>
            <p className="mt-3 text-[13px] leading-6 text-muted">
              Serbest modda istediğin kelimeyi söyle; test listesinde hazır kelimelerle
              otomatik kayıt ve fonem bazlı geri bildirim al.
            </p>
          </button>

          <button className="tool-card text-left" onClick={() => onNavigate("settings")}>
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-warn-soft text-warn">
                <Settings size={20} />
              </div>
              <div>
                <div className="font-semibold text-ink">Ayarlar</div>
                <div className="text-[13px] text-muted">Model, G2P, VAD hassasiyeti</div>
              </div>
            </div>
            <p className="mt-3 text-[13px] leading-6 text-muted">
              Varsayılan modeli ve G2P seçeneklerini seç. Öğrenci modu için sessizlik
              eşiği ve otomatik ilerleme davranışını yapılandır.
            </p>
          </button>
        </div>
      </div>

      {/* System overview */}
      <div>
        <div className="eyebrow mb-3">Sistem</div>
        <div className="card space-y-4">
          <SystemBlock
            icon={<BookOpen size={16} />}
            title="G2P Pipeline"
            description="Normalize → sözlük arama → istisna listesi → morfololji analizi → grafem haritası → vokal armoni → allofon kuralları → vurgu atama. 15+ kural zinciri."
          />
          <div className="border-t border-line" />
          <SystemBlock
            icon={<Cpu size={16} />}
            title="wav2vec 2.0 İnce Ayarı"
            description="facebook/mms-300m ve facebook/wav2vec2-xls-r-300m modelleri 49 Türkçe fonem üzerinde fine-tune edildi. MMS-FA hizalama modeli ile fonem sınırları belirleniyor."
          />
          <div className="border-t border-line" />
          <SystemBlock
            icon={<FlaskConical size={16} />}
            title="Hibrit GOP Skorlama"
            description="Her fonem için canonical ve best log-olasılık oranından GOP skoru hesaplanır. İstanbul Türkçesi standardında allofon, uzun ünlü, tempo, akıcılık ve anlaşılabilirlik birlikte puanlanır."
          />
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-xl border border-line bg-canvas px-4 py-3">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 font-serif text-[24px] font-semibold text-ink">{value}</div>
      <div className="text-[12px] text-muted">{sub}</div>
    </div>
  );
}

function SystemBlock({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="flex gap-3">
      <div className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-brand-soft text-brand">
        {icon}
      </div>
      <div>
        <div className="font-semibold text-[14px] text-ink">{title}</div>
        <p className="mt-1 text-[13px] leading-6 text-muted">{description}</p>
      </div>
    </div>
  );
}
