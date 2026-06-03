import clsx from "clsx";
import { AudioLines, Database, FlaskConical, Home, LogOut, Mic, Settings as SettingsIcon, Wand2 } from "lucide-react";
import type { ElementType } from "react";
import { useEffect, useState } from "react";
import { adminLogin } from "./api";
import { SettingsProvider } from "./context/SettingsContext";
import { AdminRecordings } from "./pages/AdminRecordings";
import { G2P } from "./pages/G2P";
import { Landing } from "./pages/Landing";
import { Learner } from "./pages/Learner";
import { Sandbox } from "./pages/Sandbox";
import { Settings } from "./pages/Settings";
import { Studio } from "./pages/Studio";

export type AdminPage = "home" | "recordings" | "g2p" | "sandbox" | "studio" | "settings";
export type Page = AdminPage | "home" | "learner";

const ADMIN_NAV_ITEMS: { id: AdminPage; label: string; icon: ElementType }[] = [
  { id: "home", label: "Ana Sayfa", icon: Home },
  { id: "recordings", label: "Kayıtlar", icon: Database },
  { id: "g2p", label: "G2P", icon: Wand2 },
  { id: "sandbox", label: "Sandbox", icon: FlaskConical },
  { id: "studio", label: "Kayıt", icon: AudioLines },
  { id: "settings", label: "Ayarlar", icon: SettingsIcon },
];

const PAGE_TITLE: Record<AdminPage, string> = {
  home: "Telaffuz YZ",
  recordings: "Kayıtlar",
  g2p: "G2P İnceleyici",
  sandbox: "Sandbox",
  studio: "Doğrulama Kaydı",
  settings: "Ayarlar",
};

function getInitialPage(): AdminPage {
  const hash = window.location.hash.slice(1) as AdminPage;
  return ADMIN_NAV_ITEMS.some((n) => n.id === hash) ? hash : "home";
}

function AppInner() {
  const isAdmin = window.location.pathname.replace(/\/$/, "") === "/admin";
  const [page, setPage] = useState<AdminPage>(getInitialPage);
  const [adminPassword, setAdminPassword] = useState(
    () => window.sessionStorage.getItem("telaffuz_admin_password") ?? ""
  );

  useEffect(() => {
    if (isAdmin) window.location.hash = page;
  }, [page]);

  useEffect(() => {
    const handler = () => {
      const hash = window.location.hash.slice(1) as AdminPage;
      if (ADMIN_NAV_ITEMS.some((n) => n.id === hash)) setPage(hash);
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  const path = window.location.pathname.replace(/\/$/, "");

  if (path === "/kayit") {
    return (
      <div className="min-h-screen bg-canvas">
        <main className="mx-auto w-full max-w-5xl px-4 py-5 sm:py-8">
          <div className="mb-5 flex items-center gap-2 font-serif text-[18px] font-semibold text-brand">
            <Mic size={20} />
            Telaffuz YZ — Demo Kaydı
          </div>
          <Studio
            collection="demo"
            manual
            title="Demo Kaydı"
            intro="Demo seti için kayıt topluyoruz. Her kelime sırayla gösterilir; kırmızı olanlar kasıtlı yanlış telaffuz ister. Mikrofon tuşuna basılı tutarak konuş, bırakınca kayıt biter ve dinlersin. Değerlendirme yapılmaz — yalnızca kayıt toplanır."
          />
        </main>
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="min-h-screen bg-canvas">
        <main className="mx-auto w-full max-w-3xl px-4 py-5 sm:py-8">
          <div className="mb-5 flex items-center justify-between">
            <div className="flex items-center gap-2 font-serif text-[18px] font-semibold text-brand">
              <Mic size={20} />
              Telaffuz YZ
            </div>
            <a className="text-[12px] font-medium text-muted hover:text-ink" href="/admin">
              Admin
            </a>
          </div>
          <Learner />
        </main>
      </div>
    );
  }

  if (!adminPassword) {
    return <AdminLogin onLogin={setAdminPassword} />;
  }

  function logout() {
    window.sessionStorage.removeItem("telaffuz_admin_password");
    setAdminPassword("");
  }

  return (
    <div className="flex min-h-screen flex-col bg-canvas">
      <header className="sticky top-0 z-20 hidden border-b border-line bg-surface/95 backdrop-blur-sm sm:block">
        <div className="mx-auto flex max-w-5xl items-center gap-1 px-4 py-2">
          <button
            className="mr-3 font-serif text-[17px] font-semibold text-brand"
            onClick={() => setPage("home")}
          >
            Telaffuz YZ Admin
          </button>
          {ADMIN_NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => setPage(item.id)}
                className={clsx(
                  "flex items-center gap-1.5 rounded-lg px-3 py-2 text-[13px] font-medium transition",
                  page === item.id
                    ? "bg-brand-soft text-brand"
                    : "text-muted hover:bg-canvas hover:text-ink"
                )}
              >
                <Icon size={15} />
                {item.label}
              </button>
            );
          })}
          <button className="ml-auto btn-sm" onClick={logout}>
            <LogOut size={14} /> Çıkış
          </button>
        </div>
      </header>

      <div className="sticky top-0 z-20 flex items-center justify-between border-b border-line bg-surface/95 px-4 py-3 backdrop-blur-sm sm:hidden">
        <span className="font-serif text-[16px] font-semibold text-ink">{PAGE_TITLE[page]}</span>
        <button className="text-muted" onClick={logout} title="Çıkış">
          <LogOut size={18} />
        </button>
      </div>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-5 pb-24 sm:pb-8">
        {page === "home" && <Landing onNavigate={navigateFromAdminLanding} />}
        {page === "recordings" && <AdminRecordings password={adminPassword} />}
        {page === "g2p" && <G2P />}
        {page === "sandbox" && <Sandbox />}
        {page === "studio" && <Studio />}
        {page === "settings" && <Settings />}
      </main>

      <nav className="fixed bottom-0 left-0 right-0 z-20 flex border-t border-line bg-surface/95 pb-[env(safe-area-inset-bottom)] backdrop-blur-sm sm:hidden">
        {ADMIN_NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = page === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              className={clsx("nav-tab flex-1", active && "nav-tab-active")}
            >
              <Icon size={20} strokeWidth={active ? 2.2 : 1.8} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>
    </div>
  );

  function navigateFromAdminLanding(next: Page) {
    if (next === "learner") {
      window.location.href = "/";
      return;
    }
    if (next === "home") {
      setPage("home");
      return;
    }
    setPage(next);
  }
}

function AdminLogin({ onLogin }: { onLogin: (password: string) => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      await adminLogin(password);
      window.sessionStorage.setItem("telaffuz_admin_password", password);
      onLogin(password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Giriş başarısız");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid min-h-screen place-items-center bg-canvas px-4">
      <form className="card w-full max-w-[380px]" onSubmit={submit}>
        <div className="eyebrow text-brand">Admin</div>
        <h1 className="title mt-3">Panel girişi</h1>
        <input
          className="field mt-5"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Şifre"
          autoFocus
        />
        {error && <p className="mt-3 text-[13px] text-bad">{error}</p>}
        <button className="btn-primary mt-4" disabled={!password || loading}>
          Giriş
        </button>
      </form>
    </div>
  );
}

export function App() {
  return (
    <SettingsProvider>
      <AppInner />
    </SettingsProvider>
  );
}
