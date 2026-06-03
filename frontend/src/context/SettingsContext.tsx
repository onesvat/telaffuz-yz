import { createContext, useContext, useState, type ReactNode } from "react";

export type AppSettings = {
  modelId: string;
  silenceMs: number;
  pedagogical: boolean;
  useReference: boolean;
};

const DEFAULTS: AppSettings = {
  modelId: "mms-1b",
  silenceMs: 1100,
  pedagogical: true,
  useReference: true,
};

const LS_KEY = "telaffuz.settings";
const SETTINGS_VERSION = 3;

type StoredSettings = Partial<AppSettings> & { version?: number };

function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return DEFAULTS;
    const stored = JSON.parse(raw) as StoredSettings;
    const merged = { ...DEFAULTS, ...stored };
    if (!merged.modelId) merged.modelId = DEFAULTS.modelId;
    return merged;
  } catch {
    return DEFAULTS;
  }
}

type Ctx = {
  settings: AppSettings;
  update: (patch: Partial<AppSettings>) => void;
};

const SettingsContext = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(loadSettings);

  function update(patch: Partial<AppSettings>) {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      localStorage.setItem(LS_KEY, JSON.stringify({ ...next, version: SETTINGS_VERSION }));
      return next;
    });
  }

  return (
    <SettingsContext.Provider value={{ settings, update }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): Ctx {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error("useSettings outside SettingsProvider");
  return ctx;
}
