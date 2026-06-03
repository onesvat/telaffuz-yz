// Turkish phoneme labels for the feedback UI.
// Adapted from user-app/src/phonemeInfo.ts (no logic changes).

export const PHONEME_LABEL_TR: Record<string, string> = {
  a: "a", e: "kapalı e", i: "i", ɯ: "ı (kalın)", o: "o", œ: "ö", u: "u", y: "ü",
  "aː": "uzun a", "eː": "uzun e", "iː": "uzun i", "ɯː": "uzun ı",
  "oː": "uzun o", "œː": "uzun ö", "uː": "uzun u", "yː": "uzun ü",
  æ: "açık e (â/e)",
  p: "p", b: "b", t: "t", d: "d", k: "kalın k", ɡ: "g",
  "t͡ʃ": "ç", "d͡ʒ": "c", f: "f", v: "v", s: "s", z: "z",
  ʃ: "ş", ʒ: "j", h: "h", m: "m", n: "n", l: "ince l",
  ɾ: "tek vuruşlu r", j: "y",
  c: "ince k", ɟ: "ince g", ɲ: "ny (genizsi n)", ŋ: "genizden n (ng)",
  ɫ: "kalın l", β: "yumuşak v", "β̞": "yumuşak v",
  "pʰ": "soluklu p", "tʰ": "soluklu t", "kʰ": "soluklu k", "cʰ": "soluklu ince k",
  "ɾ̞̊": "sessiz r (kelime sonu)",
  "ˈ": "vurgu", ".": "hece sınırı",
};

export function phonemeLabel(symbol: string | null | undefined): string {
  if (!symbol) return "";
  return PHONEME_LABEL_TR[symbol] ?? symbol;
}
