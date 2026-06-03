// Fallback speaker instructions per error rule, used only when a prompt row
// has no explicit speaker_instruction_tr.

export const ERROR_INSTRUCTIONS_TR: Record<string, string> = {
  arabic_persian_long_vowel_shortened:
    "Arapça/Farsça kökenli olup yazılışta şapka (^) olmayan ama uzun okunması gereken a/u/i sesini KISA okuyun.",
  back_for_front: "Ön ünlüyü (i/e/ö/ü) art ünlü (ı/a/o/u) gibi telaffuz edin.",
  ch_to_sh: "Ç sesini Ş olarak telaffuz edin.",
  clear_l_to_dark_l: "İnce (clear) L sesini kalın (dark) L olarak telaffuz edin.",
  c_to_ch: "C sesini Ç olarak telaffuz edin.",
  c_to_j: "C sesini J olarak telaffuz edin.",
  e_for_ae: "Açık e (æ) sesini kapalı e olarak telaffuz edin.",
  final_z_to_s: "Kelime sonundaki Z sesini S olarak telaffuz edin.",
  front_rounded_to_back:
    "Ön yuvarlak ünlüleri (ö/ü) düz art ünlü (o/u) olarak telaffuz edin.",
  h_kept: "Düşmesi veya yutulması gereken H sesini açıkça vurgulayarak okuyun.",
  j_to_c: "J sesini C olarak telaffuz edin.",
  k_to_g: "K sesini G olarak telaffuz edin.",
  loanword_softened: "Yabancı kökenli olup sert kalması gereken ünsüzü yumuşatarak okuyun.",
  long_vowel_shortened: "Uzun okunması gereken ünlü harfi KISA okuyun.",
  n_kept: "Düşmesi gereken N sesini düşürmeden okuyun.",
  no_narrowing: "Daralma (a→ı, e→i) olması gereken yerde daraltma YAPMADAN okuyun.",
  palatal_to_velar: "İnce (palatal) k/g/l seslerini kalın (velar) olarak telaffuz edin.",
  sh_to_s: "Ş sesini S olarak telaffuz edin.",
  soft_g_realized_as_g: "Yumuşak G (ğ) sesini normal G gibi sert okuyun.",
  soft_g_to_g: "Yumuşak G (ğ) sesini normal G gibi okuyun.",
  soft_g_to_v: "Yumuşak G (ğ) sesini V olarak telaffuz edin.",
  s_to_z: "S sesini Z olarak telaffuz edin.",
  stress_misplaced: "Vurguyu yanlış heceye koyarak okuyun.",
  stress_misplaced_final: "Vurguyu yanlışlıkla son heceye koyarak okuyun.",
  stress_misplaced_initial: "Vurguyu yanlışlıkla ilk heceye koyarak okuyun.",
  velar_for_palatal: "İnce (palatal) k/g/l seslerini kalın (velar) olarak telaffuz edin.",
  velar_to_palatal: "Kalın (velar) k/g/l seslerini ince (palatal) olarak telaffuz edin.",
  vowel_e_to_ε: "Kapalı e sesini açık e (ε) olarak telaffuz edin.",
  vowel_ε_to_e: "Açık e (ε) sesini kapalı e olarak telaffuz edin.",
  vowel_ı_to_i: "I sesini İ olarak telaffuz edin.",
  vowel_ö_to_o: "Ö sesini O olarak telaffuz edin.",
  vowel_ü_to_u: "Ü sesini U olarak telaffuz edin.",
  y_kept: "Düşmesi veya kaynaşması gereken Y sesini belirgin okuyun.",
  z_to_s: "Z sesini S olarak telaffuz edin.",
};

export function speakerInstructionForRule(ruleId: string | undefined | null): string {
  if (!ruleId) return "";
  return ERROR_INSTRUCTIONS_TR[ruleId] ?? "";
}

export function isErrorPrompt(prompt: {
  read_mode?: string;
  intended_error?: string;
}): boolean {
  const readMode = (prompt.read_mode || "").toLowerCase();
  const intended = (prompt.intended_error || "").toLowerCase();
  return readMode === "error" || (intended !== "" && intended !== "none");
}
