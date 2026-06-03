import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        serif: ["'Source Serif 4'", "Georgia", "serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        canvas: "#f3f5f8",
        surface: "#ffffff",
        line: "#e2e6ee",
        ink: "#1b2233",
        muted: "#697086",
        brand: "#2d4b8e",
        "brand-soft": "#e9eff9",
        ok: "#1a7a44",
        "ok-soft": "#e6f3ec",
        bad: "#bf3349",
        "bad-soft": "#fae9ec",
        warn: "#a76410",
        "warn-soft": "#fbf0db",
        extra: "#5750ad",
        "extra-soft": "#ecebf8",
      },
      boxShadow: {
        card: "0 1px 2px rgba(27,34,51,0.04), 0 14px 38px rgba(27,34,51,0.07)",
        soft: "0 1px 2px rgba(27,34,51,0.05)",
      },
      keyframes: {
        "pulse-ring": {
          "0%": { transform: "scale(0.92)", opacity: "0.55" },
          "70%": { transform: "scale(1.25)", opacity: "0" },
          "100%": { transform: "scale(1.25)", opacity: "0" },
        },
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      animation: {
        "pulse-ring": "pulse-ring 1.5s cubic-bezier(0.22,1,0.36,1) infinite",
        "fade-up": "fade-up 0.28s ease-out both",
        "fade-in": "fade-in 0.2s ease-out both",
      },
    },
  },
  plugins: [],
} satisfies Config;
