import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const allowedHosts = (process.env.TELAFFUZ_FRONTEND_ALLOWED_HOSTS ?? "")
  .split(",")
  .map((host) => host.trim())
  .filter(Boolean);

export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts,
    proxy: {
      "/api": `http://127.0.0.1:${process.env.TELAFFUZ_API_PORT ?? "8000"}`,
    },
  },
});
