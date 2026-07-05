import { defineConfig } from "vitest/config";
import path from "path";

// Standalone Vitest config (separate from vite.config.ts). Tests here exercise
// data-layer hooks against a real QueryClient — no app plugins needed.
export default defineConfig({
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.{test,spec}.{ts,tsx}"],
    restoreMocks: true,
    unstubGlobals: true,
  },
});
