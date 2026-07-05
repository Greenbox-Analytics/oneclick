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
    // In CI, also emit JUnit XML so dorny/test-reporter can render a per-test
    // check run on the PR (see .github/workflows/ci.yml). Local runs stay clean.
    reporters: process.env.CI ? ["default", "junit"] : ["default"],
    outputFile: { junit: "./junit.xml" },
  },
});
