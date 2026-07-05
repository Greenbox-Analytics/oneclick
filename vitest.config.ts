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
    // Fake env so modules that instantiate the Supabase client at import time
    // (via @/lib/apiFetch) don't throw "supabaseUrl is required" in CI, where
    // there's no .env. Values are dummy — tests never make real network calls.
    env: {
      VITE_SUPABASE_URL: "https://fake.supabase.co",
      VITE_SUPABASE_ANON_KEY: "fake-anon-key-for-tests",
      VITE_BACKEND_API_URL: "https://fake-backend.example.com",
    },
    // In CI, also emit JUnit XML so dorny/test-reporter can render a per-test
    // check run on the PR (see .github/workflows/ci.yml). Local runs stay clean.
    reporters: process.env.CI ? ["default", "junit"] : ["default"],
    outputFile: { junit: "./junit.xml" },
  },
});
