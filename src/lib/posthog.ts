import posthog from "posthog-js";

const API_KEY = import.meta.env.VITE_POSTHOG_API_KEY as string | undefined;
const HOST = (import.meta.env.VITE_POSTHOG_HOST as string | undefined) ?? "https://us.i.posthog.com";

let initialized = false;

/**
 * Initialize PostHog. No-op if VITE_POSTHOG_API_KEY is unset (e.g., dev/test).
 * Idempotent — safe to call multiple times.
 */
export function initPostHog(): void {
  if (initialized) return;
  if (!API_KEY) return; // disabled (no key configured)

  posthog.init(API_KEY, {
    api_host: HOST,
    capture_pageview: true,  // autocapture $pageview on route changes
    capture_pageleave: true, // also capture $pageleave for dwell-time approximation
    autocapture: false,      // we don't want every click captured; only our explicit events
    disable_session_recording: true, // privacy-forward default for beta v1
    persistence: "localStorage+cookie",
    person_profiles: "identified_only", // create person profiles only after identify()
  });

  initialized = true;
}

/**
 * Identify the current user. Merges anonymous events under this distinct_id.
 * No-op if PostHog not initialized.
 */
export function identifyUser(userId: string, properties: Record<string, unknown> = {}): void {
  if (!initialized) return;
  posthog.identify(userId, properties);
}

/**
 * Reset the local distinct_id. Call on logout to prevent the next user's
 * events from being attributed to the previous user.
 */
export function resetUser(): void {
  if (!initialized) return;
  posthog.reset();
}

/**
 * Capture an event. No-op if PostHog not initialized.
 * Use the typed wrappers in `useAnalytics` instead of calling this directly
 * from consumers — keeps event names + properties consistent.
 */
export function capture(event: string, properties: Record<string, unknown> = {}): void {
  if (!initialized) return;
  posthog.capture(event, properties);
}
