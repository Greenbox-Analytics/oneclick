/**
 * Durable per-user "onboarding is complete" flag in localStorage.
 *
 * Why this exists:
 *   Multiple components (ProtectedRoute, useOnboardingStatus, Onboarding's
 *   loadProfile) independently query Supabase for `profiles.onboarding_completed`
 *   on every route change. Any one of them seeing a transient `false` (race
 *   after the post-onboarding upsert, network blip, brief auth-token
 *   reshuffle, RLS edge case) would render <Navigate to="/onboarding">, which
 *   would in turn navigate back to /dashboard — the ping-pong that trips
 *   Chromium's "Throttling navigation to prevent the browser from hanging"
 *   protection.
 *
 *   Once the user has finished onboarding once, they have finished it forever
 *   from this browser's perspective. We persist that fact locally so guards
 *   can trust an in-memory signal instead of re-querying the database every
 *   route mount.
 *
 *   Scoped per user.id so a session swap on the same browser doesn't leak
 *   one user's "completed" state to another.
 */

const PREFIX = "msanii_onboarded.";

export function isOnboardedCached(userId: string | null | undefined): boolean {
  if (!userId) return false;
  try {
    return localStorage.getItem(`${PREFIX}${userId}`) === "1";
  } catch {
    return false;
  }
}

export function markOnboardedCached(userId: string | null | undefined): void {
  if (!userId) return;
  try {
    localStorage.setItem(`${PREFIX}${userId}`, "1");
  } catch {
    /* localStorage unavailable — non-fatal, the DB query will still work */
  }
}
