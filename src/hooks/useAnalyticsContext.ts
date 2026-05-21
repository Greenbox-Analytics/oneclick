import { useEffect, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { apiFetch, API_URL } from "@/lib/apiFetch";
import { identifyUser } from "@/lib/posthog";

const CACHE_KEY = "msanii.analytics_context.v1";
const TTL_MS = 5 * 60 * 1000;
const CONTEXT_UPDATED_EVENT = "msanii:analytics-context-updated";

/**
 * Window event fired immediately after the analytics-context cache is
 * (over)written. Reactive subscribers (e.g. useTesterStatus) listen for this
 * to re-render the instant the bootstrap POST refreshes the cache — no page
 * reload required. NOT fired on clearCache (anti-flicker — see comment there).
 */
export const ANALYTICS_CONTEXT_UPDATED_EVENT = CONTEXT_UPDATED_EVENT;

export interface AnalyticsContext {
  is_tester: boolean;
  is_admin: boolean; // server-derived (don't leak ADMIN_EMAILS into the JS bundle)
  plan: string; // "free" | "pro"
  role: string | null;
  email: string | null;
  signed_up_at: string | null;
  tester_granted_at: string | null;
  tester_expires_at: string | null;
}

interface CacheEntry {
  user_id: string;
  fetched_at: number;
  ctx: AnalyticsContext;
}

function readCache(userId: string): CacheEntry | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const entry: CacheEntry = JSON.parse(raw);
    if (entry.user_id !== userId) return null;
    return entry;
  } catch {
    return null;
  }
}

/**
 * Read the cached analytics context for a user without triggering a fetch.
 * Returns null when no cache exists yet (first login of the session) OR when
 * the cache belongs to a different user. Useful for hooks that want to fast-
 * path on `is_admin === false` to skip admin-only network calls.
 *
 * Does NOT respect TTL — returns whatever's there. Callers that need
 * freshness should rely on useAnalyticsContext to refresh the cache.
 */
export function peekCachedAnalyticsContext(userId: string | undefined | null): AnalyticsContext | null {
  if (!userId) return null;
  const entry = readCache(userId);
  return entry?.ctx ?? null;
}

function writeCache(entry: CacheEntry): void {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(entry));
  } catch {
    /* localStorage may be unavailable / full; ignore */
  }
  try {
    window.dispatchEvent(new Event(CONTEXT_UPDATED_EVENT));
  } catch {
    /* SSR / no-window environments — ignore */
  }
}

function clearCache(): void {
  try {
    localStorage.removeItem(CACHE_KEY);
  } catch {
    /* ignore */
  }
  // INTENTIONALLY no dispatch here. clearCache is called from two places:
  //   1) Sign-out — AuthContext re-renders with user=null, making
  //      useTesterStatus's userId undefined and forcing isTester=false on
  //      the next render. The banner hides correctly without our event.
  //   2) Inside refreshAnalyticsContext, immediately before the new fetch.
  //      Firing here would briefly publish "no tester" (peek returns null),
  //      flicker the banner off, then writeCache would flicker it back on.
  //      Skip the intermediate event; writeCache is the only authoritative
  //      "context updated" signal.
}

/**
 * Force-refresh the cached analytics context. Call after a successful Stripe
 * checkout so the cached `plan: "free"` is replaced with `plan: "pro"` before
 * any consumer (e.g. the dashboard UpgradeBanner) re-renders against the
 * stale value. Safe to call without an active mount of useAnalyticsContext.
 */
export async function refreshAnalyticsContext(
  userId: string,
  authEmail: string | null | undefined,
): Promise<void> {
  clearCache();
  const controller = new AbortController();
  await fetchAndIdentify(userId, authEmail, controller.signal);
}

async function fetchAndIdentify(
  userId: string,
  authEmail: string | null | undefined,
  signal: AbortSignal,
): Promise<void> {
  try {
    const ctx = await apiFetch<AnalyticsContext>(`${API_URL}/me/analytics-context`, { signal });
    if (signal.aborted) return;
    writeCache({ user_id: userId, fetched_at: Date.now(), ctx });
    identifyUser(userId, {
      email: ctx.email ?? authEmail ?? undefined,
      signed_up_at: ctx.signed_up_at,
      plan: ctx.plan,
      role: ctx.role,
      is_tester: ctx.is_tester,
      is_admin: ctx.is_admin,
      tester_granted_at: ctx.tester_granted_at,
      tester_expires_at: ctx.tester_expires_at,
    });
  } catch (err) {
    if ((err as { name?: string })?.name === "AbortError") return;
    console.warn("Failed to fetch analytics context:", err);
  }
}

/**
 * Side-effect-only hook: ensures PostHog is identified with the latest
 * person properties for the current user, and refreshes when stale or on
 * window focus. Returns nothing — consumers don't need the context object;
 * identify() is the entire payoff.
 */
export function useAnalyticsContext(): void {
  const { user } = useAuth();
  const lastUserIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!user) {
      clearCache();
      lastUserIdRef.current = null;
      return;
    }

    const userId = user.id;
    const authEmail = user.email;
    const controller = new AbortController();

    const cached = readCache(userId);
    const isStale = !cached || Date.now() - cached.fetched_at > TTL_MS;

    if (isStale || lastUserIdRef.current !== userId) {
      fetchAndIdentify(userId, authEmail, controller.signal);
    } else if (cached) {
      // Cache hit: re-emit identify so PostHog has the properties this session.
      identifyUser(userId, {
        email: cached.ctx.email ?? authEmail ?? undefined,
        signed_up_at: cached.ctx.signed_up_at,
        plan: cached.ctx.plan,
        role: cached.ctx.role,
        is_tester: cached.ctx.is_tester,
        is_admin: cached.ctx.is_admin,
        tester_granted_at: cached.ctx.tester_granted_at,
        tester_expires_at: cached.ctx.tester_expires_at,
      });
    }
    lastUserIdRef.current = userId;

    const onFocus = () => {
      const c = readCache(userId);
      if (!c || Date.now() - c.fetched_at > TTL_MS) {
        fetchAndIdentify(userId, authEmail, controller.signal);
      }
    };
    window.addEventListener("focus", onFocus);
    return () => {
      controller.abort();
      window.removeEventListener("focus", onFocus);
    };
  }, [user?.id, user?.email]);
}
