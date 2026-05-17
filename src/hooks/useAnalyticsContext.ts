import { useEffect, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { apiFetch, API_URL } from "@/lib/apiFetch";
import { identifyUser } from "@/lib/posthog";

const CACHE_KEY = "msanii.analytics_context.v1";
const TTL_MS = 5 * 60 * 1000;

interface AnalyticsContext {
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

function writeCache(entry: CacheEntry): void {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(entry));
  } catch {
    /* localStorage may be unavailable / full; ignore */
  }
}

function clearCache(): void {
  try {
    localStorage.removeItem(CACHE_KEY);
  } catch {
    /* ignore */
  }
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
