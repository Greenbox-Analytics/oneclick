import { useSyncExternalStore } from "react";
import { useAuth } from "@/contexts/AuthContext";
import {
  ANALYTICS_CONTEXT_UPDATED_EVENT,
  peekCachedAnalyticsContext,
} from "@/hooks/useAnalyticsContext";

function subscribe(callback: () => void): () => void {
  window.addEventListener(ANALYTICS_CONTEXT_UPDATED_EVENT, callback);
  return () => window.removeEventListener(ANALYTICS_CONTEXT_UPDATED_EVENT, callback);
}

/**
 * Reactive admin-status hook. Reads is_admin from the cached analytics context
 * (set by /me/analytics-context — covers both env-admins and DB admins).
 * Subscribes to ANALYTICS_CONTEXT_UPDATED_EVENT so the badge appears/disappears
 * immediately after a promote/demote without requiring a page reload.
 *
 * Primitive boolean snapshot — no stringify dance needed.
 */
export function useAdminStatus(): boolean {
  const { user } = useAuth();
  const userId = user?.id;

  return useSyncExternalStore(
    subscribe,
    () => peekCachedAnalyticsContext(userId)?.is_admin === true,
    () => false, // SSR fallback
  );
}
