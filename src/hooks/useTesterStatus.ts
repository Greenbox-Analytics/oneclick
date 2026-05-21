import { useSyncExternalStore } from "react";
import { useAuth } from "@/contexts/AuthContext";
import {
  ANALYTICS_CONTEXT_UPDATED_EVENT,
  peekCachedAnalyticsContext,
} from "@/hooks/useAnalyticsContext";

export interface TesterStatus {
  isTester: boolean;
  grantedAt: string | null;
  expiresAt: string | null;
}

function subscribe(callback: () => void): () => void {
  window.addEventListener(ANALYTICS_CONTEXT_UPDATED_EVENT, callback);
  return () => window.removeEventListener(ANALYTICS_CONTEXT_UPDATED_EVENT, callback);
}

/**
 * Reactive tester-status hook. Subscribes to ANALYTICS_CONTEXT_UPDATED_EVENT
 * so the banner re-renders the moment the bootstrap POST refreshes the cache —
 * no page reload required. Returns a stable JSON snapshot so useSyncExternalStore's
 * referential-equality check doesn't infinite-loop on identical reads.
 */
export function useTesterStatus(): TesterStatus {
  const { user } = useAuth();
  const userId = user?.id;

  const snapshot = useSyncExternalStore(
    subscribe,
    () => {
      const ctx = peekCachedAnalyticsContext(userId);
      return JSON.stringify({
        isTester: ctx?.is_tester ?? false,
        grantedAt: ctx?.tester_granted_at ?? null,
        expiresAt: ctx?.tester_expires_at ?? null,
      });
    },
    () => JSON.stringify({ isTester: false, grantedAt: null, expiresAt: null }), // SSR fallback
  );

  return JSON.parse(snapshot);
}
