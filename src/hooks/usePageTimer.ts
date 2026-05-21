import { useEffect, useRef } from "react";
import { useLocation } from "react-router-dom";
import { capture } from "@/lib/posthog";

/**
 * Track dwell time per route. Fires `page_time_spent` events on:
 *  - route change (delta from previous-route arrival)
 *  - page hide (pagehide event or visibilitychange → hidden)
 *
 * Mount once near the top of the routed tree (inside <Router>).
 */
export function usePageTimer(): void {
  const { pathname } = useLocation();
  const enteredAtRef = useRef<number>(performance.now());
  const currentPathRef = useRef<string>(pathname);

  // Fire on route change: capture time spent on the PREVIOUS route, then reset.
  useEffect(() => {
    const now = performance.now();
    const prevPath = currentPathRef.current;
    if (prevPath !== pathname) {
      const duration_ms = Math.round(now - enteredAtRef.current);
      capture("page_time_spent", { $pathname: prevPath, duration_ms });
      enteredAtRef.current = now;
      currentPathRef.current = pathname;
    }
  }, [pathname]);

  // Fire on page hide / tab switch.
  useEffect(() => {
    const flush = () => {
      const now = performance.now();
      const duration_ms = Math.round(now - enteredAtRef.current);
      if (duration_ms > 100) {
        // Skip noise from immediate hide-on-open (e.g., bot prefetches)
        capture("page_time_spent", { $pathname: currentPathRef.current, duration_ms });
      }
      // Reset so a subsequent visibilitychange→visible doesn't double-count
      enteredAtRef.current = now;
    };

    const handleVisibility = () => {
      if (document.visibilityState === "hidden") flush();
    };

    window.addEventListener("pagehide", flush);
    document.addEventListener("visibilitychange", handleVisibility);

    return () => {
      window.removeEventListener("pagehide", flush);
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  }, []);
}
