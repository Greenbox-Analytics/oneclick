import { useCallback } from "react";
import { useSearchParams } from "react-router-dom";

/**
 * Tabbed-page state that lives in the URL (`?tab=`) rather than in component
 * state, so the tab survives a refresh, a shared link, and a back-navigation
 * into the page. Reading `?tab=` once into `useState` (the old shape) loses it
 * on all three: the URL kept saying `?tab=members` while the page rendered
 * whatever the default tab was.
 *
 * Switching tabs writes with `replace` — a tab is a view of one page, not a
 * new destination, so it must not stack history entries the user then has to
 * press Back through to leave the page.
 *
 * An unrecognised or absent `?tab=` resolves to `fallback` without rewriting
 * the URL, so a stale deep link degrades to the default tab instead of 404ing.
 */
export function useTabParam(validTabs: readonly string[], fallback: string) {
  const [searchParams, setSearchParams] = useSearchParams();
  const raw = searchParams.get("tab") ?? "";
  const activeTab = validTabs.includes(raw) ? raw : fallback;

  const setActiveTab = useCallback(
    (tab: string) => {
      setSearchParams(
        (prev) => {
          // Merge, never replace: `taskId`, `artist` and board filters share
          // this query string.
          const next = new URLSearchParams(prev);
          next.set("tab", tab);
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return [activeTab, setActiveTab] as const;
}
