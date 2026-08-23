import { useCallback } from "react";
import { useNavigate } from "react-router-dom";

/**
 * Returns a handler that takes the user back to the last page they visited,
 * mirroring the browser's Back button (`navigate(-1)`).
 *
 * When there is no previous in-app page to return to — the user opened this
 * page directly via a shared/deep link, a fresh tab, or a hard refresh —
 * `navigate(-1)` would either do nothing or bounce them out of the app, so we
 * fall back to a sensible route instead.
 *
 * The signal is React Router's own history-stack index, which it stores in
 * `window.history.state.idx` (0 for the entry the app was opened on, +1 per
 * push). `location.key` is NOT a substitute: it only reads `"default"` on an
 * untouched first entry, so any `replace` landing on that entry — a deep link
 * bounced through `ProtectedRoute` to `/auth`, a post-login redirect, a
 * legacy-route redirect — mints a fresh key and made Back walk the user out of
 * the app. `idx` survives `replace` and survives a refresh, because it lives
 * in the history entry itself.
 *
 * @param fallback Route to use when there's no in-app history. Defaults to
 *   the dashboard.
 */
export function useSmartBack(fallback: string = "/dashboard") {
  const navigate = useNavigate();

  return useCallback(() => {
    // `idx` is null on the very first render before React Router stamps it,
    // and absent entirely if some other code replaced history state.
    const idx = (window.history.state as { idx?: number } | null)?.idx;
    if (typeof idx === "number" && idx > 0) {
      navigate(-1);
    } else {
      navigate(fallback);
    }
  }, [navigate, fallback]);
}
