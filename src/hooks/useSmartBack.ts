import { useCallback } from "react";
import { useLocation, useNavigate } from "react-router-dom";

/**
 * Returns a handler that takes the user back to the last page they visited,
 * mirroring the browser's Back button (`navigate(-1)`).
 *
 * When there is no previous in-app page to return to — the user opened this
 * page directly via a shared/deep link, a fresh tab, or a hard refresh —
 * `navigate(-1)` would either do nothing or bounce them out of the app, so we
 * fall back to a sensible route instead.
 *
 * React Router assigns every navigation a unique `location.key`; only the very
 * first entry in the history stack gets the literal key `"default"`. That makes
 * it a reliable signal for "is there anywhere to go back to within the app?".
 *
 * @param fallback Route to use when there's no in-app history. Defaults to
 *   the dashboard.
 */
export function useSmartBack(fallback: string = "/dashboard") {
  const navigate = useNavigate();
  const location = useLocation();

  return useCallback(() => {
    if (location.key !== "default") {
      navigate(-1);
    } else {
      navigate(fallback);
    }
  }, [navigate, location.key, fallback]);
}
