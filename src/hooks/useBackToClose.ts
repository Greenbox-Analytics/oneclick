import { useEffect } from "react";

/**
 * Makes the browser/hardware Back button close an open overlay (sheet, drawer,
 * detail panel) instead of leaving the page. On Android especially, Back is
 * the reflex for "dismiss this" — without it, opening the nav drawer and
 * pressing Back navigates away from the page underneath.
 *
 * How: while the overlay is open we park one extra history entry, reusing the
 * CURRENT entry's state object so React Router's `idx` is unchanged. RR
 * computes its transitions from that index, so a pop between two entries
 * sharing an index is a no-op to the router (delta 0) — the pop reaches only
 * our listener, which closes the overlay. Nothing re-renders, nothing
 * navigates.
 *
 * Closing via the UI instead unwinds the parked entry — but only if it is
 * still the current one. If the overlay closed *because* something navigated
 * (tapping a nav link), the current entry is the new page and calling back()
 * would undo that navigation; the guard skips it. The parked entry is then
 * left mid-stack pointing at the same URL as the page it was opened on, so
 * Back from the new page still lands exactly where the user expects.
 *
 * @param open  Whether the overlay is currently open.
 * @param onClose  Called when Back is pressed while it's open.
 */
export function useBackToClose(open: boolean, onClose: () => void) {
  useEffect(() => {
    if (!open) return;

    const state = window.history.state as Record<string, unknown> | null;
    window.history.pushState({ ...state, __overlay: true }, "");

    const handlePop = () => onClose();
    window.addEventListener("popstate", handlePop);

    return () => {
      window.removeEventListener("popstate", handlePop);
      const current = window.history.state as { __overlay?: boolean } | null;
      if (current?.__overlay) window.history.back();
    };
    // `onClose` is intentionally not a dep: overlays commonly pass an inline
    // arrow, and re-running would re-park an entry on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);
}
