// Shared helpers for the royalty-splits UI, used by both the compact
// RoyaltySplitsTable (read-only display) and the EditSplitsDialog (editing).
// Kept in a plain module so neither component file exports non-components.

/** Stable avatar colors for split parties, cycled by row index. */
export const SPLIT_PALETTE = [
  "#7c5cff",
  "#1f8a5b",
  "#d9762b",
  "#d24b6e",
  "#2a6fdb",
  "#0f6b43",
];

/** Coerce free-typed input into an integer percentage in [0, 100]. */
export function clampPct(v: string | number): number {
  const n = parseInt(String(v).replace(/[^0-9]/g, ""), 10);
  if (Number.isNaN(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

/** Sum master and publishing percentages across split rows. */
export function splitTotals(
  rows: Array<{ master?: number; publishing?: number }>
): { master: number; publishing: number } {
  return rows.reduce(
    (acc, r) => ({
      master: acc.master + (r.master || 0),
      publishing: acc.publishing + (r.publishing || 0),
    }),
    { master: 0, publishing: 0 }
  );
}

/**
 * Allowed overage above 100% for a stake-total pool. Absorbs tiny rounding from
 * AI-parsed fractional splits — kept in sync with the backend STAKE_TOTAL_MARGIN
 * (src/backend/registry/service.py) and the validate_stake_total DB trigger.
 */
export const SPLIT_TOTAL_MARGIN = 0.5;

/**
 * Whether master and/or publishing totals exceed 100% by more than `margin`.
 * Mirrors the per-stake_type 100% pools the backend enforces (soundexchange is
 * excluded, consistent with splitTotals).
 */
export function splitsOverCap(
  rows: Array<{ master?: number; publishing?: number }>,
  margin: number = SPLIT_TOTAL_MARGIN
): { master: boolean; publishing: boolean; over: boolean } {
  const totals = splitTotals(rows);
  const master = totals.master > 100 + margin;
  const publishing = totals.publishing > 100 + margin;
  return { master, publishing, over: master || publishing };
}
