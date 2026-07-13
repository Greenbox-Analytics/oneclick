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
