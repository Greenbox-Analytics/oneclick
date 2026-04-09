/**
 * Parse a "YYYY-MM-DD" string as local midnight.
 * Avoids the UTC-parsing bug in `new Date("2026-04-09")`.
 */
export function parseDateString(dateStr: string): Date {
  const [y, m, d] = dateStr.split("-").map(Number);
  return new Date(y, m - 1, d);
}

/**
 * Get today's date as "YYYY-MM-DD" in the given IANA timezone.
 * Falls back to the browser's system timezone if none provided.
 */
export function getTodayString(timezone?: string): string {
  const tz = timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
  return new Intl.DateTimeFormat("en-CA", { timeZone: tz }).format(new Date());
}
