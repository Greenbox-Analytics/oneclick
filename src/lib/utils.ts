import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { differenceInCalendarDays } from "date-fns";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/** Format an ISO date as e.g. "Jul 22, 2026". Returns "—" for null/invalid. */
export function fmtDate(iso?: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/**
 * Calendar days from today until an ISO date, as a short label beside the date
 * fmtDate renders: "12 days left", "1 day left", "today". Calendar days (in the
 * viewer's zone, like fmtDate) rather than 24-hour blocks, so the count matches
 * the date shown. Empty for null/invalid dates and once the date has passed.
 */
export function fmtDaysLeft(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const days = differenceInCalendarDays(d, new Date());
  if (days < 0) return "";
  if (days === 0) return "today";
  return days === 1 ? "1 day left" : `${days} days left`;
}

/** Format an ISO date as e.g. "Jul 22" (no year). Returns "" for null/invalid. */
export function fmtDay(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? ""
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Human-readable byte size, e.g. "1.5 MB". Returns "0 B" for falsy/negative. */
export function formatBytes(bytes: number): string {
  if (!bytes || bytes < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}
