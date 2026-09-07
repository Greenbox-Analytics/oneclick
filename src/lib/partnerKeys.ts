// src/lib/partnerKeys.ts
// Pure logic for OrgApiKeysPanel — kept out of the component so it can be
// unit-tested without a DOM (partner-keys-helpers.test.ts).
import { addDays, addMonths, addYears } from "date-fns";
import type { PartnerKey } from "@/hooks/usePartnerKeys";
import type { PartnerKeyUsageRow } from "@/hooks/useOrgs";

export type KeyStatus = "active" | "revoked" | "expired";

/** Stored status, except an ACTIVE key past its expiry reads "expired". The
 * server never sends "expired": the auth path checks `expires_at <= now()`
 * itself and a stored state would drift from the clock. Revoked wins. */
export function keyStatus(key: Pick<PartnerKey, "status" | "expires_at">, now: Date = new Date()): KeyStatus {
  if (key.status === "revoked") return "revoked";
  if (key.expires_at && new Date(key.expires_at).getTime() <= now.getTime()) return "expired";
  return "active";
}

/** "2026-09-30" (from <input type="date">) -> the LAST second of that local
 * day as a UTC ISO string. The backend check is `expires_at <= now()`, so a
 * bare date would kill the key as the day BEGINS. */
export function endOfLocalDayIso(dateInput: string): string {
  // A date-time string WITHOUT an offset parses as local time (ES spec);
  // a bare date would parse as UTC, which is exactly the bug this avoids.
  return new Date(`${dateInput}T23:59:59`).toISOString();
}

/** yyyy-mm-dd of a Date in local time. */
export function toInputValue(d: Date): string {
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${d.getFullYear()}-${mm}-${dd}`;
}

/** yyyy-mm-dd of tomorrow in local time — the date input's `min`. */
export function tomorrowInputValue(now: Date = new Date()): string {
  return toInputValue(new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1));
}

export type ExpiryPreset = "never" | "7d" | "1m" | "3m" | "6m" | "1y" | "custom";

export const EXPIRY_PRESETS: { id: ExpiryPreset; label: string }[] = [
  { id: "never", label: "Never" },
  { id: "7d", label: "7 days" },
  { id: "1m", label: "1 month" },
  { id: "3m", label: "3 months" },
  { id: "6m", label: "6 months" },
  { id: "1y", label: "1 year" },
  { id: "custom", label: "Custom" },
];

/** yyyy-mm-dd (local) a duration preset resolves to, counting from `now`.
 * `addMonths`/`addYears` clamp month-end overflow (Jan 31 + 1mo = Feb 28/29).
 * "never" and "custom" have no computed date — the caller resolves those. */
export function expiryFromPreset(preset: ExpiryPreset, now: Date = new Date()): string | null {
  switch (preset) {
    case "7d":
      return toInputValue(addDays(now, 7));
    case "1m":
      return toInputValue(addMonths(now, 1));
    case "3m":
      return toInputValue(addMonths(now, 3));
    case "6m":
      return toInputValue(addMonths(now, 6));
    case "1y":
      return toInputValue(addYears(now, 1));
    default:
      return null;
  }
}

/** Helper copy under the expiry picker. `dateInput` is yyyy-mm-dd (local) or
 * null for "never expires". Parses at noon local to dodge UTC day-shifting. */
export function expiryLabel(dateInput: string | null): string {
  if (!dateInput) return "Never expires. You can revoke it any time.";
  const formatted = new Date(`${dateInput}T12:00:00`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  return `Stops working at the end of ${formatted}.`;
}

export interface KeyRow {
  key: PartnerKey;
  status: KeyStatus;
  usage?: PartnerKeyUsageRow;
}

/** Keys with their effective status and this period's spend joined on, live
 * before inactive, newest first. */
export function keyRows(keys: PartnerKey[], usage: PartnerKeyUsageRow[] | undefined, now: Date = new Date()): KeyRow[] {
  const usageById = new Map((usage ?? []).map((u) => [u.keyId, u]));
  return keys
    .map((k) => ({ key: k, status: keyStatus(k, now), usage: usageById.get(k.id) }))
    .sort((a, b) => {
      const liveA = a.status === "active" ? 0 : 1;
      const liveB = b.status === "active" ? 0 : 1;
      if (liveA !== liveB) return liveA - liveB;
      return b.key.created_at.localeCompare(a.key.created_at);
    });
}
