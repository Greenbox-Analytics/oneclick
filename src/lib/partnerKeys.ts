// src/lib/partnerKeys.ts
// Pure logic for OrgApiKeysPanel, kept out of the component so it unit-tests
// without a DOM.
import { addDays, addMonths, addYears, format } from "date-fns";
import type { PartnerKey } from "@/hooks/usePartnerKeys";
import type { PartnerKeyUsageRow } from "@/hooks/useOrgs";

export type KeyStatus = "active" | "revoked" | "expired";

/** Stored status, except an ACTIVE key past its expiry reads "expired" — the
 * server never sends that, since a stored state would drift from the clock.
 * Revoked wins. */
export function keyStatus(key: Pick<PartnerKey, "status" | "expires_at">, now: Date = new Date()): KeyStatus {
  if (key.status === "revoked") return "revoked";
  if (key.expires_at && new Date(key.expires_at).getTime() <= now.getTime()) return "expired";
  return "active";
}

/** "2026-09-30" -> the LAST second of that local day, as UTC. The backend
 * checks `expires_at <= now()`, so a bare date would kill the key as the day
 * BEGINS. A date-time without an offset parses as local (ES spec); a bare date
 * would parse as UTC, which is the bug being avoided. */
export function endOfLocalDayIso(dateInput: string): string {
  return new Date(`${dateInput}T23:59:59`).toISOString();
}

/** yyyy-mm-dd of a Date in local time — date-fns `format` is local by design,
 * which is what a <input type="date"> value has to be. */
export const toInputValue = (d: Date): string => format(d, "yyyy-MM-dd");

/** yyyy-mm-dd of tomorrow in local time — the date input's `min`. */
export const tomorrowInputValue = (now: Date = new Date()): string => toInputValue(addDays(now, 1));

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

// How far out each duration preset lands. addMonths/addYears clamp month-end
// overflow (Jan 31 + 1mo = Feb 28/29); "never" and "custom" have no computed
// date, so they are absent and the caller resolves them.
const PRESET_OFFSET: Partial<Record<ExpiryPreset, (now: Date) => Date>> = {
  "7d": (now) => addDays(now, 7),
  "1m": (now) => addMonths(now, 1),
  "3m": (now) => addMonths(now, 3),
  "6m": (now) => addMonths(now, 6),
  "1y": (now) => addYears(now, 1),
};

/** yyyy-mm-dd (local) a duration preset resolves to, counting from `now`. */
export function expiryFromPreset(preset: ExpiryPreset, now: Date = new Date()): string | null {
  const offset = PRESET_OFFSET[preset];
  return offset ? toInputValue(offset(now)) : null;
}

/** Copy under the expiry picker. Parses at noon local to dodge day-shifting. */
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

/** Keys with status and this period's spend joined on; live first, newest first. */
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
