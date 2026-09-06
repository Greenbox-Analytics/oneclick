// src/lib/partnerKeys.ts
// Pure logic for OrgApiKeysPanel — kept out of the component so it can be
// unit-tested without a DOM (partner-keys-helpers.test.ts).
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

/** yyyy-mm-dd of tomorrow in local time — the date input's `min`. */
export function tomorrowInputValue(now: Date = new Date()): string {
  const t = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1);
  const mm = String(t.getMonth() + 1).padStart(2, "0");
  const dd = String(t.getDate()).padStart(2, "0");
  return `${t.getFullYear()}-${mm}-${dd}`;
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
