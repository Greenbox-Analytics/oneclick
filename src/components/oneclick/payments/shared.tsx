// src/components/oneclick/payments/shared.tsx
import { useState, useRef, useEffect } from "react";
import { Check, ChevronDown, Globe, Hourglass, Clock, CheckCheck } from "lucide-react";
import { cn } from "@/lib/utils";
import { getAuthHeaders } from "@/lib/apiFetch";

// ---------------------------------------------------------------------------
// Authenticated PDF download (backend StreamingResponse → browser download)
// ---------------------------------------------------------------------------

export async function downloadPdf(url: string, filename: string): Promise<void> {
  const headers = await getAuthHeaders();
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(objectUrl);
}

// ---------------------------------------------------------------------------
// Currency metadata (symbol/flag/name only — NO toUSD conversion rates).
// Amounts arriving from the server are already converted; this map is used
// purely for display formatting and the currency selector UI.
// ---------------------------------------------------------------------------

export type CurrencyCode = string; // open-ended; server may return any ISO code

export interface CurrencyMeta {
  code: string;
  symbol: string;
  name: string;
  flag: string;
}

export const CURRENCIES: Record<string, CurrencyMeta> = {
  USD: { code: "USD", symbol: "$",   name: "US Dollar",         flag: "🇺🇸" },
  GBP: { code: "GBP", symbol: "£",   name: "British Pound",     flag: "🇬🇧" },
  EUR: { code: "EUR", symbol: "€",   name: "Euro",              flag: "🇪🇺" },
  CAD: { code: "CAD", symbol: "C$",  name: "Canadian Dollar",   flag: "🇨🇦" },
  AUD: { code: "AUD", symbol: "A$",  name: "Australian Dollar", flag: "🇦🇺" },
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

export function fmtMoney(amount: number, cur: string, opts?: { dp?: number }): string {
  const c = CURRENCIES[cur];
  const symbol = c?.symbol ?? cur + " ";
  const defaultDp = 2;
  const dp = opts?.dp != null ? opts.dp : defaultDp;
  const n = Math.abs(amount).toLocaleString("en-US", {
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });
  return (amount < 0 ? "−" : "") + symbol + n;
}

export const money = fmtMoney;

export function fmtDate(iso: string | null, opts?: Intl.DateTimeFormatOptions) {
  if (!iso) return "—";
  return new Date(iso + "T00:00:00").toLocaleDateString("en-US", opts ?? { year: "numeric", month: "short", day: "numeric" });
}

export const initials = (name: string) =>
  name.split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();

// ---------------------------------------------------------------------------
// Stable color derived from an arbitrary string id (no static party map).
// Uses a simple djb2-style hash → hue in HSL.
// ---------------------------------------------------------------------------

const AVATAR_SATURATION = 58;
const AVATAR_LIGHTNESS  = 42;

export function idToHue(id: string): number {
  let h = 5381;
  for (let i = 0; i < id.length; i++) h = ((h << 5) + h) ^ id.charCodeAt(i);
  return ((h >>> 0) % 360);
}

export function idToColor(id: string): string {
  return `hsl(${idToHue(id)}, ${AVATAR_SATURATION}%, ${AVATAR_LIGHTNESS}%)`;
}

// ---------------------------------------------------------------------------
// PartyAvatar — derives stable color from id, no longer reads paymentsData
// ---------------------------------------------------------------------------

export function PartyAvatar({ id, name, size = 30 }: { id: string; name?: string; size?: number }) {
  const label = name ?? id;
  return (
    <span
      className="inline-flex items-center justify-center rounded-full font-semibold text-white shrink-0"
      style={{ background: idToColor(id), width: size, height: size, fontSize: size * 0.4 }}
      title={label}
    >
      {initials(label)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Amt / showAmt — amounts already converted by server; no client FX
// ---------------------------------------------------------------------------

export type Display = "base" | "native";

/**
 * Format an already-converted amount in `curCode`.
 *
 * New signature (Task 14+):
 *   showAmt(amount, curCode, opts?)
 *
 * Legacy 4-arg signature accepted for backward compat while Task-15/16 modal
 * files are still using the old calling convention (will be removed when those
 * are rewritten):
 *   showAmt(amount, curCode, display, base)
 *
 * In both cases NO client-side FX is applied — amounts are assumed to be
 * already in the target currency.
 */
export function showAmt(
  amount: number,
  curCode: string,
  optsOrDisplay?: { nativeAmount?: number; nativeCur?: string } | string,
  _base?: string, // legacy 4th arg — ignored; no client FX
): { main: string; sub: string | null } {
  const main = fmtMoney(amount, curCode);
  let sub: string | null = null;
  if (optsOrDisplay && typeof optsOrDisplay === "object") {
    if (optsOrDisplay.nativeAmount != null && optsOrDisplay.nativeCur && optsOrDisplay.nativeCur !== curCode) {
      sub = fmtMoney(optsOrDisplay.nativeAmount, optsOrDisplay.nativeCur);
    }
  }
  // If called with the legacy (display, base) args, sub is simply null — the
  // caller will be rewritten in Task 15/16.
  return { main, sub };
}

export function Amt(props: {
  amount: number;
  cur: string;
  /** Optional native (payout-currency) sub-line */
  nativeAmount?: number;
  nativeCur?: string;
  className?: string;
  subClassName?: string;
}) {
  const { main, sub } = showAmt(props.amount, props.cur, {
    nativeAmount: props.nativeAmount,
    nativeCur: props.nativeCur,
  });
  return (
    <div>
      <div className={cn("font-mono tabular-nums", props.className)}>{main}</div>
      {sub && <div className={cn("text-xs text-muted-foreground font-mono", props.subClassName)}>{sub}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// StatusBadge
// ---------------------------------------------------------------------------

export function StatusBadge({ kind, children }: { kind: "paid" | "sched" | "out" | "settled" | "partial" | "owed" | "scheduled"; children: React.ReactNode }) {
  // normalise server status strings to badge kind
  const normalised: Record<string, string> = { owed: "out", scheduled: "sched", settled: "settled" };
  const k = normalised[kind] ?? kind;
  const tone: Record<string, string> = {
    paid:    "bg-[hsl(var(--pay-paid-bg))] text-[hsl(var(--pay-paid-fg))]",
    sched:   "bg-[hsl(var(--pay-sched-bg))] text-[hsl(var(--pay-sched-fg))]",
    out:     "bg-[hsl(var(--pay-out-bg))] text-[hsl(var(--pay-out-fg))]",
    partial: "bg-[hsl(var(--pay-partial-bg))] text-[hsl(var(--pay-partial-fg))]",
    settled: "bg-muted text-muted-foreground",
  };
  return (
    <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold whitespace-nowrap", tone[k] ?? tone.settled)}>
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// CurrencySelect — reporting (base) currency dropdown
// ---------------------------------------------------------------------------

export function CurrencySelect({ value, onChange }: { value: string; onChange: (c: string) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const k = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", h);
    document.addEventListener("keydown", k);
    return () => { document.removeEventListener("mousedown", h); document.removeEventListener("keydown", k); };
  }, []);
  const c = CURRENCIES[value] ?? { flag: "🌐", code: value, name: value };
  return (
    <div className="relative shrink-0" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title="Reporting currency"
        className="inline-flex h-10 items-center gap-2 rounded-lg border border-input bg-card px-3 hover:bg-muted"
      >
        <Globe className="h-[15px] w-[15px] text-muted-foreground" />
        <span className="text-[15px]">{c.flag}</span>
        <span className="text-sm font-bold tabular-nums">{c.code}</span>
        <ChevronDown className="h-3.5 w-3.5 opacity-60" />
      </button>
      {open && (
        <div className="absolute right-0 top-[calc(100%+6px)] z-[70] w-60 rounded-xl border border-border bg-card p-1.5 shadow-lg">
          <div className="px-2 py-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Show all amounts in</div>
          {Object.values(CURRENCIES).map((o) => (
            <button
              key={o.code}
              type="button"
              onClick={() => { onChange(o.code); setOpen(false); }}
              className={cn("flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left hover:bg-muted", o.code === value && "bg-secondary/70")}
            >
              <span className="text-base">{o.flag}</span>
              <span className="flex flex-1 flex-col">
                <span className="text-[13px] font-bold">{o.code}</span>
                <span className="text-[11.5px] text-muted-foreground">{o.name}</span>
              </span>
              {o.code === value && <Check className="h-[15px] w-[15px] text-primary" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// partyStatusKind / STATUS_ICON — used by tables
// Maps server `status` strings ("owed"|"scheduled"|"settled") to badge kinds.
// ---------------------------------------------------------------------------

export function partyStatusKind(status: string): "out" | "sched" | "settled" {
  if (status === "owed") return "out";
  if (status === "scheduled") return "sched";
  return "settled";
}

export const STATUS_ICON = { out: Hourglass, sched: Clock, settled: CheckCheck } as const;
