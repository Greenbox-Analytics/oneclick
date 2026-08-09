// src/components/admin/ui.tsx
// Small shared pieces of the admin console shell — tone-coded tags and stat
// tiles. Deliberately not shadcn Badge: the console needs six tones (and a
// square-ish chip), Badge ships four pill variants.
import { cn } from "@/lib/utils";

export type Tone = "neutral" | "paid" | "ok" | "warn" | "bad" | "info";

/** Just the YYYY-MM-DD. Timestamps reach the console in two shapes — PostgREST
 * ISO ("2026-08-04T14:33:19Z") and Python `str(datetime)` ("2026-08-04
 * 14:33:19.909032+00:00", what /admin/users returns) — so slice rather than
 * split on "T", which silently printed the whole timestamp for the latter. */
export const shortDate = (ts: string | null | undefined): string =>
  ts ? ts.slice(0, 10) : "—";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "border-border bg-muted text-muted-foreground",
  paid: "border-primary/25 bg-secondary text-primary",
  ok: "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900 dark:bg-emerald-950 dark:text-emerald-300",
  warn: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-300",
  bad: "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950 dark:text-red-300",
  info: "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-900 dark:bg-blue-950 dark:text-blue-300",
};

export const ORG_STATUS_TONE: Record<string, Tone> = {
  active: "ok",
  pending: "warn",
  suspended: "bad",
  archived: "neutral",
};

export function Tag({
  tone = "neutral",
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[11px] font-semibold",
        TONE_CLASS[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function StatTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <div className="text-[11.5px] font-medium text-muted-foreground">{label}</div>
      <div className="mt-1.5 font-mono text-2xl font-semibold tabular-nums tracking-tight">
        {value}
      </div>
      {hint && <div className="mt-0.5 text-[11.5px] text-muted-foreground">{hint}</div>}
    </div>
  );
}

/** Card header used across the console's panels: title + optional subtitle + slot. */
export function PanelHeader({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-3 border-b border-border/60 px-4 py-3">
      <div className="min-w-0 flex-1">
        <h3 className="text-[13.5px] font-semibold">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

/** One row in a "needs attention" / list card. */
export function QueueRow({
  onClick,
  children,
}: {
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <div
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 border-b border-border/60 px-4 py-3 last:border-b-0",
        onClick && "cursor-pointer hover:bg-muted/40",
      )}
    >
      {children}
    </div>
  );
}
