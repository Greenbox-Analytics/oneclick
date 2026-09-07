// src/lib/orgUsage.ts
// Pure helpers behind the Teams Usage card: folding per-action spend into
// tools, bucketing the per-day series, building the table rows. No React.
import type {
  ActionSpend,
  FolderUsageRow,
  OrgSeatUsage,
  PartnerKeyUsageRow,
  SeriesDay,
  UsageRange,
} from "@/hooks/useOrgs";
import type { PartnerKey } from "@/hooks/usePartnerKeys";
import { fmtDate } from "./utils";
import { TOOLS, toolOf, type ToolId } from "./usageTools";

export const USAGE_RANGES: { id: UsageRange; label: string; title: string }[] = [
  { id: "7d", label: "7D", title: "Last 7 days" },
  { id: "14d", label: "14D", title: "Last 14 days" },
  { id: "mtd", label: "MTD", title: "This billing period" },
  { id: "1y", label: "1Y", title: "Last 365 days" },
  { id: "all", label: "All", title: "All time" },
];

export type ToolTotals = Record<ToolId, number>;
export const emptyTotals = (): ToolTotals => ({ oneclick: 0, registry: 0, splitsheet: 0, zoe: 0 });

/** Credits per tool; actions no tool owns are dropped. */
export function foldTools(actions: ActionSpend[]): ToolTotals {
  const t = emptyTotals();
  for (const a of actions) {
    const id = toolOf(a.action);
    if (id) t[id] += a.credits;
  }
  return t;
}

export const sumTools = (t: ToolTotals): number => TOOLS.reduce((n, tool) => n + t[tool.id], 0);

export function totals(series: SeriesDay[]): { credits: number; runs: number } {
  let credits = 0;
  let runs = 0;
  for (const d of series ?? []) for (const a of d.actions ?? []) { credits += a.credits; runs += a.runs; }
  return { credits, runs };
}

export function byToolTotals(series: SeriesDay[]): ToolTotals {
  const t = emptyTotals();
  for (const d of series ?? []) {
    const f = foldTools(d.actions ?? []);
    for (const tool of TOOLS) t[tool.id] += f[tool.id];
  }
  return t;
}

export function topTool(t: ToolTotals): { id: ToolId; label: string; share: number } | null {
  const total = sumTools(t);
  if (!total) return null;
  const best = TOOLS.reduce((a, b) => (t[b.id] > t[a.id] ? b : a));
  return { id: best.id, label: best.label, share: t[best.id] / total };
}

/** Percent change vs the previous window; null when there is nothing to compare. */
export function delta(current: number, previous: number | null | undefined): number | null {
  if (previous == null) return null;
  if (previous === 0) return current === 0 ? 0 : null;
  return Math.round(((current - previous) / previous) * 100) || 0;
}

// ---- bucketing ------------------------------------------------------------------

const utc = (day: string) => new Date(`${day}T00:00:00Z`);
const iso = (d: Date) => d.toISOString().slice(0, 10);

/** The Monday of the ISO week a UTC day falls in. */
export function weekStart(day: string): string {
  const d = utc(day);
  d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7));
  return iso(d);
}

export const monthStart = (day: string): string => `${day.slice(0, 7)}-01`;

export const bucketOf = (day: string, range: UsageRange): string =>
  range === "1y" ? weekStart(day) : range === "all" ? monthStart(day) : day;

export function bucketLabel(bucket: string, range: UsageRange): string {
  const d = utc(bucket);
  if (range === "all") return d.toLocaleDateString("en-US", { month: "short", year: "numeric", timeZone: "UTC" });
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
}

export interface Bucket {
  bucket: string;
  label: string;
  tools: ToolTotals;
  credits: number;
}

/** Days -> chart buckets: days for 7d/14d/mtd, ISO weeks for 1y, months for
 * all. Day and week ranges gap-fill from `since` to `today` so a quiet day
 * reads as zero; all-time lists only months with spend. */
export function bucketSeries(series: SeriesDay[], range: UsageRange, since: string | null, today: string = iso(new Date())): Bucket[] {
  const map = new Map<string, ToolTotals>();
  if (since && range !== "all") {
    const step = range === "1y" ? 7 : 1;
    const end = utc(today);
    for (const d = utc(bucketOf(since.slice(0, 10), range)); d <= end; d.setUTCDate(d.getUTCDate() + step)) map.set(iso(d), emptyTotals());
  }
  for (const day of series ?? []) {
    const key = bucketOf(day.day, range);
    const t = map.get(key) ?? emptyTotals();
    const f = foldTools(day.actions ?? []);
    for (const tool of TOOLS) t[tool.id] += f[tool.id];
    map.set(key, t);
  }
  return [...map.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([bucket, tools]) => ({ bucket, label: bucketLabel(bucket, range), tools, credits: sumTools(tools) }));
}

// ---- tables ---------------------------------------------------------------------

export interface MemberRow {
  seat: OrgSeatUsage;
  /** Product spend plus spend through keys this member created. */
  total: number;
  runs: number;
  tools: ToolTotals;
  /** Keys this member created — a key is the org's, attributed by creator. */
  keys: PartnerKey[];
  /** Per-day spend for the detail dialog; [] on a backend that doesn't send it. */
  series: SeriesDay[];
}

const sumRuns = (actions: ActionSpend[] | undefined): number =>
  (actions ?? []).reduce((n, a) => n + a.runs, 0);

export function memberRows(seats: OrgSeatUsage[], keys: PartnerKey[]): MemberRow[] {
  return seats
    .map((seat) => ({
      seat,
      total: seat.spentThisPeriod + (seat.apiCredits ?? 0),
      runs: sumRuns(seat.byAction),
      tools: foldTools(seat.byAction ?? []),
      keys: keys.filter((k) => k.created_by != null && k.created_by === seat.userId),
      series: seat.series ?? [],
    }))
    .sort((a, b) => b.total - a.total);
}

export interface KeyUsageRow {
  row: PartnerKeyUsageRow;
  total: number;
  runs: number;
  tools: ToolTotals;
  series: SeriesDay[];
}

/** Payload-driven: label, status and folder ride on the usage row, one per
 * LISTED key (zero-spend included). No join with the key list — a key past its
 * 30-day retention is absent here, while its spend still counts. */
export function keyUsageRows(byKey: PartnerKeyUsageRow[]): KeyUsageRow[] {
  return (byKey ?? [])
    .map((row) => ({ row, total: row.credits, runs: row.runs, tools: foldTools(row.byAction ?? []), series: row.series ?? [] }))
    .sort((a, b) => b.total - a.total || a.row.label.localeCompare(b.row.label));
}

export interface FolderRow {
  row: FolderUsageRow;
  total: number;
  runs: number;
  tools: ToolTotals;
  series: SeriesDay[];
}

export function folderUsageRows(byFolder: FolderUsageRow[]): FolderRow[] {
  return (byFolder ?? [])
    .map((row) => ({ row, total: row.credits, runs: row.runs, tools: foldTools(row.byAction ?? []), series: row.series ?? [] }))
    .sort((a, b) => b.total - a.total || a.row.name.localeCompare(b.row.name));
}

// ---- detail subjects ------------------------------------------------------------

/** What the detail dialog describes — a member, key or folder — flattened so
 * the dialog never branches on which it got. */
export interface UsageSubject {
  kind: "member" | "key" | "folder";
  id: string;
  title: string;
  subtitle: string;
  total: number;
  runs: number;
  tools: ToolTotals;
  series: SeriesDay[];
  facts: { label: string; value: string }[];
}

// Stored lower-case; the dialog shows them as words.
const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);

export function memberSubject(r: MemberRow): UsageSubject {
  return {
    kind: "member",
    id: r.seat.orgMemberId,
    title: r.seat.email ?? "Unknown",
    subtitle: `Member · ${cap(r.seat.role)}`,
    total: r.total,
    runs: r.runs,
    tools: r.tools,
    series: r.series,
    facts: [
      { label: "Role", value: cap(r.seat.role) },
      ...(r.seat.status !== "active" ? [{ label: "Status", value: cap(r.seat.status) }] : []),
      { label: "Keys created", value: r.keys.length ? String(r.keys.length) : "None" },
    ],
  };
}

export function keySubject(r: KeyUsageRow): UsageSubject {
  return {
    kind: "key",
    id: r.row.keyId,
    title: r.row.label,
    subtitle: `API key · ${r.row.folderName ?? "No folder"}`,
    total: r.total,
    runs: r.runs,
    tools: r.tools,
    series: r.series,
    facts: [
      { label: "Key", value: `${r.row.keyPrefix}…` },
      { label: "Status", value: cap(r.row.status) },
      { label: "Folder", value: r.row.folderName ?? "No folder" },
      { label: "Last used", value: r.row.lastUsedAt ? fmtDate(r.row.lastUsedAt) : "Never" },
    ],
  };
}

export function folderSubject(r: FolderRow): UsageSubject {
  return {
    kind: "folder",
    id: r.row.folderId ?? "unfiled",
    title: r.row.name,
    subtitle: "Folder",
    total: r.total,
    runs: r.runs,
    tools: r.tools,
    series: r.series,
    facts: [{ label: "Keys", value: String(r.row.keys) }],
  };
}

/** Keys nested under their folder for the profile card. One group per folder
 * row; anything whose folder isn't listed falls into a synthetic "No folder"
 * group, which sorts last. */
export function groupKeysByFolder(
  byKey: PartnerKeyUsageRow[],
  byFolder: FolderUsageRow[],
): { folder: FolderRow; keys: KeyUsageRow[] }[] {
  const keys = keyUsageRows(byKey);
  const folders = folderUsageRows(byFolder);
  const known = new Set(folders.map((f) => f.row.folderId));
  const groups = folders
    .filter((f) => f.row.folderId != null)
    .map((folder) => ({ folder, keys: keys.filter((k) => k.row.folderId === folder.row.folderId) }));

  const unfiled = keys.filter((k) => k.row.folderId == null || !known.has(k.row.folderId));
  const unfiledRow = folders.find((f) => f.row.folderId == null);
  if (unfiledRow || unfiled.length > 0) {
    // No unfiled row in the payload: subtotal it ourselves so the heading
    // reads the same as a server-built one.
    const total = unfiled.reduce((n, k) => n + k.total, 0);
    const tools = emptyTotals();
    for (const k of unfiled) for (const t of TOOLS) tools[t.id] += k.tools[t.id];
    groups.push({
      folder: unfiledRow ?? {
        row: { folderId: null, name: "No folder", keys: unfiled.length, credits: total, runs: 0, byAction: [] },
        total,
        runs: unfiled.reduce((n, k) => n + k.runs, 0),
        tools,
        series: mergeSeries(unfiled.map((k) => k.series)),
      },
      keys: unfiled,
    });
  }
  return groups;
}

/** Several per-day series summed into one. */
export function mergeSeries(all: SeriesDay[][]): SeriesDay[] {
  const days = new Map<string, Map<string, ActionSpend>>();
  for (const series of all) {
    for (const d of series ?? []) {
      const acts = days.get(d.day) ?? new Map<string, ActionSpend>();
      for (const a of d.actions ?? []) {
        const cur = acts.get(a.action) ?? { action: a.action, credits: 0, runs: 0 };
        acts.set(a.action, { action: a.action, credits: cur.credits + a.credits, runs: cur.runs + a.runs });
      }
      days.set(d.day, acts);
    }
  }
  return [...days.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([day, acts]) => ({ day, actions: [...acts.values()].sort((a, b) => b.credits - a.credits) }));
}

export function windowLabel(range: UsageRange, since: string | null, today: Date = new Date()): string {
  if (!since || range === "all") return "All time";
  const fmt = (d: Date) => d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
  return `${fmt(new Date(since))} – ${fmt(today)}`;
}
