// src/components/oneclick/payments/paymentsData.ts
// Port of claude_design payments/payments-data.jsx — Royalty Payments seed model.
// Amounts are kept in each party's contract currency; converted at display/payout time.

export type CurrencyCode = "USD" | "GBP" | "EUR" | "CAD" | "NGN" | "KES";

export interface Currency {
  code: CurrencyCode; symbol: string; name: string; flag: string; toUSD: number;
}
export interface Period {
  id: string; label: string; short: string; start: string; end: string; closed: boolean;
}
export interface Project { id: string; name: string; artist: string; }
export interface Work { id: string; title: string; projectId: string; }
export type IncomeType = "master" | "publishing";
export type MethodStatus = "connected" | "invited" | "none";
export interface PayoutMethod { platform: string; handle: string; status: MethodStatus; }
export interface Party {
  id: string; name: string; role: string; color: string; currency: CurrencyCode;
  method: PayoutMethod; location: string; taxId: string; joined: string;
}
export interface Accrual {
  partyId: string; period: string; workId: string; type: IncomeType;
  amount: number; src: "oneclick" | "manual";
}
export interface PayoutRun {
  id: string; label: string; payCurrency: CurrencyCode; status: "paid" | "scheduled";
  createdAt: string; paidAt: string | null; scheduledFor?: string | null;
  periods: string[]; partyIds: string[]; note: string;
  paidParties?: string[]; paidAtBy?: Record<string, string>;
}

export const CURRENCIES: Record<CurrencyCode, Currency> = {
  USD: { code: "USD", symbol: "$",  name: "US Dollar",       flag: "🇺🇸", toUSD: 1 },
  GBP: { code: "GBP", symbol: "£",  name: "British Pound",   flag: "🇬🇧", toUSD: 1.272 },
  EUR: { code: "EUR", symbol: "€",  name: "Euro",            flag: "🇪🇺", toUSD: 1.081 },
  CAD: { code: "CAD", symbol: "C$", name: "Canadian Dollar", flag: "🇨🇦", toUSD: 0.731 },
  NGN: { code: "NGN", symbol: "₦",  name: "Nigerian Naira",  flag: "🇳🇬", toUSD: 0.000643 },
  KES: { code: "KES", symbol: "KSh",name: "Kenyan Shilling", flag: "🇰🇪", toUSD: 0.00773 },
};

export const FX_SPREAD = 0.018;
export const FX_ASOF = "2026-06-16";

export const fxMid = (from: CurrencyCode, to: CurrencyCode): number =>
  from === to ? 1 : CURRENCIES[from].toUSD / CURRENCIES[to].toUSD;
export const fxBank = (from: CurrencyCode, to: CurrencyCode): number =>
  from === to ? 1 : fxMid(from, to) * (1 - FX_SPREAD);
export const toUSD = (amount: number, cur: CurrencyCode): number => amount * CURRENCIES[cur].toUSD;
export const toBase = (amount: number, from: CurrencyCode, base: CurrencyCode = "USD"): number =>
  amount * fxMid(from, base);

export function fmtMoney(amount: number, cur: CurrencyCode, opts?: { dp?: number }): string {
  const c = CURRENCIES[cur] ?? CURRENCIES.USD;
  const dp = opts?.dp != null ? opts.dp : cur === "NGN" || cur === "KES" ? 0 : 2;
  const n = Math.abs(amount).toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
  return (amount < 0 ? "−" : "") + c.symbol + n;
}

export const PERIODS: Period[] = [
  { id: "2025Q3", label: "Q3 2025", short: "Q3 '25", start: "2025-07-01", end: "2025-09-30", closed: true },
  { id: "2025Q4", label: "Q4 2025", short: "Q4 '25", start: "2025-10-01", end: "2025-12-31", closed: true },
  { id: "2026Q1", label: "Q1 2026", short: "Q1 '26", start: "2026-01-01", end: "2026-03-31", closed: true },
  { id: "2026Q2", label: "Q2 2026", short: "Q2 '26", start: "2026-04-01", end: "2026-06-30", closed: false },
];
export const PERIOD: Record<string, Period> = Object.fromEntries(PERIODS.map((p) => [p.id, p]));

export const PROJECTS: Record<string, Project> = {
  "midnight-ep":        { id: "midnight-ep",        name: "Midnight EP",        artist: "Nova Sky" },
  "starlight-sessions": { id: "starlight-sessions", name: "Starlight Sessions", artist: "Nova Sky" },
  "lagos-nights":       { id: "lagos-nights",       name: "Lagos Nights",       artist: "Kojo Mensah" },
  "london-sessions":    { id: "london-sessions",    name: "London Sessions",    artist: "Kojo Mensah" },
  "afterglow":          { id: "afterglow",          name: "Afterglow",          artist: "Lumen" },
  "rooted":             { id: "rooted",             name: "Rooted (LP)",        artist: "Amara Diallo" },
};
export const WORK: Record<string, Work> = {
  "neon-tide":    { id: "neon-tide",    title: "Neon Tide",    projectId: "midnight-ep" },
  "glass-hours":  { id: "glass-hours",  title: "Glass Hours",  projectId: "midnight-ep" },
  "cold-light":   { id: "cold-light",   title: "Cold Light",   projectId: "starlight-sessions" },
  "lagos-nights": { id: "lagos-nights", title: "Lagos Nights", projectId: "lagos-nights" },
  "harmattan":    { id: "harmattan",    title: "Harmattan",    projectId: "lagos-nights" },
  "thames":       { id: "thames",       title: "Thames",       projectId: "london-sessions" },
  "afterglow":    { id: "afterglow",    title: "Afterglow",    projectId: "afterglow" },
  "paper-walls":  { id: "paper-walls",  title: "Paper Walls",  projectId: "afterglow" },
  "rooted":       { id: "rooted",       title: "Rooted",       projectId: "rooted" },
};

export const PARTIES: Record<string, Party> = {
  marcus:  { id: "marcus",  name: "Marcus Adebayo", role: "Producer",         color: "#1f8a5b", currency: "USD", method: { platform: "PayPal", handle: "marcus@adebayobeats.com", status: "connected" }, location: "Lagos, NG", taxId: "W-8BEN on file", joined: "2025-05-30" },
  velvet:  { id: "velvet",  name: "Velvet",         role: "Featured Artist",  color: "#d24b6e", currency: "GBP", method: { platform: "PayPal", handle: "velvet@velvetmusic.co", status: "connected" }, location: "London, UK", taxId: "—", joined: "2026-01-10" },
  twobars: { id: "twobars", name: "TwoBars",        role: "Producer",         color: "#d9762b", currency: "NGN", method: { platform: "PayPal", handle: "accounts@twobars.ng", status: "connected" }, location: "Lagos, NG", taxId: "W-8BEN on file", joined: "2025-09-01" },
  amara:   { id: "amara",   name: "Amara Diallo",   role: "Songwriter",       color: "#7c5cff", currency: "EUR", method: { platform: "PayPal", handle: "amara@diallo.fr", status: "connected" }, location: "Paris, FR", taxId: "W-8BEN on file", joined: "2025-11-02" },
  lumen:   { id: "lumen",   name: "Lumen",          role: "Artist / Owner",   color: "#0ea5a0", currency: "GBP", method: { platform: "PayPal", handle: "lumen@lumensound.uk", status: "connected" }, location: "Manchester, UK", taxId: "—", joined: "2026-02-16" },
  skyline: { id: "skyline", name: "Skyline Records",role: "Label",            color: "#475569", currency: "USD", method: { platform: "PayPal", handle: "payments@skylinerecords.com", status: "connected" }, location: "New York, US", taxId: "W-9 on file", joined: "2025-04-01" },
  jduru:   { id: "jduru",   name: "Joy Duru",       role: "Mix Engineer",     color: "#c2410c", currency: "KES", method: { platform: "PayPal", handle: "joy.duru@gmail.com", status: "invited" }, location: "Nairobi, KE", taxId: "Pending", joined: "2026-03-04" },
};

// One row per (party, period, work, income-type). Amount in party's contract currency.
export const ACCRUALS: Accrual[] = [
  { partyId: "marcus", period: "2025Q3", workId: "cold-light",  type: "master", amount: 312, src: "oneclick" },
  { partyId: "marcus", period: "2025Q4", workId: "neon-tide",   type: "master", amount: 486, src: "oneclick" },
  { partyId: "marcus", period: "2025Q4", workId: "glass-hours", type: "master", amount: 140, src: "oneclick" },
  { partyId: "marcus", period: "2026Q1", workId: "neon-tide",   type: "master", amount: 642, src: "oneclick" },
  { partyId: "marcus", period: "2026Q1", workId: "cold-light",  type: "master", amount: 188, src: "manual" },
  { partyId: "marcus", period: "2026Q2", workId: "neon-tide",   type: "master", amount: 503, src: "oneclick" },
  { partyId: "velvet", period: "2025Q4", workId: "neon-tide",   type: "master", amount: 210, src: "oneclick" },
  { partyId: "velvet", period: "2026Q1", workId: "neon-tide",   type: "master", amount: 338, src: "oneclick" },
  { partyId: "velvet", period: "2026Q2", workId: "neon-tide",   type: "master", amount: 176, src: "oneclick" },
  { partyId: "twobars", period: "2025Q3", workId: "lagos-nights", type: "master", amount: 318000, src: "oneclick" },
  { partyId: "twobars", period: "2025Q4", workId: "lagos-nights", type: "master", amount: 472000, src: "oneclick" },
  { partyId: "twobars", period: "2025Q4", workId: "harmattan",    type: "master", amount: 196000, src: "oneclick" },
  { partyId: "twobars", period: "2026Q1", workId: "lagos-nights", type: "master", amount: 528000, src: "oneclick" },
  { partyId: "twobars", period: "2026Q1", workId: "thames",       type: "master", amount: 84000,  src: "manual" },
  { partyId: "twobars", period: "2026Q2", workId: "lagos-nights", type: "master", amount: 301000, src: "oneclick" },
  { partyId: "amara", period: "2025Q3", workId: "rooted",    type: "publishing", amount: 144, src: "oneclick" },
  { partyId: "amara", period: "2025Q4", workId: "harmattan", type: "publishing", amount: 238, src: "oneclick" },
  { partyId: "amara", period: "2025Q4", workId: "rooted",    type: "publishing", amount: 162, src: "oneclick" },
  { partyId: "amara", period: "2026Q1", workId: "harmattan", type: "publishing", amount: 296, src: "oneclick" },
  { partyId: "amara", period: "2026Q2", workId: "rooted",    type: "publishing", amount: 158, src: "oneclick" },
  { partyId: "lumen", period: "2026Q1", workId: "afterglow",   type: "master",     amount: 624, src: "oneclick" },
  { partyId: "lumen", period: "2026Q1", workId: "afterglow",   type: "publishing", amount: 280, src: "oneclick" },
  { partyId: "lumen", period: "2026Q2", workId: "afterglow",   type: "master",     amount: 452, src: "oneclick" },
  { partyId: "lumen", period: "2026Q2", workId: "paper-walls", type: "master",     amount: 118, src: "oneclick" },
  { partyId: "skyline", period: "2025Q3", workId: "cold-light", type: "master", amount: 880,  src: "oneclick" },
  { partyId: "skyline", period: "2025Q4", workId: "neon-tide",  type: "master", amount: 1120, src: "oneclick" },
  { partyId: "skyline", period: "2026Q1", workId: "neon-tide",  type: "master", amount: 1340, src: "oneclick" },
  { partyId: "skyline", period: "2026Q2", workId: "neon-tide",  type: "master", amount: 712,  src: "oneclick" },
  { partyId: "jduru", period: "2026Q1", workId: "rooted",    type: "master", amount: 38500, src: "manual" },
  { partyId: "jduru", period: "2026Q2", workId: "harmattan", type: "master", amount: 21200, src: "manual" },
];

export const SEED_PAYOUT_RUNS: PayoutRun[] = [
  { id: "run-2025q3", label: "Q3 2025 payout", payCurrency: "USD", status: "paid",
    createdAt: "2025-10-06", paidAt: "2025-10-08", periods: ["2025Q3"],
    partyIds: ["marcus", "twobars", "amara", "skyline"], note: "First quarterly run." },
  { id: "run-2025q4", label: "Q4 2025 payout", payCurrency: "USD", status: "paid",
    createdAt: "2026-01-09", paidAt: "2026-01-12", periods: ["2025Q4"],
    partyIds: ["marcus", "velvet", "twobars", "amara", "skyline"], note: "" },
  { id: "run-2026q1", label: "Q1 2026 payout", payCurrency: "USD", status: "scheduled",
    createdAt: "2026-04-10", paidAt: null, scheduledFor: "2026-04-15", periods: ["2026Q1"],
    partyIds: ["marcus", "velvet", "twobars", "amara", "lumen", "skyline", "jduru"],
    note: "Awaiting approval from finance.", paidParties: [], paidAtBy: {} },
];

export const PARTY_IDS = Object.keys(PARTIES); // stable module constant — use in memo deps to satisfy exhaustive-deps
export const party = (id: string): Party => PARTIES[id];
export const cur = (id: CurrencyCode): Currency => CURRENCIES[id];

export const accrualsFor = (partyId: string, periodIds?: string[]): Accrual[] =>
  ACCRUALS.filter((a) => a.partyId === partyId && (!periodIds || periodIds.includes(a.period)));

export const earned = (partyId: string, periodIds?: string[]): number =>
  accrualsFor(partyId, periodIds).reduce((s, a) => s + a.amount, 0);

export const runPartyPaid = (run: PayoutRun, partyId: string): boolean =>
  run.status === "paid" ? true : (run.paidParties ?? []).includes(partyId);

export function runEffectiveStatus(run: PayoutRun): "paid" | "partial" | "scheduled" {
  if (run.status === "paid") return "paid";
  const elig = run.partyIds.filter((id) => earned(id, run.periods) > 0);
  const done = elig.filter((id) => runPartyPaid(run, id)).length;
  if (done === 0) return "scheduled";
  if (done >= elig.length) return "paid";
  return "partial";
}
export const isPeriodPaid = (runs: PayoutRun[], partyId: string, periodId: string): boolean =>
  runs.some((r) => r.partyIds.includes(partyId) && r.periods.includes(periodId) && runPartyPaid(r, partyId));
export const isPeriodScheduled = (runs: PayoutRun[], partyId: string, periodId: string): boolean =>
  runs.some((r) => r.partyIds.includes(partyId) && r.periods.includes(periodId) && !runPartyPaid(r, partyId));

export const paid = (runs: PayoutRun[], partyId: string): number =>
  ACCRUALS.filter((a) => a.partyId === partyId && isPeriodPaid(runs, partyId, a.period)).reduce((s, a) => s + a.amount, 0);
export const scheduled = (runs: PayoutRun[], partyId: string): number =>
  ACCRUALS.filter((a) => a.partyId === partyId && !isPeriodPaid(runs, partyId, a.period) && isPeriodScheduled(runs, partyId, a.period)).reduce((s, a) => s + a.amount, 0);
export const outstanding = (runs: PayoutRun[], partyId: string): number => earned(partyId) - paid(runs, partyId);
export const unscheduledOutstanding = (runs: PayoutRun[], partyId: string): number =>
  ACCRUALS.filter((a) => a.partyId === partyId && !isPeriodPaid(runs, partyId, a.period) && !isPeriodScheduled(runs, partyId, a.period)).reduce((s, a) => s + a.amount, 0);

export interface ProjectRollup {
  projectId: string; amount: number;
  works: Record<string, { amount: number; rows: { period: string; amount: number; type: IncomeType; src: string }[] }>;
}
export function partyByProject(partyId: string): ProjectRollup[] {
  const map: Record<string, ProjectRollup> = {};
  accrualsFor(partyId).forEach((a) => {
    const pid = WORK[a.workId].projectId;
    if (!map[pid]) map[pid] = { projectId: pid, amount: 0, works: {} };
    map[pid].amount += a.amount;
    if (!map[pid].works[a.workId]) map[pid].works[a.workId] = { amount: 0, rows: [] };
    map[pid].works[a.workId].amount += a.amount;
    map[pid].works[a.workId].rows.push({ period: a.period, amount: a.amount, type: a.type, src: a.src });
  });
  Object.values(map).forEach((proj) => Object.values(proj.works).forEach((w) =>
    w.rows.sort((x, y) => (x.period < y.period ? -1 : 1))));
  return Object.values(map).sort((x, y) => y.amount - x.amount);
}

export const SHARE_BY_ROLE: Record<string, number> = {
  "Producer|master": 0.25, "Featured Artist|master": 0.15, "Artist / Owner|master": 0.5,
  "Artist / Owner|publishing": 0.5, "Songwriter|publishing": 0.2, "Label|master": 0.45, "Mix Engineer|master": 0.05,
};
export const accrualShare = (partyId: string, type: IncomeType): number =>
  SHARE_BY_ROLE[PARTIES[partyId].role + "|" + type] ?? 0.2;
export const ROYALTY_TYPE_LABEL: Record<IncomeType, string> = { master: "Streaming", publishing: "Publishing" };

export interface BreakdownRow {
  partyId: string; role: string; currency: CurrencyCode; workId: string; song: string; projectId: string;
  period: string; type: IncomeType; royaltyType: string; share: number; revenue: number; amount: number; payAmount: number; src: string;
}
export function payoutBreakdown(run: PayoutRun) {
  const rows: BreakdownRow[] = [];
  run.partyIds.forEach((pid) => {
    const p = PARTIES[pid];
    const rate = fxBank(p.currency, run.payCurrency);
    accrualsFor(pid, run.periods).forEach((a) => {
      const share = accrualShare(pid, a.type);
      const revenue = a.amount / share;
      rows.push({ partyId: pid, role: p.role, currency: p.currency, workId: a.workId, song: WORK[a.workId].title,
        projectId: WORK[a.workId].projectId, period: a.period, type: a.type, royaltyType: ROYALTY_TYPE_LABEL[a.type],
        share, revenue, amount: a.amount, payAmount: a.amount * rate, src: a.src });
    });
  });
  rows.sort((x, y) => y.payAmount - x.payAmount);
  const byPartyMap: Record<string, number> = {};
  rows.forEach((r) => { byPartyMap[r.partyId] = (byPartyMap[r.partyId] ?? 0) + r.payAmount; });
  const byParty = Object.entries(byPartyMap).map(([pid, amount]) => ({ partyId: pid, amount })).sort((a, b) => b.amount - a.amount);
  const total = byParty.reduce((s, x) => s + x.amount, 0);
  const songs = new Set(rows.map((r) => r.workId)).size;
  return { rows, byParty, songs, total };
}

export const payoutDate = (run: PayoutRun): string => run.paidAt || run.scheduledFor || run.createdAt;
export const payoutLabel = (run: PayoutRun): string =>
  new Date(payoutDate(run) + "T00:00:00").toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" });
export function periodSpanLabel(ids: string[]): string {
  if (!ids || !ids.length) return "current balances";
  const sorted = [...ids].sort();
  const a = PERIOD[sorted[0]], b = PERIOD[sorted[sorted.length - 1]];
  if (!a || !b) return "";
  const s = new Date(a.start + "T00:00:00"), e = new Date(b.end + "T00:00:00");
  const mo = (d: Date) => d.toLocaleDateString("en-US", { month: "short" });
  return s.getFullYear() === e.getFullYear()
    ? `${mo(s)}–${mo(e)} ${e.getFullYear()}`
    : `${mo(s)} ${s.getFullYear()}–${mo(e)} ${e.getFullYear()}`;
}
