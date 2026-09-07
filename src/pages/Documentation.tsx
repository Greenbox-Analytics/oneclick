import {
  useState, useMemo, useCallback, useEffect, useRef, useLayoutEffect,
  createContext, useContext,
} from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Music, ArrowLeft, ArrowRight, BookOpen, Calculator, Bot, FileText,
  Users, LayoutGrid, Folder, FolderOpen, Shield, Lightbulb, Rocket,
  Info, CheckCircle2, Zap, Volume2, StickyNote, Settings, Lock,
  Scale, FileCheck, UserPlus, Pencil, User, LogOut,
  AlertTriangle, Copy, Search, Plug, ThumbsUp, ThumbsDown, Wallet,
  DollarSign, Receipt, BarChart3, SplitSquareHorizontal, Coins, KeyRound, ChevronRight, List,
} from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useSmartBack } from "@/hooks/useSmartBack";
import { useToolPrices } from "@/hooks/useCreditPacks";
import { ACTION_ORDER, estimateCredits, SIZED_ACTIONS, TOOL_META, type ToolCreditPrices } from "@/lib/credits";
import type { CreditAction } from "@/hooks/useCreditUsage";
import { PartnerApiConsole, type ConsoleKind } from "@/components/docs/PartnerApiConsole";
import { MethodBadge, ResponseExample, Tag } from "@/components/docs/apiBits";
import {
  API_SAMPLES, ERROR_EVENT_RESPONSE, PARTNER_API_URL, REGISTRY_PRICE, ROYALTIES_PRICE, ROYALTIES_RESPONSE,
  SPLIT_SHEET_HEADERS, SPLIT_SHEET_PRICE, SPLIT_SHEET_SAMPLE, SPLITS_RESPONSE, ZOE_PRICE, ZOE_RESPONSE,
} from "@/components/docs/partnerApiSamples";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel,
  DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

// ---------------------------------------------------------------------------
// Section model — grouped sidebar nav (module scope: rendering-hoist-jsx)
// ---------------------------------------------------------------------------

// Flip to true to hide Metadata Registry / Work Detail docs and works references.
const HIDE_REGISTRY_AND_WORKS = false;

interface SectionMeta { id: string; label: string; icon: React.ElementType; group: string; }

// The sidebar has two folds: Platform (the product docs, grouped below) and
// API (its own reference nav, further down). The flat SECTIONS list still
// carries "api" last, so prev/next and the mobile chips reach it.
const PLATFORM_GROUPS: { group: string; ids: string[] }[] = [
  { group: "Getting started", ids: ["getting-started"] },
  { group: "Roster & projects", ids: ["artist-management", "portfolio", "project-detail", "work-detail", "rights-registry"] },
  { group: "Tools", ids: ["oneclick", "royalty-tracking", "zoe", "split-sheet"] },
  { group: "Workspace", ids: ["workspace", "integrations", "credits", "best-practices"] },
];
const NAV_GROUPS = [...PLATFORM_GROUPS, { group: "Platform", ids: ["api"] }];

const SECTION_LABELS: Record<string, { label: string; icon: React.ElementType }> = {
  "getting-started": { label: "Getting Started", icon: Rocket },
  "artist-management": { label: "Artist Management", icon: Users },
  portfolio: { label: "Portfolio", icon: Folder },
  "project-detail": { label: "Project Detail", icon: FolderOpen },
  "work-detail": { label: "Work Detail", icon: FileText },
  "rights-registry": { label: "Metadata Registry", icon: Shield },
  oneclick: { label: "OneClick", icon: Calculator },
  "royalty-tracking": { label: "Royalty Tracking", icon: Wallet },
  zoe: { label: "Zoe AI", icon: Bot },
  "split-sheet": { label: "Split Sheet", icon: Scale },
  workspace: { label: "Workspace", icon: LayoutGrid },
  integrations: { label: "Integrations", icon: Plug },
  api: { label: "API", icon: KeyRound },
  credits: { label: "Credits & Pricing", icon: Coins },
  "best-practices": { label: "Best Practices", icon: Lightbulb },
};

const HIDDEN_SECTION_IDS = new Set<string>(HIDE_REGISTRY_AND_WORKS ? ["rights-registry", "work-detail"] : []);

// Flat, ordered list — drives prev/next + sidebar order (js-index-maps)
const SECTIONS: SectionMeta[] = NAV_GROUPS.flatMap((g) =>
  g.ids
    .filter((id) => !HIDDEN_SECTION_IDS.has(id))
    .map((id) => ({ id, label: SECTION_LABELS[id].label, icon: SECTION_LABELS[id].icon, group: g.group }))
);

const SECTION_INDEX = new Map(SECTIONS.map((s, i) => [s.id, i]));

const SECTION_DESCRIPTIONS: Record<string, string> = {
  "getting-started": "Go from an empty workspace to your first royalty breakdown in a few minutes.",
  portfolio: "Browse your projects as a card grid grouped by year and artist.",
  "project-detail": HIDE_REGISTRY_AND_WORKS
    ? "The central hub for a project — files, audio, members, notes, and settings."
    : "The central hub for a project — works, files, audio, members, notes, and settings.",
  "work-detail": "Manage a single work — ownership splits, collaborators, licensing, agreements, and industry codes.",
  "rights-registry": "Track ownership, manage collaborator invitations, and confirm rights across all your works.",
  oneclick: "Calculate royalty splits and payments from your contracts in one click using AI.",
  "royalty-tracking": "Turn every OneClick run into an ongoing record of who's owed what — per collaborator, per period, per project.",
  zoe: "Your AI contract assistant. Ask questions and get answers grounded in your documents.",
  "split-sheet": "Create split sheet agreements and generate clean PDF or Word documents ready for signing.",
  "artist-management": "Manage your roster with profiles, streaming links, and organized projects.",
  workspace: "Your project-management hub with Kanban boards, a calendar, and integrations.",
  integrations: "Connect Msanii to Google Drive — and see what's coming next.",
  api: "Run Msanii's tools from your own systems — royalty calculations, splits from contracts, split sheets and Zoe — with request and response shapes, billing, and a live console.",
  credits: "What each AI action costs, and what's free.",
  "best-practices": "Tips for getting the most out of Msanii.",
};

// Lets content-level cards switch the active section (avoids prop drilling).
const SelectSectionContext = createContext<(id: string) => void>(() => {});

// ---------------------------------------------------------------------------
// API reference — tabs, sidebar nav, console kind
// ---------------------------------------------------------------------------

type ApiTabId = "overview" | "royalties" | "registry" | "splitsheet" | "zoe" | "errors" | "billing";

const API_TABS: { id: ApiTabId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "royalties", label: "Royalty calculation" },
  { id: "registry", label: "Splits" },
  { id: "splitsheet", label: "Split sheet" },
  { id: "zoe", label: "Zoe" },
  { id: "errors", label: "Errors" },
  { id: "billing", label: "Billing & limits" },
];
const API_TAB_IDS = new Set<string>(API_TABS.map((t) => t.id));
const parseApiTab = (s: string | null): ApiTabId => (s && API_TAB_IDS.has(s) ? (s as ApiTabId) : "overview");

// Which console sits beside each tab: the billed endpoints get their own, the
// rest share the free key check (GET /zoe/v1/models).
const CONSOLE_KIND: Record<ApiTabId, ConsoleKind> = {
  overview: "check", errors: "check", billing: "check",
  royalties: "royalties", registry: "registry", splitsheet: "splitsheet", zoe: "zoe",
};

interface ApiNavItem {
  key: string; tab: ApiTabId; label: string; anchor?: string;
  method?: "GET" | "POST" | "{ }"; icon?: React.ElementType;
}
const API_NAV: { group: string; items: ApiNavItem[] }[] = [
  { group: "Start here", items: [
    { key: "overview", tab: "overview", label: "Overview", icon: Info },
    { key: "billing", tab: "billing", label: "Billing & limits", icon: Coins },
  ] },
  { group: "Endpoints", items: [
    { key: "ep-royalties", tab: "royalties", label: "/oneclick/v1/royalties", method: "POST" },
    { key: "ep-registry", tab: "registry", label: "/registry/v1/splits", method: "POST" },
    { key: "ep-splitsheet", tab: "splitsheet", label: "/splitsheet/v1/documents", method: "POST" },
    { key: "ep-zoe", tab: "zoe", label: "/zoe/v1/chat/completions", method: "POST" },
    { key: "ep-models", tab: "overview", anchor: "connect", label: "/zoe/v1/models", method: "GET" },
  ] },
  { group: "Schemas", items: [
    { key: "sch-statement", tab: "royalties", anchor: "the-statement-file", label: "statement", method: "{ }" },
    { key: "sch-terms", tab: "royalties", anchor: "contract-terms", label: "contract_terms", method: "{ }" },
    { key: "sch-expenses", tab: "royalties", anchor: "expenses", label: "expenses", method: "{ }" },
    { key: "sch-payment", tab: "royalties", anchor: "result-event", label: "payment", method: "{ }" },
    { key: "sch-splits", tab: "registry", anchor: "result-event", label: "splits", method: "{ }" },
    { key: "sch-contributor", tab: "splitsheet", anchor: "contributors", label: "contributor", method: "{ }" },
  ] },
  { group: "Errors", items: [
    { key: "err-http", tab: "errors", anchor: "http-status-codes", label: "HTTP status codes", icon: AlertTriangle },
    { key: "err-stream", tab: "errors", anchor: "stream-error-codes", label: "Stream error codes", icon: List },
  ] },
];
// The nav row that lights up when a tab is reached without clicking a row.
const API_NAV_DEFAULT_KEY: Record<ApiTabId, string> = {
  overview: "overview", billing: "billing", errors: "err-http",
  royalties: "ep-royalties", registry: "ep-registry", splitsheet: "ep-splitsheet", zoe: "ep-zoe",
};

interface ApiTabState { tab: ApiTabId; select: (tab: ApiTabId, anchor?: string) => void; }
const ApiTabContext = createContext<ApiTabState>({ tab: "overview", select: () => {} });

// xl breakpoint (1280px) — where the console lives in the right rail rather
// than under the article. Read synchronously so the console doesn't jump on
// first paint; jsdom has no matchMedia, so tests render it inline.
const WIDE_QUERY = "(min-width: 1280px)";
function useWide() {
  const [wide, setWide] = useState(() => typeof window !== "undefined" && !!window.matchMedia?.(WIDE_QUERY).matches);
  useEffect(() => {
    const mql = window.matchMedia?.(WIDE_QUERY);
    if (!mql) return;
    const onChange = () => setWide(mql.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);
  return wide;
}

// ---------------------------------------------------------------------------
// Sidebar folds — Platform (product docs) and API (the reference nav)
// ---------------------------------------------------------------------------

const navRowCls = (on: boolean) =>
  `flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-left text-[13.5px] transition-colors ${on ? "bg-primary/10 font-semibold text-primary" : "font-medium text-muted-foreground hover:bg-muted/60 hover:text-foreground"}`;

function NavGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-3.5 last:mb-1">
      <div className="mb-1.5 ml-2.5 text-[10.5px] font-bold uppercase tracking-[0.1em] text-muted-foreground">{label}</div>
      <div className="grid gap-px">{children}</div>
    </div>
  );
}

function Fold({ label, icon: Icon, badge, open, onToggle, children }: {
  label: string; icon: React.ElementType; badge?: string; open: boolean; onToggle: () => void; children: React.ReactNode;
}) {
  return (
    <div className="mb-1">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="mb-0.5 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-sm font-bold tracking-tight text-foreground transition-colors hover:bg-muted/60"
      >
        <ChevronRight className={`h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform ${open ? "rotate-90" : ""}`} />
        <Icon className="h-[15px] w-[15px] shrink-0 opacity-80" />
        {label}
        {badge && (
          <>
            {" "}
            <span className="ml-auto font-mono text-[10.5px] font-semibold text-muted-foreground">{badge}</span>
          </>
        )}
      </button>
      {open && <div className="mb-3.5 ml-[13px] border-l border-border pl-2.5">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Heading id helpers — power the "On this page" rail + scroll-spy
// ---------------------------------------------------------------------------

function childrenToText(node: React.ReactNode): string {
  if (node == null || node === false || node === true) return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(childrenToText).join("");
  if (typeof node === "object" && "props" in (node as { props?: unknown })) {
    return childrenToText((node as { props?: { children?: React.ReactNode } }).props?.children);
  }
  return "";
}

const slugify = (s: string) =>
  s.toLowerCase().trim().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");

// ---------------------------------------------------------------------------
// Prose + layout primitives — ported from the docs design, on theme tokens
// ---------------------------------------------------------------------------

function SectionHeading({ children }: { children: React.ReactNode }) {
  const id = slugify(childrenToText(children));
  return (
    <h2
      id={id || undefined}
      data-doc-heading={id || undefined}
      data-doc-level="2"
      className="scroll-mt-28 mb-4 flex items-center gap-2 text-[19px] font-bold tracking-tight text-foreground"
    >
      {children}
    </h2>
  );
}

function Step({ num, title, children, isLast = false }: {
  num: number; title: React.ReactNode; children: React.ReactNode; isLast?: boolean;
}) {
  return (
    <div className="relative pl-12 pb-7 last:pb-0">
      {!isLast && <span className="absolute left-[15px] top-9 bottom-0 w-px bg-border" />}
      <span className="absolute left-0 top-0 grid h-8 w-8 place-items-center rounded-full bg-primary text-[13px] font-bold text-primary-foreground">
        {num}
      </span>
      <div className="pt-1">
        <h4 className="mb-1 text-[15px] font-semibold text-foreground">{title}</h4>
        <div className="text-sm leading-relaxed text-muted-foreground">{children}</div>
      </div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description }: {
  icon: React.ElementType; title: string; description: string; color?: string;
}) {
  return (
    <div className="flex gap-3.5 rounded-xl border border-border bg-card p-4 transition-colors hover:bg-muted/40">
      <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-primary/10 text-primary">
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

const CALLOUT_STYLES = {
  info: { bar: "border-l-blue-500", bg: "bg-blue-500/5", chip: "bg-blue-500/15 text-blue-500", label: "text-blue-500", icon: Info, fallback: "Note" },
  tip: { bar: "border-l-primary", bg: "bg-primary/5", chip: "bg-primary/15 text-primary", label: "text-primary", icon: Lightbulb, fallback: "Tip" },
  important: { bar: "border-l-amber-500", bg: "bg-amber-500/5", chip: "bg-amber-500/15 text-amber-500", label: "text-amber-500", icon: AlertTriangle, fallback: "Heads up" },
} as const;

function Callout({ type = "info", title, anchor, children }: {
  type?: "info" | "tip" | "important"; title?: string; anchor?: string; children: React.ReactNode;
}) {
  const s = CALLOUT_STYLES[type];
  const IconEl = s.icon;
  return (
    <div className={`my-5 flex gap-3.5 rounded-r-xl border-l-[3px] ${s.bar} ${s.bg} py-4 pl-4 pr-5`}>
      <div className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md ${s.chip}`}>
        <IconEl className="h-3.5 w-3.5" />
      </div>
      <div className="min-w-0">
        <div id={anchor} data-doc-heading={anchor} data-doc-level={anchor ? "2" : undefined} className={`mb-1 scroll-mt-28 text-[11px] font-bold uppercase tracking-wider ${s.label}`}>{title || s.fallback}</div>
        <div className="text-sm leading-relaxed text-foreground">{children}</div>
      </div>
    </div>
  );
}

function PropTable({ rows, headers = ["Item", "Status", "Description"] }: {
  rows: [string, string, string][]; headers?: [string, string, string];
}) {
  return (
    <div className="my-5 overflow-hidden rounded-xl border border-border">
      <table className="w-full border-collapse text-[13.5px]">
        <thead>
          <tr className="bg-muted/50 text-left">
            {headers.map((h) => (
              <th key={h} className="border-b border-border px-3.5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-border/60 last:border-0">
              <td className="whitespace-nowrap px-3.5 py-2.5 font-semibold text-foreground">{r[0]}</td>
              <td className="whitespace-nowrap px-3.5 py-2.5 font-mono text-[12px] text-muted-foreground">{r[1]}</td>
              <td className="px-3.5 py-2.5 leading-snug text-muted-foreground">{r[2]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CodeBlock({ label, children }: { label?: string; children: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(children).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    }).catch(() => {});
  };
  return (
    <div className="my-5 overflow-hidden rounded-xl border border-border bg-muted/40">
      <div className="flex items-center gap-2.5 border-b border-border px-3.5 py-2.5">
        <span className="flex gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-[#ff5f57]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#febc2e]" />
          <span className="h-2.5 w-2.5 rounded-full bg-[#28c840]" />
        </span>
        {label && <span className="font-mono text-[11px] text-muted-foreground">{label}</span>}
        <button
          onClick={copy}
          className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
        >
          {copied ? <CheckCircle2 className="h-3 w-3 text-primary" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto px-4 py-3.5 font-mono text-[12.5px] leading-relaxed text-foreground/90">{children}</pre>
    </div>
  );
}

interface QuickCard { title: string; desc: string; icon: React.ElementType; target: string; }
function QuickCards({ cards }: { cards: QuickCard[] }) {
  const select = useContext(SelectSectionContext);
  return (
    <div className="grid gap-3.5 sm:grid-cols-3">
      {cards.map((c) => {
        const Icon = c.icon;
        return (
          <button
            key={c.target}
            onClick={() => select(c.target)}
            className="group rounded-xl border border-border bg-card p-4 text-left transition-all hover:-translate-y-0.5 hover:border-primary/40"
          >
            <div className="mb-3.5 grid h-9 w-9 place-items-center rounded-lg bg-primary/10 text-primary">
              <Icon className="h-[18px] w-[18px]" />
            </div>
            <div className="flex items-center justify-between text-[15px] font-semibold text-foreground">
              {c.title}
              <ArrowRight className="h-4 w-4 text-muted-foreground transition-all group-hover:translate-x-0.5 group-hover:text-primary" />
            </div>
            <div className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{c.desc}</div>
          </button>
        );
      })}
    </div>
  );
}

const TOOLKIT = [
  { target: "rights-registry", name: "Metadata Registry", icon: Shield, desc: "Track ownership and confirm rights across every work." },
  { target: "oneclick", name: "OneClick", icon: Calculator, desc: "Cross-reference contracts against a statement to calculate who's owed what." },
  { target: "zoe", name: "Zoe", icon: Bot, desc: "Ask plain-language questions of any contract and get cited answers." },
  { target: "split-sheet", name: "Split Sheet", icon: Scale, desc: "Generate balanced publishing & master splits as a PDF or DOCX." },
  { target: "workspace", name: "Workspace", icon: LayoutGrid, desc: "Boards and a calendar wired into Drive." },
];

function ToolGrid() {
  const select = useContext(SelectSectionContext);
  const tools = TOOLKIT.filter((t) => !HIDDEN_SECTION_IDS.has(t.target));
  return (
    <div className="grid gap-3.5 sm:grid-cols-2">
      {tools.map((t) => {
        const Icon = t.icon;
        return (
          <button
            key={t.target}
            onClick={() => select(t.target)}
            className="group flex gap-3.5 rounded-xl border border-border bg-card p-4 text-left transition-all hover:-translate-y-0.5 hover:border-primary/40"
          >
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-lg border border-border bg-muted/40 text-primary">
              <Icon className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="text-[15px] font-semibold text-foreground">{t.name}</div>
              <div className="mt-0.5 text-[13px] leading-relaxed text-muted-foreground">{t.desc}</div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

function RolePills() {
  return (
    <div className="flex flex-wrap gap-2">
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-purple-500/10 text-purple-600 dark:text-purple-400 border border-purple-500/20">Owner</span>
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">Admin</span>
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">Editor</span>
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">Viewer</span>
    </div>
  );
}

function TabChips() {
  const tabs = [
    ...(HIDE_REGISTRY_AND_WORKS ? [] : [{ icon: Music, label: "Works" }]),
    { icon: FileText, label: "Files" },
    { icon: Volume2, label: "Audio" }, { icon: Users, label: "Members" },
    { icon: StickyNote, label: "Notes" }, { icon: Settings, label: "Settings" },
  ];
  return (
    <div className="flex flex-wrap gap-2">
      {tabs.map((t) => (
        <Badge key={t.label} variant="outline" className="gap-1.5 px-3 py-1.5 text-xs font-medium">
          <t.icon className="w-3 h-3" /> {t.label}
        </Badge>
      ))}
    </div>
  );
}

function StatusBadges() {
  return (
    <div className="flex flex-wrap gap-2">
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-gray-500/10 text-gray-600 dark:text-gray-400 border border-gray-500/20">Draft</span>
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">Pending</span>
      <span className="px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">Registered</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Content components — each at module scope (rerender-no-inline-components)
// ---------------------------------------------------------------------------

const GettingStartedContent = () => (
  <div className="space-y-8">
    <QuickCards
      cards={[
        { title: "Add an artist", desc: "Build a profile and start a roster.", icon: Users, target: "artist-management" },
        { title: "Calculate royalties", desc: "Run OneClick on a statement.", icon: Calculator, target: "oneclick" },
        { title: "Ask Zoe", desc: "Question any contract, with citations.", icon: Bot, target: "zoe" },
      ]}
    />
    <div>
      <SectionHeading>Get set up</SectionHeading>
      <div className="space-y-0">
        <Step num={1} title="Create your account">
          Sign up with your Google account or email and password. Once signed in, you'll land on your <strong>Dashboard</strong> — your central hub for every tool and feature.
        </Step>
        <Step num={2} title="Add your first artist">
          Go to <strong>Artist Profiles</strong> and click <strong>Add Artist</strong>. Fill in their name, bio, and genres, and connect streaming profiles (Spotify, Apple Music, SoundCloud). Add social and custom links for press kits or EPKs.
        </Step>
        <Step num={3} title={HIDE_REGISTRY_AND_WORKS ? "Create a project" : "Create a project and add works"}>
          {HIDE_REGISTRY_AND_WORKS ? (
            <>Create a <strong>Project</strong> in your Portfolio and upload contracts and audio into it. Msanii stores your files securely and makes contracts searchable by Zoe.</>
          ) : (
            <>Create a <strong>Project</strong> in your Portfolio, then add <strong>Works</strong> to it. Upload contracts and audio and link them to specific works. Msanii stores your files securely and makes contracts searchable by Zoe.</>
          )}
        </Step>
        <Step num={4} title="Explore the tools" isLast>
          Head to the <strong>Tools</strong> page for OneClick (royalty calculations), Zoe (AI contract analysis), and the Split Sheet generator. Each tool reads from your uploaded data — upload once, benefit everywhere.
        </Step>
      </div>
    </div>
    <div>
      <SectionHeading>The toolkit</SectionHeading>
      <ToolGrid />
    </div>
    <Callout type="tip" title="Quick start">
      The fastest path: Add artist → Create project → Upload a contract → Ask Zoe about it. You'll be productive in under five minutes.
    </Callout>
  </div>
);

const PortfolioContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed">
        The Portfolio is your home for all projects. Projects are shown as a card grid grouped by <strong>Year</strong> then by <strong>Artist</strong>. Each artist section is collapsible for quick scanning.
      </p>
    </div>
    <div>
      <SectionHeading>Two sections</SectionHeading>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Folder} title="My Projects" description="Projects you created and own. Full control over settings, members, and content." color="blue" />
        <FeatureCard icon={Users} title="Shared with Me" description="Projects where someone added you as a member. A role badge shows your access level." color="purple" />
      </div>
    </div>
    <div>
      <SectionHeading>Project cards</SectionHeading>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <FeatureCard icon={FileText} title="Project name & artist" description="Identify the project and its primary artist at a glance." color="emerald" />
        <FeatureCard icon={FileText} title="File count" description="Contracts, split sheets, royalty statements, and other documents in the project." color="amber" />
        <FeatureCard icon={Volume2} title="Audio count" description="Number of audio files uploaded to the project." color="rose" />
        <FeatureCard icon={Users} title="Member count" description="How many collaborators have access to this project." color="teal" />
        <FeatureCard icon={Zap} title="Last updated" description="When the project was last modified." color="indigo" />
      </div>
    </div>
    <Callout type="info" title="Navigation">
      {HIDE_REGISTRY_AND_WORKS ? (
        <>Click any project card to open its <strong>Project Detail</strong> page with full access to files, audio, members, and more.</>
      ) : (
        <>Click any project card to open its <strong>Project Detail</strong> page with full access to works, files, members, and more.</>
      )}
    </Callout>
  </div>
);

const ProjectDetailContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        {HIDE_REGISTRY_AND_WORKS ? (
          <>The Project Detail page is the central hub for everything in a project. The tabs give you organized access to every aspect. Click the project title to rename it inline.</>
        ) : (
          <>The Project Detail page is the central hub for everything in a project. Six tabs give you organized access to every aspect. Click a project or work title to rename it inline.</>
        )}
      </p>
      <TabChips />
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      {!HIDE_REGISTRY_AND_WORKS && (
        <FeatureCard icon={Music} title="Works tab" description="Create and manage works (tracks/compositions). Set type (single, EP track, album track, composition, or custom), ISRC, and link audio files." color="blue" />
      )}
      <FeatureCard
        icon={FileText}
        title="Files tab"
        description={
          HIDE_REGISTRY_AND_WORKS
            ? "4 folder categories: Contracts, Split Sheets, Royalty Statements, Other. Upload files directly or import from Google Drive."
            : "4 folder categories: Contracts, Split Sheets, Royalty Statements, Other. Upload files and link them to specific works with 'Relevant works' labels."
        }
        color="emerald"
      />
      <FeatureCard
        icon={Volume2}
        title="Audio tab"
        description={
          HIDE_REGISTRY_AND_WORKS
            ? "Upload and manage audio files. Files are project-scoped."
            : "Upload and manage audio files. Link audio to works. Files are project-scoped — only this project's audio appears in work dropdowns."
        }
        color="purple"
      />
      <FeatureCard
        icon={Users}
        title="Members tab"
        description={
          HIDE_REGISTRY_AND_WORKS
            ? "Manage project-level access (Owner, Admin, Editor, Viewer). Invite new members by email."
            : "Manage project-level access (Owner, Admin, Editor, Viewer) and view work-only collaborators. Invite new members by email."
        }
        color="amber"
      />
      <FeatureCard icon={StickyNote} title="Notes tab" description="Rich text notes scoped to the project using the BlockNote editor. Great for meeting notes, strategy docs, or context." color="teal" />
      <FeatureCard icon={Settings} title="Settings tab" description="Edit project name and description, view the primary artist, leave the project (non-owners) or delete it (owner only)." color="red" />
    </div>
    <div>
      <SectionHeading>Role permissions</SectionHeading>
      <RolePills />
      <div className="mt-3 rounded-xl border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/30">
              <th className="text-left p-3 font-medium text-foreground">Capability</th>
              <th className="p-3 text-center font-medium text-purple-600 dark:text-purple-400">Owner</th>
              <th className="p-3 text-center font-medium text-blue-600 dark:text-blue-400">Admin</th>
              <th className="p-3 text-center font-medium text-amber-600 dark:text-amber-400">Editor</th>
              <th className="p-3 text-center font-medium text-emerald-600 dark:text-emerald-400">Viewer</th>
            </tr>
          </thead>
          <tbody className="text-muted-foreground">
            {(HIDE_REGISTRY_AND_WORKS
              ? [
                  ["See all files", true, true, true, true],
                  ["Upload files & audio", true, true, true, false],
                  ["Manage members", true, true, false, false],
                  ["Edit project settings", true, true, false, false],
                  ["Delete project", true, false, false, false],
                ]
              : [
                  ["See all works & files", true, true, true, true],
                  ["Create/edit works", true, true, true, false],
                  ["Upload files & audio", true, true, true, false],
                  ["Manage members", true, true, false, false],
                  ["Edit project settings", true, true, false, false],
                  ["Delete project", true, false, false, false],
                ]
            ).map(([cap, ...roles], i) => (
              <tr key={i} className="border-b border-border/50 last:border-b-0">
                <td className="p-3 text-foreground/80">{cap as string}</td>
                {(roles as boolean[]).map((has, j) => (
                  <td key={j} className="p-3 text-center">
                    {has ? <CheckCircle2 className="w-4 h-4 text-emerald-500 mx-auto" /> : <span className="text-muted-foreground/30">—</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
    <Callout type="important" title="Inline editing">
      {HIDE_REGISTRY_AND_WORKS ? (
        <>Click any project title to rename it directly. Press <strong>Enter</strong> to save, <strong>Escape</strong> to cancel.</>
      ) : (
        <>Click any project or work title to rename it directly. Press <strong>Enter</strong> to save, <strong>Escape</strong> to cancel.</>
      )}
    </Callout>
  </div>
);

const WorkDetailContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        The Work Detail page is where you manage a single work — a track, composition, or recording. Get here by clicking a work in the <strong>Project Detail → Works tab</strong> or from the <strong>Metadata Registry</strong>. Everything about the work lives on one page: identity codes, ownership splits, licensing, agreements, and collaboration status.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Shield} title="Work header" description="Title (inline-editable by the owner), status badge, work type, and industry codes (ISRC, ISWC, UPC)." color="blue" />
        <FeatureCard icon={Pencil} title="Owner actions" description="Register the work, export its metadata as a PDF, invite collaborators, edit metadata, or delete it." color="purple" />
      </div>
    </div>
    <div>
      <SectionHeading>Industry codes</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        These standard identifiers let distributors, collection societies, and platforms track and pay royalties. Add them via the <strong>Edit</strong> button on the work header.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <FeatureCard icon={Info} title="ISRC" description="International Standard Recording Code — uniquely identifies a sound recording. Each master gets its own. Format: CC-XXX-YY-NNNNN." color="blue" />
        <FeatureCard icon={Info} title="ISWC" description="International Standard Musical Work Code — identifies the underlying composition (melody + lyrics). Format: T-NNN.NNN.NNN-C." color="purple" />
        <FeatureCard icon={Info} title="UPC" description="Universal Product Code — identifies the release as a whole (album, EP, single). 12-digit barcode." color="teal" />
      </div>
    </div>
    <div>
      <SectionHeading>Ownership splits</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Ownership is tracked in two separate columns: <strong>Master</strong> (recording rights) and <strong>Publishing</strong> (songwriting/composition rights). Each stake has a percentage, role, and optional IPI number.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <FeatureCard icon={Volume2} title="Master ownership" description="Who owns the recording itself — typically the artist, label, or producer." color="blue" />
        <FeatureCard icon={StickyNote} title="Publishing ownership" description="Who owns the composition — typically the songwriter, composer, or publisher." color="purple" />
      </div>
      <Callout type="tip" title="Totals should reach 100%">
        For owners and editors, each column shows a running total and flags anything that doesn't add up to 100%. A work-only collaborator sees only their own splits, so that balance check doesn't apply to their view.
      </Callout>
    </div>
    <div>
      <SectionHeading><UserPlus className="w-4 h-4" /> Inviting collaborators</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Only the work owner can invite collaborators. Each invite captures the collaborator's identity, role, and optional ownership stakes — so they know exactly what they're accepting.
      </p>
      <div className="space-y-0">
        <Step num={1} title="Click 'Invite' on the work header">
          Opens the invite modal. Optionally select from your artist roster to prefill email and name.
        </Step>
        <Step num={2} title="Fill in details">
          Enter email, name, and role (Artist, Producer, Songwriter, Composer, Publisher, Label, or a custom "Other" role). Choose stake type: None, Master only, Publishing only, or Both — with percentages for each.
        </Step>
        <Step num={3} title="Collaborator receives an email">
          The invite includes the work title, your name, their role, and stake percentages so they can review before accepting.
        </Step>
        <Step num={4} title="Accept or decline" isLast>
          The collaborator sees the invite in their <strong>Metadata Registry</strong>. Accepting confirms their stake; declining removes them. Files linked to the work become accessible only after they accept.
        </Step>
      </div>
    </div>
    <div>
      <SectionHeading>Work statuses & registration</SectionHeading>
      <StatusBadges />
      <p className="text-sm text-muted-foreground mt-3 leading-relaxed mb-4">
        Works move through three statuses as collaborators confirm their stakes:
      </p>
      <div className="rounded-xl border border-border p-4 bg-muted/20 mb-4">
        <div className="flex items-center gap-3 text-sm">
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-500/10 text-gray-400 border border-gray-500/20">Draft</span>
          <ArrowRight className="w-3 h-3 text-muted-foreground" />
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">Pending</span>
          <ArrowRight className="w-3 h-3 text-muted-foreground" />
          <span className="px-2 py-0.5 rounded text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Registered</span>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          <strong>Draft:</strong> Still being set up — add ownership, invite collaborators.<br />
          <strong>Pending:</strong> Submitted for approval — waiting on collaborators to accept.<br />
          <strong>Registered:</strong> All collaborators accepted — the work is fully confirmed.
        </p>
      </div>
      <Callout type="important" title="What changes status?">
        Changing ownership stakes or collaborators on a registered work returns it to draft so everyone re-confirms. Metadata edits (renaming, updating ISRC) are safe and don't change the status.
      </Callout>
    </div>
    <div>
      <SectionHeading>Access & permissions</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        There are two layers of access for works. Understanding the difference is key:
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Users} title="Project members" description="Added at the project level (Owner/Admin/Editor/Viewer). They see ALL works in that project and manage files, audio, and notes." color="purple" />
        <FeatureCard icon={UserPlus} title="Work collaborators" description="Invited to a specific work via the Metadata Registry. They only see the work they were invited to — not the full project or other works." color="blue" />
      </div>
      <Callout type="info" title="When to use which">
        Use <strong>project members</strong> for your internal team (managers, assistants) who need access to everything in a project. Use <strong>work collaborators</strong> for external parties (producers, featured artists) who should only see their specific work and its ownership details.
      </Callout>
      <Callout type="tip" title="What a work collaborator sees">
        A work collaborator sees <strong>only their own royalty splits</strong> — not the full ownership breakdown — so the "should total 100%" balance check doesn't appear on their view. That check is for owners and editors who manage the complete split. The work's <strong>owner is always shown</strong> on the page (in the View-only banner and the Your Access card), including their <strong>contact email</strong>, so a collaborator knows exactly who to reach out to for edit access or a fuller view.
      </Callout>
    </div>
  </div>
);

const RightsRegistryContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        The Metadata Registry is your ownership-tracking dashboard. It shows everything you own and everything you're involved in across all projects, and it's where you add new works — the <strong>Add work</strong> button walks you through registering a track step by step.
      </p>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: "Total works", desc: "Everything you're on" },
          { label: "Released", desc: "Live tracks" },
          { label: "Unreleased", desc: "Still in the works" },
          { label: "Need attention", desc: "Splits or codes missing" },
          { label: "Shared with you", desc: "Works you're invited to" },
        ].map((card) => (
          <div key={card.label} className="rounded-xl border border-border p-3 bg-card">
            <p className="text-xs font-semibold text-primary uppercase tracking-wide">{card.label}</p>
            <p className="text-xs text-muted-foreground mt-1">{card.desc}</p>
          </div>
        ))}
      </div>
    </div>
    <div>
      <SectionHeading>Dashboard</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        The registry opens on a dashboard of every work you're involved in. Toggle between <strong>My Works</strong> and <strong>Shared with Me</strong>, use the stat cards across the top to filter the list, and click <strong>Add work</strong> to register a new track.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Folder} title="My Works / Shared with Me" description="Switch between works you own and works other people have shared with you." color="purple" />
        <FeatureCard icon={Zap} title="Stat filters" description="Total, Released, Unreleased, Need attention, and Shared with you — click a card to filter the list." color="amber" />
        <FeatureCard icon={Search} title="Search & sort" description="Find a work by title and sort by recently added, title A–Z, or release date." color="teal" />
        <FeatureCard icon={Shield} title="Status at a glance" description="Every row shows its registry status — Draft, Pending, or Registered." color="blue" />
      </div>
    </div>
    <div>
      <SectionHeading><Music className="w-4 h-4" /> Adding a work</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        The Add Work wizard registers a track in a few guided steps:
      </p>
      <div className="space-y-0">
        <Step num={1} title="Pick the artist and project">
          Choose where the work belongs. You can create a new artist or project without leaving the wizard. (This step is skipped when you start from a project page.)
        </Step>
        <Step num={2} title="Released or unreleased?">
          For released tracks we search Spotify and auto-fill the metadata — ISRC, UPC, label, release date, and credited artists — for you to review. For unreleased tracks, enter the details you know now and fill in the rest later.
        </Step>
        <Step num={3} title="Set the royalty splits">
          Either <strong>get splits from the contract</strong> — pick contracts already in the project or upload PDFs, and AI reads each party's split (several contracts can be combined; disagreements between them are flagged for you to fix) — or <strong>add them by hand</strong>.
        </Step>
        <Step num={4} title="Review and confirm" isLast>
          A final summary shows every party and split before the work and its stakes are saved.
        </Step>
      </div>
      <Callout type="tip" title="Contracts are attached automatically">
        Any contract used to read splits is added to the work's <strong>Related documents</strong>, so the paper trail stays with the work.
      </Callout>
    </div>
    <div>
      <SectionHeading>The work page</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        Click any work to open its page — everything about the work lives there:
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Scale} title="Royalty splits" description="Master and publishing splits per party, with running totals. SoundExchange royalties, when present, are shown separately — they're paid directly by SoundExchange and never counted in the master total." color="blue" />
        <FeatureCard icon={FileText} title="Related documents" description="Contracts and split sheets linked to the work — pick from the project's files or upload new ones." color="amber" />
        <FeatureCard icon={UserPlus} title="Collaborators & access" description="Invite collaborators with their stakes, derive a collaborator's split straight from the linked contracts, and control exactly what each person can see." color="purple" />
        <FeatureCard icon={FileCheck} title="Traceability & Export Metadata" description="A quick audit of what's on file — linked documents, ISRC, recorded stakes — plus a one-click Export Metadata PDF." color="emerald" />
      </div>
    </div>
    <div>
      <SectionHeading>Work statuses</SectionHeading>
      <StatusBadges />
      <p className="text-sm text-muted-foreground mt-3 leading-relaxed">
        Works move from <strong>Draft</strong> → <strong>Pending</strong> (submitted for collaborator approval) → <strong>Registered</strong> (all collaborators confirmed). Changing ownership stakes or collaborators on a registered work returns it to draft for re-confirmation; metadata edits like renaming are safe.
      </p>
    </div>
    <div>
      <SectionHeading><UserPlus className="w-4 h-4" /> Inviting collaborators</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        When inviting a collaborator, the form captures everything needed:
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <FeatureCard icon={Users} title="Email & name" description="Who you're inviting — they'll receive a rich email with all the details." color="blue" />
        <FeatureCard icon={Shield} title="Role & stakes" description="Set master %, publishing %, or both, and choose their role (producer, songwriter, etc.)." color="emerald" />
        <FeatureCard icon={FileText} title="Notes & terms" description="Add context about the arrangement. The collaborator sees this in the invite email." color="amber" />
        <FeatureCard icon={Lock} title="Access control" description="Files become accessible only after acceptance. The invite email contains all decision-making info." color="red" />
      </div>
      <Callout type="tip" title="Let the contract fill in the numbers">
        When the work has linked contracts, the invite form can <strong>derive the collaborator's split from the contracts</strong> — AI finds their share so you don't have to type it.
      </Callout>
    </div>
  </div>
);

const OneClickContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>How it works</SectionHeading>
      <div className="space-y-0">
        <Step num={1} title="Select an artist">Choose the artist whose royalties you want to calculate from your roster.</Step>
        <Step num={2} title="Upload or select contracts">Upload contract PDFs or pick from existing files. OneClick's AI reads them to extract parties, works, and royalty split percentages for each revenue type. Contracts can come from a project's files, files linked to a work, or an artist's profile documents.</Step>
        <Step num={3} title="Upload a royalty statement">Add the statement with the actual revenue figures — <strong>CSV or Excel</strong>. OneClick auto-detects the columns.</Step>
        <Step num={4} title="Calculate">OneClick applies the contract terms to the statement and produces a breakdown of what each party is owed, <strong>per song and per payee</strong>. It streams its progress as it downloads files, extracts parties, works, and splits, and runs the numbers.</Step>
        <Step num={5} title="Export & share" isLast>Download the results as <strong>CSV</strong> or <strong>Excel</strong>, view the payout distribution chart, or send the result straight to <strong>Google Drive</strong>.</Step>
      </div>
    </div>
    <Callout type="tip" title="Best results">
      Upload clear, text-based PDF contracts. Scanned images may have lower accuracy. Review the extracted splits before calculating to make sure the AI read the contract correctly.
    </Callout>
    <Callout type="info" title="Cached results">
      Each calculation is cached against the exact statement and contract set, so reopening it is instant. Press <strong>Recalculate</strong> after you change an input.
    </Callout>
    <Callout type="info" title="Data sources">
      OneClick can pull contracts from <strong>project files</strong>, <strong>work-linked files</strong>, and <strong>artist profile documents</strong> — all from the document selection step.
    </Callout>
  </div>
);

const ZoeContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>How to use Zoe</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Zoe answers two kinds of questions — general music-industry questions and questions about your own contracts. For contract questions it uses semantic search to find the relevant clauses and grounds every answer in your actual documents.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Bot} title="Ask a question" description="Type specific questions like 'What is the royalty rate for streaming?' for the best results." color="indigo" />
        <FeatureCard icon={Zap} title="Suggested prompts" description="Use the suggested prompts to jump-start common questions and clarify which document you mean." color="amber" />
        <FeatureCard icon={FileText} title="Source citations" description="Zoe points back to the source document — with page numbers where available — so you can verify against the original." color="emerald" />
        {!HIDE_REGISTRY_AND_WORKS && (
          <FeatureCard icon={Shield} title="Shared works" description="Reach contracts from works you collaborate on via the 'From Shared Works' source option." color="blue" />
        )}
      </div>
    </div>
    <div>
      <SectionHeading>What Zoe can answer</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Zoe picks the right source for each question, so you can ask about the wider music business or drill into a specific agreement in the same chat.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard
          icon={BookOpen}
          title="General industry questions"
          description="Ask how the music business works — royalty types, publishing vs. masters, common deal structures, or what a term means. Zoe draws on a built-in music-industry knowledge base, so no contract is required (e.g. 'What's a typical producer royalty?')."
          color="indigo"
        />
        <FeatureCard
          icon={FileText}
          title="Contract-specific questions"
          description="Ask about a document you've uploaded — rates, terms, expiry, exclusivity, recoupment. Zoe searches the selected contracts and answers from their actual text, with citations (e.g. 'When does this agreement expire?')."
          color="emerald"
        />
      </div>
      <Callout type="tip" title="Not sure which you're asking?">
        If a question could go either way, Zoe asks a quick follow-up — use the suggested prompts to point it at a specific contract or keep the answer general.
      </Callout>
    </div>
    <div>
      <SectionHeading>Example questions</SectionHeading>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {[
          "What are the royalty split percentages?",
          "When does this agreement expire?",
          "What rights does the label have over masters?",
          "Summarize the key terms of this publishing deal.",
          "Are there any exclusivity clauses?",
          "What is the advance recoupment structure?",
        ].map((q) => (
          <div key={q} className="p-3 rounded-lg border border-border bg-muted/20 text-sm text-muted-foreground italic transition-colors hover:bg-muted/40">
            "{q}"
          </div>
        ))}
      </div>
    </div>
    <Callout type="info" title="Chat sessions">
      Zoe keeps the context of your current conversation while you chat. Starting a <strong>New Chat</strong> clears it — Zoe doesn't keep a long-term archive of past conversations, so capture anything important before you move on.
    </Callout>
  </div>
);

const SplitSheetContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Step-by-step guide</SectionHeading>
      <div className="space-y-0">
        <Step num={1} title="Enter song details">Enter the song title, date, and any notes. This information appears at the top of the generated split sheet.</Step>
        <Step num={2} title="Define splits">Add each contributor with their role (songwriter, producer, performer, and more). Set publishing and master ownership percentages. Optionally add IPI/CAE numbers, publisher names, and label info.</Step>
        <Step num={3} title="Review & download" isLast>Each column (publishing and master) must total 100% — the export stays locked until it does. Download as a polished <strong>PDF or Word (DOCX)</strong> document with signature blocks, and optionally save it to the project.</Step>
      </div>
    </div>
    <Callout type="tip" title="Pro tip">
      Create split sheets <strong>before</strong> starting a project to avoid disputes later. Include IPI numbers when available — collecting societies need them to route royalties properly.
    </Callout>
  </div>
);

const ArtistManagementContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Artist profiles</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">Each artist profile is your private space for managing that artist's information, documents, and notes.</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Users} title="Basic info" description="Name, bio, genres, and profile image." color="blue" />
        <FeatureCard icon={Music} title="DSP links" description="Connect Spotify, Apple Music, and SoundCloud profiles." color="emerald" />
        <FeatureCard icon={Zap} title="Social & custom links" description="Instagram, TikTok, YouTube, websites, EPKs, press kits, and linktrees." color="purple" />
        <FeatureCard icon={StickyNote} title="Notes" description="Rich text notes on the profile for meeting notes, strategy docs, or context." color="amber" />
      </div>
    </div>
    <div>
      <SectionHeading>Projects & documents</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed">
        {HIDE_REGISTRY_AND_WORKS ? (
          <>Organize an artist's work into <strong>Projects</strong>. Each project stores contracts, royalty statements, and audio files. Uploaded contracts are automatically available to OneClick and Zoe — upload once, benefit everywhere.</>
        ) : (
          <>Organize an artist's work into <strong>Projects</strong>. Each project stores contracts, royalty statements, works, and audio files. Uploaded contracts are automatically available to OneClick and Zoe — upload once, benefit everywhere.</>
        )}
      </p>
    </div>
    <Callout type="important" title="Privacy">
      {HIDE_REGISTRY_AND_WORKS ? (
        <>Artist profiles are <strong>private to you</strong>. Only you can see an artist's notes and profile details.</>
      ) : (
        <>Artist profiles are <strong>private to you</strong>. Only you can see an artist's notes and profile details. Works and their linked contracts can be shared with collaborators through the Metadata Registry.</>
      )}
    </Callout>
  </div>
);

const WorkspaceContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Features</SectionHeading>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={LayoutGrid} title="Kanban boards" description="Drag-and-drop task boards with columns you define. Create tasks with titles, descriptions, priority, due dates, labels, and color coding." color="sky" />
        <FeatureCard icon={Zap} title="Calendar view" description="See tasks on a timeline (day, week, month, or year). Track deadlines, release dates, and contract expirations at a glance." color="amber" />
        <FeatureCard icon={Plug} title="Integrations" description="Connect Google Drive to move files into and out of your projects." color="blue" />
        <FeatureCard icon={Settings} title="Settings" description="Configure timezone and time format (12h/24h). Preferences apply across the Dashboard and Workspace." color="emerald" />
      </div>
    </div>
    <Callout type="info" title="Task organization">
      Create parent tasks with subtasks for hierarchical tracking. Add comments for collaboration notes, link tasks to artists, projects, and contracts, and filter boards by artist to focus on specific work.
    </Callout>
  </div>
);

const IntegrationsContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Connected services</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        Connect Msanii to the tools your team already uses from <strong>Workspace → Settings → Integrations</strong>. Connected services let you move files, tasks, and notifications between Msanii and the apps you live in.
      </p>
      <PropTable
        rows={[
          ["Google Drive", "Connected", "Import files into a project, export files back to Drive, and set up folder sync."],
        ]}
      />
    </div>
    <div>
      <SectionHeading>How to connect</SectionHeading>
      <CodeBlock label="Workspace → Settings → Integrations">{`Connect a service in two clicks:

  ✓ Google Drive          connected`}</CodeBlock>
    </div>
    <Callout type="tip" title="Spotify metadata — no setup needed">
      When you mark a work as <strong>Released</strong>, Msanii can pull its ISRC, UPC, release date, and cover art from Spotify automatically. There's nothing to connect — it's built into the Metadata Registry.
    </Callout>
    <Callout type="info" title="What's live today">
      Google Drive is connected today — import files into a project and export them back to Drive. Need to run calculations from your own systems? See the <strong>API</strong> section.
    </Callout>
  </div>
);

// Prices come from GET /billing/credit-packs (the same `credit_prices` rows the
// charge is computed from), never literals — the base rates have already moved
// once, and a hardcoded table is a promise that silently goes wrong.
/** "What will this cost me?" — the one question the tables above can't answer,
 *  because it depends on how long your contracts are and how often you run.
 *  Math lives in lib/credits.ts and is pinned to the backend's own test values. */
const CostEstimator = ({ prices }: { prices?: ToolCreditPrices | null }) => {
  const [action, setAction] = useState<CreditAction>("oneclick_run");
  const [pages, setPages] = useState(15);
  const [runs, setRuns] = useState(10);

  const PRICE_KEY: Record<CreditAction, keyof ToolCreditPrices> = {
    oneclick_run: "oneclickRun",
    registry_parse: "registryParse",
    zoe_message: "zoeMessage",
    split_sheet: "splitSheet",
  };
  const base = prices?.[PRICE_KEY[action]] ?? 0;
  const sized = SIZED_ACTIONS.includes(action);
  const each = estimateCredits(action, pages, base);
  const monthly = each * Math.max(0, runs);

  const field = "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground";

  return (
    <div className="my-5 rounded-xl border border-border bg-muted/20 p-5">
      <div className="grid gap-4 sm:grid-cols-[1.4fr_1fr] sm:items-end">
        <label className="block">
          <span className="mb-1.5 block text-[12px] font-medium text-muted-foreground">Action</span>
          <select className={field} value={action} onChange={(e) => setAction(e.target.value as CreditAction)}>
            {ACTION_ORDER.map((a) => (
              <option key={a} value={a}>{TOOL_META[a].label}</option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className="mb-1.5 block text-[12px] font-medium text-muted-foreground">Per month</span>
          <input
            type="number"
            min={0}
            className={`${field} tabular-nums`}
            value={runs}
            onChange={(e) => setRuns(Math.max(0, Number(e.target.value) || 0))}
          />
        </label>
      </div>

      {sized && (
        <label className="mt-4 block">
          <span className="mb-1.5 flex items-baseline justify-between text-[12px] font-medium text-muted-foreground">
            Pages per run
            <span className="tabular-nums text-foreground">{pages}</span>
          </span>
          <input
            type="range"
            min={1}
            max={150}
            value={pages}
            onChange={(e) => setPages(Number(e.target.value))}
            className="w-full accent-primary"
          />
        </label>
      )}

      <div className="mt-5 flex flex-wrap items-baseline gap-x-2 gap-y-1 border-t border-border/60 pt-4">
        <span className="text-[26px] font-bold tabular-nums text-foreground">{each}</span>
        <span className="text-sm text-muted-foreground">credits each</span>
        <span className="text-muted-foreground/50">·</span>
        <span className="text-[26px] font-bold tabular-nums text-foreground">{monthly.toLocaleString()}</span>
        <span className="text-sm text-muted-foreground">per month</span>
      </div>
      <p className="mt-1.5 text-[11px] text-muted-foreground">
        An estimate. Long documents vary in density, and anything you&apos;ve already run costs the base.
      </p>
    </div>
  );
};

const CreditsContent = () => {
  const { data } = useToolPrices();
  const prices = data?.prices;
  const cr = (n: number | null | undefined) => (n == null ? "—" : `${n} credits`);

  return (
    <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
      <div>
        <SectionHeading>What things cost</SectionHeading>
        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
          AI actions cost credits. Everything else — artists, projects, files, boards, invoices — is free.
        </p>
        <div className="my-5 overflow-hidden rounded-xl border border-border">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="bg-muted/50 text-left">
                {["Action", "Cost", "Includes"].map((h) => (
                  <th key={h} className="border-b border-border px-3.5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ["OneClick run", cr(prices?.oneclickRun), "One statement calculated · ~10 pages of contracts"],
                ["Contract parse", cr(prices?.registryParse), "Splits and terms read out · ~10 pages"],
                ["Zoe message", cr(prices?.zoeMessage), "One answer, plus the reading behind it"],
                ["Split sheet", cr(prices?.splitSheet), "One document (PDF and Word count separately)"],
              ].map((r) => (
                <tr key={r[0]} className="border-b border-border/60 last:border-0">
                  <td className="whitespace-nowrap px-3.5 py-2.5 font-semibold text-foreground">{r[0]}</td>
                  <td className="whitespace-nowrap px-3.5 py-2.5 font-semibold tabular-nums text-foreground">{r[1]}</td>
                  <td className="px-3.5 py-2.5 leading-snug text-muted-foreground">{r[2]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Charged per action, not per file. Three short contracts cost the same as one of the same total
          length.
        </p>
      </div>

      <div>
        <SectionHeading>Long documents</SectionHeading>
        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
          Every action includes about <strong>10 pages</strong>. After that, roughly{" "}
          <strong>2 credits per 5 extra pages</strong>. Pages are counted across the whole run.
        </p>
        <div className="my-5 overflow-hidden rounded-xl border border-border">
          <table className="w-full border-collapse text-[13.5px]">
            <thead>
              <tr className="bg-muted/50 text-left">
                {["Contract length", "OneClick run"].map((h) => (
                  <th key={h} className="border-b border-border px-3.5 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[["Up to 10 pages", "30 credits"], ["15 pages", "32 credits"], ["30 pages", "37 credits"], ["60 pages", "46 credits"], ["100 pages", "58 credits"]].map((r) => (
                <tr key={r[0]} className="border-b border-border/60 last:border-0">
                  <td className="whitespace-nowrap px-3.5 py-2.5 font-semibold text-foreground">{r[0]}</td>
                  <td className="whitespace-nowrap px-3.5 py-2.5 tabular-nums text-muted-foreground">{r[1]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Callout type="info" title="Put simply">
          A normal contract costs the listed price. A long one costs a bit more.
        </Callout>
      </div>

      <div>
        <SectionHeading>Estimate your cost</SectionHeading>
        <CostEstimator prices={prices} />
      </div>

      <div>
        <SectionHeading>Free</SectionHeading>
        <ul className="space-y-2 text-sm text-muted-foreground leading-relaxed list-disc pl-5">
          <li>Greetings and thanks to Zoe — only real questions are charged</li>
          <li>Reviewing expenses and confirming a OneClick result</li>
          <li>Reading, exporting, or downloading anything you&apos;ve already made</li>
        </ul>
      </div>

      <div>
        <SectionHeading>Your credits</SectionHeading>
        <p className="text-sm text-muted-foreground leading-relaxed mb-4">
          Monthly credits reset on your billing date and don&apos;t roll over. Credits you buy never expire
          and are spent last. Track them under <strong>Account &amp; Billing → Credits &amp; usage</strong>.
        </p>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Run out? Buy more, or turn on pay-per-use and it goes on your next invoice. On a team, you draw
          from the shared pool up to the limit your admin sets.
        </p>
      </div>
    </div>
  );
};

const BestPracticesContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <Callout type="tip" title="Organize with projects">
      Create a project for each deal, album, or major agreement. Name them descriptively (e.g. "2024 Publishing Deal — Universal" rather than "Deal 1"). Upload both contracts and royalty statements to the same project for seamless OneClick calculations.
    </Callout>
    <Callout type="important" title="Contract management">
      Upload contracts as soon as they're signed. Use text-based PDFs (not scanned images) for the best AI accuracy. After uploading, use Zoe to verify key terms were correctly extracted, and track expiration dates with Workspace boards.
    </Callout>
    {!HIDE_REGISTRY_AND_WORKS && (
      <Callout type="info" title="Rights & ownership">
        Use the Metadata Registry to track ownership of every work before distributing or licensing. When inviting collaborators, include detailed stake information so everyone has a clear record. Keep files linked to works — collaborators see linked files once they accept.
      </Callout>
    )}
    <Callout type="tip" title="Workflow recommendations">
      <strong>New artist:</strong> Create profile → Add projects → Upload contracts → Set up a Workspace board.<br />
      <strong>Royalty period:</strong> Upload statements → Run OneClick → Export Excel → Record payments.<br />
      <strong>New collaboration:</strong> Generate a split sheet → Share the PDF → Upload the signed copy to the project.
    </Callout>
  </div>
);

const RoyaltyTrackingContent = () => (
  <div className="space-y-8 divide-y divide-border/30 [&>*]:pt-6 [&>*:first-child]:pt-0">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Royalty Tracking is the second tab on the OneClick page (<strong>Tools → OneClick → Royalty Tracking</strong>). Every time you run OneClick on a royalty statement, it records what each collaborator earned — per project and per statement period — and tracks paid vs. still-owed over time. You can issue per-collaborator invoices and see exactly how each amount was derived.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <FeatureCard
          icon={Users}
          title="Parties"
          description="Every collaborator with their earned, paid, and outstanding totals plus a status badge at a glance."
        />
        <FeatureCard
          icon={Receipt}
          title="Payouts"
          description="The invoices you've created — draft or paid — with full breakdowns of how each amount was calculated."
        />
        <FeatureCard
          icon={BarChart3}
          title="Periods"
          description="A collaborator × statement-period ledger of earnings, so you can see every figure across time."
        />
      </div>
    </div>

    <div>
      <SectionHeading>Getting data in</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        No extra steps required. Run OneClick on a royalty statement from the <strong>Calculate</strong> tab and the Royalty Tracking tab populates automatically. When you upload the statement, choose its <strong>currency</strong> so all amounts are labelled correctly from the start.
      </p>
      <Callout type="tip" title="Re-running a statement">
        If you upload a revised statement, re-running OneClick refreshes the figures for that period automatically. Already-issued invoices are kept for your records.
      </Callout>
    </div>

    <div>
      <SectionHeading>Reporting currency</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        The currency selector in the top-right of the Royalty Tracking tab re-expresses all totals in the currency you pick. Supported currencies are <strong>USD, GBP, EUR, CAD, AUD, NGN,</strong> and <strong>AED</strong>. Conversions use <strong>Bank of Canada</strong> official daily rates where available, with a free mid-market fallback for the rest (rates are cached).
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard
          icon={DollarSign}
          title="Per-collaborator payout currency"
          description="Each collaborator can have their own payout currency set in their drawer — invoices are converted to that currency automatically."
        />
        <FeatureCard
          icon={Wallet}
          title="Display currency"
          description="The top-right selector changes how all totals are displayed on screen without affecting the underlying figures."
        />
      </div>
    </div>

    <div>
      <SectionHeading>Collaborator drawer</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Click any collaborator in the <strong>Parties</strong> view to open their drawer. It shows:
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        <FeatureCard icon={BarChart3} title="Balance summary" description="Current earned, paid, and outstanding totals in their payout currency." />
        <FeatureCard icon={FileText} title="Earnings breakdown" description="Earnings drilled down by project → statement → line item, so every figure is traceable." />
        <FeatureCard icon={Receipt} title="Payment history" description="A log of every payout invoice created for this collaborator." />
        <FeatureCard icon={SplitSquareHorizontal} title="Split a profile" description="If one name was matched to two different people, use Split to separate them into distinct records." />
      </div>
      <p className="text-sm text-muted-foreground leading-relaxed">
        You can also set the collaborator's <strong>payout currency</strong> directly from the drawer — it applies to all future invoices for that person.
      </p>
    </div>

    <div>
      <SectionHeading>Creating a payout (invoice)</SectionHeading>
      <div className="space-y-0">
        <Step num={1} title="Click 'New payout'">
          Opens the payout dialog. You'll see all collaborators who currently have an outstanding balance.
        </Step>
        <Step num={2} title="Select collaborators">
          Check each person you want to pay out in this batch. Their outstanding balance is shown next to their name.
        </Step>
        <Step num={3} title="Confirm">
          Each selected collaborator gets their own <strong>draft invoice</strong> with a detailed breakdown — the statement total per project at each period, the contract split that was applied, and the amount owed converted to their payout currency.
        </Step>
        <Step num={4} title="Mark paid or cancel" isLast>
          Open the invoice and click <strong>Mark as paid</strong> to record the payout, or <strong>Cancel</strong> to discard the draft and release the balance back to outstanding.
        </Step>
      </div>
      <Callout type="info" title="Payment recording">
        Marking an invoice paid records the payout inside Msanii. Sending money (e.g. via PayPal or bank transfer) happens outside the app for now — direct payment integrations are a planned addition.
      </Callout>
    </div>

    <div>
      <SectionHeading>Deleting royalty entries</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        To remove a project's royalty data, go to <strong>Project Detail → Settings → Delete royalty entries</strong>. Any invoices already issued for that project are kept for your records — only the underlying earnings data is removed.
      </p>
      <Callout type="important" title="Invoices are preserved">
        Deleting royalty entries does not delete invoices. If you need to clean up a payout record, cancel the draft invoice before deleting entries.
      </Callout>
    </div>
  </div>
);

// ---------------------------------------------------------------------------
// Map section id -> content component (module-level, stable reference)
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// API — the partner-facing reference. Mirrors docs/partner-api-reference.md;
// keep the two in step. The Try-it box posts straight to the partner host.
// ---------------------------------------------------------------------------

function SubHeading({ children }: { children: string }) {
  const id = slugify(children);
  return (
    <h3 id={id} data-doc-heading={id} data-doc-level="3" className="scroll-mt-28 mt-7 mb-2 text-[15px] font-semibold text-foreground">
      {children}
    </h3>
  );
}

const P = ({ children }: { children: React.ReactNode }) => (
  <p className="text-sm text-muted-foreground leading-relaxed mb-3">{children}</p>
);

// ---------------------------------------------------------------------------
// API reference — tabbed content. Ported from the "API docs — two directions"
// design (direction A: one docs page, a tab strip, the console in the rail).
// Mirrors docs/partner-api-reference.md — keep the two in step.
// ---------------------------------------------------------------------------

function ApiTabStrip({ tab, onSelect }: { tab: ApiTabId; onSelect: (tab: ApiTabId) => void }) {
  return (
    <div role="tablist" className="no-scrollbar mb-6 mt-5 flex gap-0.5 overflow-x-auto border-b border-border">
      {API_TABS.map((t) => {
        const on = t.id === tab;
        return (
          <button
            key={t.id}
            role="tab"
            aria-selected={on}
            onClick={() => onSelect(t.id)}
            className={`relative whitespace-nowrap rounded-t-lg px-3.5 py-2.5 text-[13.5px] font-semibold transition-colors ${on ? "text-primary after:absolute after:inset-x-2.5 after:-bottom-px after:h-0.5 after:bg-primary after:content-['']" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"}`}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

function EndpointRow({ method, path, description, tag, onClick }: {
  method: "GET" | "POST"; path: string; description: string; tag: string; onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="grid w-full grid-cols-[auto_1fr_auto] items-center gap-3.5 rounded-[10px] border border-border px-3.5 py-3 text-left transition-colors hover:border-primary/40 hover:bg-muted/50"
    >
      <MethodBadge method={method} />
      <span className="min-w-0">
        <span className="block font-mono text-[13.5px] text-foreground">{path}</span>
        <span className="block text-[13.5px] text-muted-foreground">{description}</span>
      </span>
      <Tag>{tag}</Tag>
    </button>
  );
}

function EndpointHero({ method, path, title, stats, children }: {
  method: "GET" | "POST"; path: string; title: string; stats: [string, string][]; children: React.ReactNode;
}) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(path).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    }).catch(() => {});
  };
  return (
    <div className="mb-6 rounded-[14px] border border-border bg-gradient-to-b from-muted/60 to-card px-5 pb-4 pt-4">
      <div className="mb-2 flex flex-wrap items-center gap-3">
        <MethodBadge method={method} />
        <span className="font-mono text-[16px] font-medium text-foreground">{path}</span>
        <button type="button" onClick={copy} className="ml-auto">
          <Tag>
            {copied ? <CheckCircle2 className="h-3 w-3 text-primary" /> : <Copy className="h-3 w-3" />}
            {copied ? "Copied" : "Copy path"}
          </Tag>
        </button>
      </div>
      <h3 className="mb-2 text-[23px] font-bold tracking-tight text-foreground">{title}</h3>
      <p className="max-w-[62ch] text-sm leading-relaxed text-muted-foreground">{children}</p>
      <div className="mt-3.5 flex flex-wrap gap-x-6 gap-y-3 border-t border-border pt-3.5">
        {stats.map(([k, v]) => (
          <div key={k}>
            <div className="mb-0.5 text-[10.5px] font-bold uppercase tracking-wider text-muted-foreground">{k}</div>
            <div className="font-mono text-[13px] text-foreground">{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Inputs / Outputs block of an endpoint tab. The name is a rail heading.
function Sect({ tone, name, sub, tag, children }: {
  tone: "in" | "out"; name: string; sub: string; tag: string; children: React.ReactNode;
}) {
  const id = slugify(name);
  return (
    <section className="mb-9 last:mb-0">
      <div className="mb-5 flex items-center gap-3 border-b border-border pb-3">
        <span className={`h-3.5 w-3.5 shrink-0 rounded ${tone === "in" ? "bg-primary" : "bg-[hsl(var(--pay-sched-fg))]"}`} />
        <h2 id={id} data-doc-heading={id} data-doc-level="2" className="scroll-mt-28 text-[17px] font-bold tracking-tight text-foreground">{name}</h2>
        <span className="text-[13px] text-muted-foreground">{sub}</span>
        <Tag className="ml-auto">{tag}</Tag>
      </div>
      {children}
    </section>
  );
}

const ApiOverviewPanel = () => {
  const { select } = useContext(ApiTabContext);
  return (
    <div>
      <SectionHeading>Connect</SectionHeading>
      <P>
        Every request goes to your base URL over HTTPS and carries your key as a bearer token. <code>GET /zoe/v1/models</code> is the one free route, so it doubles as the key check: it proves the base URL, the key and the network path, and never uses credits — the console on this page runs it. A missing, unknown, revoked or expired key is a <code>401</code> with code <code>invalid_key</code> on every route, as is a team whose API access has been turned off.
      </P>
      {PARTNER_API_URL ? <CodeBlock label="Base URL">{PARTNER_API_URL}</CodeBlock> : <P>Your base URL comes from your Msanii contact.</P>}
      <CodeBlock label="Header">{`Authorization: Bearer mk_live_…`}</CodeBlock>
      <CodeBlock label={API_SAMPLES.models.title}>{API_SAMPLES.models.code}</CodeBlock>
      <SectionHeading>Endpoints</SectionHeading>
      <P>
        Your own software runs Msanii&apos;s tools over HTTPS: royalty calculations, splits from contracts, split sheets and Zoe. A key is your team&apos;s credential and spends your team&apos;s credits, so it belongs on your servers — never in a browser or a mobile app. Pick an endpoint to read its reference; the console on this page runs each one against your key.
      </P>
      <div className="mb-6 grid gap-2">
        <EndpointRow method="POST" path="/oneclick/v1/royalties" description="Royalty calculation from a statement plus a contract." tag={`${ROYALTIES_PRICE} credits`} onClick={() => select("royalties")} />
        <EndpointRow method="POST" path="/registry/v1/splits" description="Splits: the deal as data, from contract PDFs." tag={`${REGISTRY_PRICE} credits`} onClick={() => select("registry")} />
        <EndpointRow method="POST" path="/splitsheet/v1/documents" description="A finished split sheet, PDF or Word." tag={`${SPLIT_SHEET_PRICE} credits`} onClick={() => select("splitsheet")} />
        <EndpointRow method="POST" path="/zoe/v1/chat/completions" description="Zoe, OpenAI-compatible: point the OpenAI SDK at /zoe/v1." tag={`${ZOE_PRICE} credits`} onClick={() => select("zoe")} />
        <EndpointRow method="GET" path="/zoe/v1/models" description={'Lists the one model, "zoe" — and doubles as the key check.'} tag="free" onClick={() => select("overview", "connect")} />
      </div>
      <Callout type="info" title="Getting access" anchor="getting-access">
        API access is included with Enterprise plans. A team admin creates keys under <strong>Teams → API keys</strong>. A key is shown once at creation and can&apos;t be recovered — only revoked and replaced — so treat it like a password.
      </Callout>
    </div>
  );
};

const ApiRoyaltiesPanel = () => (
  <div>
    <EndpointHero
      method="POST"
      path="/oneclick/v1/royalties"
      title="Royalty calculation"
      stats={[
        ["Price", `${ROYALTIES_PRICE} credits`],
        ["Request", "multipart/form-data"],
        ["Response", "text/event-stream"],
        ["Typical time", "2 s – 2 min"],
        ["Idempotent", "with header"],
      ]}
    >
      Send a royalty statement and the contract that governs it; get back who is owed what. The contract can be PDF files that Msanii&apos;s AI reads, or structured terms you already hold.
    </EndpointHero>

    <Sect tone="in" name="Inputs" sub="What you send" tag="multipart/form-data">
      <P>
        Send exactly one of <code>contracts</code> (PDFs, read by Msanii&apos;s AI) or <code>contract_terms</code> (structured JSON). Sending both, or neither, is a <code>422</code>.
      </P>
      <PropTable
        headers={["Field", "Type", "Description"]}
        rows={[
          ["statement", "file, required", "The royalty statement: .csv, .xlsx or .xls, up to 10 MB."],
          ["contracts", "file, repeated", "Contract PDFs — up to 10 files, 20 MB in total. Send the field once per file; several PDFs are merged into one set of terms."],
          ["contract_terms", "JSON string", "The contract, already structured (below). No AI runs, so it always costs the base price."],
          ["expenses", "JSON string", "Optional recoupable costs, deducted from net-basis shares (below)."],
          ["Idempotency-Key", "header", "Optional. A retry with the same key and the same inputs in the same billing period is charged once. Recommended on every call."],
        ]}
      />

      <SubHeading>The statement file</SubHeading>
      <P>
        One row per song earnings line. Msanii finds the two columns it needs by header name, case-insensitively: a <strong>title</strong> column (a header containing title, song, track, release title…) and an <strong>amount payable</strong> column (net payable, net earnings, net revenue, payable to artist…; failing that, payable, amount, earnings, payment or revenue, as long as the header doesn&apos;t say withheld, deduction, fee, commission or advance). Rows with the same title are summed, so one line per month or per platform needs no pre-aggregation.
      </P>
      <CodeBlock label="statement.csv — the smallest valid statement">{`Title,Net Payable
Blue Sky,1000.00
Red Sun,500.00`}</CodeBlock>

      <SubHeading>Contract terms</SubHeading>
      <P>
        <code>contract_terms</code> is the JSON form of a contract: <code>parties</code> (each with a <code>name</code>, a free-text <code>role</code> such as producer or artist, and optional <code>aliases</code>), <code>works</code> (each with a <code>title</code>, matched to statement titles fuzzily — case, punctuation and suffixes such as &ldquo;(Remix)&rdquo; are tolerated) and <code>royalty_shares</code>. An optional <code>default_basis</code> (&ldquo;gross&rdquo; or &ldquo;net&rdquo;) applies to shares that set none. The Splits endpoint returns this exact shape from a PDF, so a contract parsed once can drive every later statement with no AI.
      </P>
      <PropTable
        headers={["royalty_shares[] field", "Type", "Description"]}
        rows={[
          ["party_name", "string, required", "Must name one of parties."],
          ["royalty_type", "string, required", "What income the share is paid from. The calculation covers streaming and master income, so use master or streaming (digital, DSP revenue and similar also count). Publishing, mechanical, sync and performance shares are ignored — they are paid from different statements."],
          ["percentage", "number, required", "0–100, applied to each matched work."],
          ["basis", "\"gross\" | \"net\"", "gross pays the percentage of the statement amount; net deducts the work's share of expenses first. Falls back to default_basis, then gross."],
          ["terms", "string", "The clause, verbatim if you have it. If it names SoundExchange, a PRO or the MLC as the payer, the share is treated as paid outside this statement and skipped."],
        ]}
      />
      <CodeBlock label="contract_terms">{`{
  "parties": [{"name": "Jane Doe", "role": "producer"}],
  "works": [{"title": "Blue Sky"}, {"title": "Red Sun"}],
  "royalty_shares": [
    {"party_name": "Jane Doe", "royalty_type": "master", "percentage": 50, "basis": "net"}
  ]
}`}</CodeBlock>
      <Callout type="info" title="Lists" anchor="lists">
        <code>parties</code>, <code>works</code> and <code>royalty_shares</code> are JSON arrays inside <code>contract_terms</code>; <code>expenses</code> is its own array, sent as a separate field. One object per entry, as many as the deal has. Two producers on two songs is two parties, two works and two shares. The console on this page builds them from rows and shows the exact body it sends.
      </Callout>

      <SubHeading>Expenses</SubHeading>
      <P>
        <code>expenses</code> is a list of costs recouped before net-basis shares are paid; gross-basis shares ignore them. Each has an <code>amount</code>, an optional <code>description</code> and optional <code>work_titles</code>. A <strong>project-wide</strong> expense (no titles) is spread across every song in the statement in proportion to its earnings. A <strong>tagged</strong> expense is applied in full to each listed song that appears in the statement; a tag that matches nothing is dropped. A song&apos;s net amount never goes below zero.
      </P>
      <CodeBlock label="expenses">{`[{"description": "Mastering", "amount": 300}]`}</CodeBlock>

      <SubHeading>Examples</SubHeading>
      <P>
        Python shown; any HTTP client works — a calculation is one multipart POST answered with server-sent events. Ask your Msanii contact for <code>msanii_partner.py</code>, a one-file client with a smoke test that exercises every endpoint against your key.
      </P>
      <CodeBlock label={API_SAMPLES.royaltiesPdf.title}>{API_SAMPLES.royaltiesPdf.code}</CodeBlock>
      <CodeBlock label={API_SAMPLES.royaltiesTerms.title}>{API_SAMPLES.royaltiesTerms.code}</CodeBlock>
    </Sect>

    <Sect tone="out" name="Outputs" sub="What comes back" tag="text/event-stream">
      <P>
        A <code>200</code> stream. While a PDF is being read the server sends a heartbeat line (<code>: ping</code>) every 15 seconds — ignore lines starting with a colon. Exactly one <code>data:</code> event follows, carrying either a result or an error.
      </P>
      <SubHeading>Result event</SubHeading>
      <P>
        Three sections: the totals, one line per party per matched work, and what the call cost. For the statement and the <code>contract_terms</code> block above, with the 300.00 project-wide expense:
      </P>
      <ResponseExample label="data: — result event" sections={ROYALTIES_RESPONSE} />

      <SubHeading>Unmatched titles</SubHeading>
      <P>
        Statement rows that match no work in the contract are skipped, not errors — a statement can carry more songs than the contract covers. If none of the contract&apos;s works appear in the statement, the run ends with a <code>NO_SONG_MATCHES</code> error event instead of a result, and nothing is charged.
      </P>
    </Sect>
  </div>
);

const ApiRegistryPanel = () => (
  <div>
    <EndpointHero
      method="POST"
      path="/registry/v1/splits"
      title="Splits"
      stats={[
        ["Price", `${REGISTRY_PRICE} credits`],
        ["Request", "multipart/form-data"],
        ["Response", "text/event-stream"],
        ["Typical time", "30 s – 2 min"],
        ["Idempotent", "with header"],
      ]}
    >
      Send contract PDFs and get the deal back as data: the terms in exactly the shape the royalty calculation accepts, plus each party&apos;s master, publishing and SoundExchange percentages. Nothing is stored.
    </EndpointHero>

    <Sect tone="in" name="Inputs" sub="What you send" tag="multipart/form-data">
      <PropTable
        headers={["Field", "Type", "Description"]}
        rows={[
          ["contracts", "file, repeated, required", "Contract PDFs — up to 10 files, 20 MB in total. Send the field once per file; several PDFs are merged into one set of terms."],
          ["main_artist_name", "string", "Optional. The artist the splits are built around: they are kept even at 0 / 0 and named in splits.main_artist, by the name the contract uses. If the name isn't found, main_artist is null and the artist is left out."],
          ["Idempotency-Key", "header", "Optional. The same key with the same files and artist in the same billing period is charged once. Recommended."],
        ]}
      />
      <SubHeading>Example</SubHeading>
      <CodeBlock label={API_SAMPLES.contractTerms.title}>{API_SAMPLES.contractTerms.code}</CodeBlock>
    </Sect>

    <Sect tone="out" name="Outputs" sub="What comes back" tag="text/event-stream">
      <P>
        The same framing as a calculation: heartbeat lines (<code>: ping</code>) while the parse runs, then exactly one <code>data:</code> event — a result or an error.
      </P>
      <SubHeading>Result event</SubHeading>
      <P>
        Two views of one contract. <code>contract_terms</code> is documented under the royalty calculation&apos;s inputs and can be sent there verbatim — parse a contract once, then run every statement against it at the base price with no AI. <code>splits</code> is the Registry&apos;s ownership view.
      </P>
      <ResponseExample label="data: — result event" sections={SPLITS_RESPONSE} />
      <SubHeading>Unreadable contracts</SubHeading>
      <P>
        A scanned image, an encrypted file or an empty PDF ends in an error event with code <code>CONTRACT_UNREADABLE</code>. An error event is never billed.
      </P>
    </Sect>
  </div>
);

const ApiSplitSheetPanel = () => (
  <div>
    <EndpointHero
      method="POST"
      path="/splitsheet/v1/documents"
      title="Split sheet"
      stats={[
        ["Price", `${SPLIT_SHEET_PRICE} credits per document`],
        ["Request", "application/json"],
        ["Response", "PDF or DOCX file"],
        ["Typical time", "seconds"],
        ["Idempotent", "with header"],
      ]}
    >
      The finished split sheet, from the same generator as Msanii&apos;s Split Sheet tool. No AI runs, so a sheet always costs exactly the base price — per document: the PDF and the DOCX of one sheet are two.
    </EndpointHero>

    <Sect tone="in" name="Inputs" sub="What you send" tag="application/json">
      <PropTable
        headers={["Field", "Type", "Description"]}
        rows={[
          ["work_title", "string, required", "Printed on the sheet and used for the file name."],
          ["work_type", "string", "Default single. Printed as given (single, album track…)."],
          ["split_type", "\"publishing\" | \"master\" | \"both\"", "Which sides the sheet covers. Default both."],
          ["date", "string, required", "Printed verbatim, so use the wording you want on the sheet."],
          ["format", "\"pdf\" | \"docx\"", "Default pdf."],
          ["contributors", "array, required", "1–50 lines, below."],
          ["Idempotency-Key", "header", "Optional. The same body in the same billing period is charged once."],
        ]}
      />
      <SubHeading>Contributors</SubHeading>
      <P>
        One line per person on the sheet. The publishing side is the composition: a self-published writer gives one <code>publishing_share</code>; a published writer sets <code>is_published</code> and splits it into <code>writer_share</code> and <code>publisher_share</code>. The master side is the recording: <code>master_percentage</code>, and optionally the <code>label</code>.
      </P>
      <PropTable
        headers={["Field", "Type", "Description"]}
        rows={[
          ["name, role", "string, required", "The person and what they did (Producer, Writer, Artist…)."],
          ["publishing_share", "number", "Their share of the composition, 0–100, when self-published."],
          ["writer_share, publisher_share", "number", "Used instead of publishing_share when is_published is true."],
          ["is_published, publisher_name, publisher_ipi", "", "The contributor's publisher, if they have one."],
          ["ipi_number", "string", "The writer's IPI / CAE number."],
          ["master_percentage", "number", "Their share of the sound recording, 0–100."],
          ["label", "string", "The label on the master side, if any."],
        ]}
      />
      <Callout type="info" title="Lists" anchor="contributor-lists"><code>contributors</code> is a JSON array: one object per person, 1–50 of them. The console on this page builds it from rows.</Callout>
      <CodeBlock label="request body">{SPLIT_SHEET_SAMPLE}</CodeBlock>
      <SubHeading>Example</SubHeading>
      <CodeBlock label={API_SAMPLES.splitSheet.title}>{API_SAMPLES.splitSheet.code}</CodeBlock>
    </Sect>

    <Sect tone="out" name="Outputs" sub="What comes back" tag="application/pdf · docx">
      <P>
        A <code>200</code> whose body is the document itself. Save the body as the file; the headers tell you what it is and what it cost.
      </P>
      <PropTable headers={["Header", "Value", "Notes"]} rows={SPLIT_SHEET_HEADERS} />
      <P>
        A sheet that could not be rendered is a <code>500</code> with code <code>internal_error</code> and a <code>request_id</code> to quote to support. It is never billed.
      </P>
    </Sect>
  </div>
);

const ApiZoePanel = () => (
  <div>
    <EndpointHero
      method="POST"
      path="/zoe/v1/chat/completions"
      title="Zoe (OpenAI-compatible)"
      stats={[
        ["Price", `${ZOE_PRICE} credits per answer`],
        ["Request", "application/json"],
        ["Response", "JSON, or an SSE stream"],
        ["Idempotent", "no"],
      ]}
    >
      Ask Zoe music-business questions from your own software. <code>/zoe/v1</code> speaks the OpenAI chat-completions protocol, so the official OpenAI SDK — or anything built on it — works unchanged.
    </EndpointHero>

    <SectionHeading>OpenAI compatibility</SectionHeading>
    <P>
      Set <code>base_url</code> to your base URL plus <code>/zoe/v1</code>, your key as the API key, and <code>model</code> to <code>zoe</code>. Both plain and streaming (<code>stream: true</code>, ending in <code>data: [DONE]</code>) responses use OpenAI&apos;s shapes, and <code>GET /zoe/v1/models</code> lists the one model for SDKs that probe it.
    </P>
    <CodeBlock label={API_SAMPLES.zoe.title}>{API_SAMPLES.zoe.code}</CodeBlock>

    <SectionHeading>Request and response</SectionHeading>
    <P>
      Zoe on the API is <strong>stateless</strong>: she keeps no memory between calls and has no access to documents stored in Msanii, so send the whole context — earlier turns, a contract&apos;s text — in <code>messages</code>, exactly as you would with OpenAI. Other OpenAI fields are ignored rather than rejected, so a stock client never fails on them.
    </P>
    <PropTable
      headers={["Field", "Type", "Description"]}
      rows={[
        ["model", "string, required", "Must be \"zoe\". Anything else is a 404 model_not_found."],
        ["messages", "array, required", "Roles system, user and assistant; content as a string or text parts. Up to 100 messages and 100,000 characters in total."],
        ["stream", "boolean", "false returns one chat.completion; true streams chat.completion.chunk frames, then data: [DONE]."],
        ["temperature", "number", "0–2, passed through."],
        ["max_tokens", "integer", "1–4,000; also the default."],
      ]}
    />
    <ResponseExample label="200 — chat.completion" sections={ZOE_RESPONSE} />

    <SectionHeading>What Zoe answers</SectionHeading>
    <P>
      Music-business questions — deals, royalties, rights, publishing, management — from general knowledge and from anything you include in the conversation. She politely declines unrelated topics. Every delivered answer is billed; a failed one (<code>502 zoe_failed</code>, or an error frame before <code>[DONE]</code> on a stream) is not.
    </P>
  </div>
);

const ApiErrorsPanel = () => (
  <div>
    <SectionHeading>HTTP status codes</SectionHeading>
    <P>
      Before any work starts, a failure is a plain HTTP error with a JSON body of the form <code>{'{"detail": {"code": "…"}}'}</code>. Nothing is charged.
    </P>
    <PropTable
      headers={["Status", "code", "Meaning"]}
      rows={[
        ["401", "invalid_key", "Missing, unknown, revoked or expired key — or the team's API access has been turned off."],
        ["402", "insufficient_credits", "The team's balance is below the price of a run; the body also carries price and balance. No work was started."],
        ["404", "model_not_found", "Zoe only: a model other than \"zoe\" was requested."],
        ["413", "file_too_large / too_many_contracts", "Statement over 10 MB, contracts over 20 MB in total, or more than 10 files."],
        ["422", "invalid_request", "Both or neither of contracts / contract_terms, a non-PDF contract, malformed JSON, no contracts on a parse, a split sheet body that fails validation — or, for Zoe, an empty or over-long messages list."],
        ["500", "internal_error", "A split sheet couldn't be rendered. Quote request_id to support."],
        ["502", "zoe_failed", "Zoe didn't answer. Retry. On a Zoe stream this arrives as an error frame before [DONE] instead."],
      ]}
    />

    <SectionHeading>Stream error codes</SectionHeading>
    <P>
      Inside a calculation or splits stream, the failure arrives as an error event — HTTP is already <code>200</code> by then. <code>message</code> and <code>suggestion</code> are safe to show a person; an error event is never billed.
    </P>
    <ResponseExample label="data: — error event" sections={ERROR_EVENT_RESPONSE} />
    <PropTable
      headers={["code", "details", "Meaning"]}
      rows={[
        ["STATEMENT_UNSUPPORTED_FORMAT", "", "Not a CSV or Excel file."],
        ["STATEMENT_EMPTY", "", "No earnings rows could be read."],
        ["STATEMENT_COLUMNS_UNDETECTABLE", "available_columns", "The title or amount column could not be identified."],
        ["NO_WORKS_IN_CONTRACT", "", "The contract lists no songs."],
        ["NO_ROYALTY_SHARES_IN_CONTRACT", "", "The contract has no percentage splits."],
        ["NO_STREAMING_EARNABLE_SHARES", "excluded_payor_count", "Splits exist, but none are paid from streaming or master income."],
        ["NO_SONG_MATCHES", "contract_works, statement_songs", "No contract work appears in the statement."],
        ["CONTRACT_UNREADABLE", "reason", "Splits only: a scanned image, an encrypted or empty file."],
        ["internal_error", "request_id", "Something failed on Msanii's side. Quote the request id to support."],
      ]}
    />
  </div>
);

const ApiBillingPanel = () => (
  <div>
    <SectionHeading>How billing works</SectionHeading>
    <ul className="mb-3 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-muted-foreground">
      <li><strong className="text-foreground">Every response says what it cost.</strong> <code>billing.credits</code> in a result event, a Zoe body or the stream&apos;s final stop frame; the <code>Msanii-Credits</code> header on a document. A replay under the same Idempotency-Key reports 0 and <code>replayed: true</code>. No token counts are returned.</li>
      <li><strong className="text-foreground">You pay only for a result you received.</strong> The charge is applied after the result event, the document, or the last chunk of a Zoe stream is returned. A run that fails before the results arrived, costs nothing.</li>
      <li><strong className="text-foreground">Each call has a base price</strong> — at the time of writing {ROYALTIES_PRICE} credits per calculation, {REGISTRY_PRICE} per splits run, {SPLIT_SHEET_PRICE} per split sheet and {ZOE_PRICE} per Zoe answer, shown as <code>price</code> in a 402. A calculation or parse over an unusually large set of PDFs, or a very long Zoe exchange, can cost more; a <code>contract_terms</code> run and a split sheet always cost exactly the base.</li>
      <li><strong className="text-foreground">Send an Idempotency-Key on calculations, parses and sheets.</strong> The same key with the same inputs is charged once per billing period and returns the same result. Without it, every call is billed. Zoe answers have no idempotency: every delivered answer is billed.</li>
      <li><strong className="text-foreground">The balance is checked first.</strong> A 402 comes back before any work starts, and its body carries the price and the balance. Only a team admin can add credits, from the team page.</li>
      <li><code>/zoe/v1/models</code> is always free.</li>
    </ul>

    <SectionHeading>Limits</SectionHeading>
    <PropTable
      headers={["Limit", "Value", "Notes"]}
      rows={[
        ["Contract files per request", "10", "PDF only."],
        ["Contracts, total size", "20 MB", ""],
        ["Statement size", "10 MB", ""],
        ["Calculation time", "seconds – 2 min", "Structured terms return in seconds; a PDF run typically takes 30–120 s. Keep read timeouts above 60 s — heartbeats keep the connection alive."],
        ["Splits run time", "30 s – 2 min", "Heartbeats keep the connection alive."],
        ["Split sheet contributors", "50", "Per document."],
        ["Zoe messages per call", "100", "100,000 characters in total across them."],
        ["Zoe max_tokens", "4,000", "Also the default."],
      ]}
    />

    <SectionHeading>Versioning</SectionHeading>
    <P>
      Everything under <code>/oneclick/v1</code>, <code>/registry/v1</code>, <code>/splitsheet/v1</code> and <code>/zoe/v1</code> is frozen: fields may be added to responses, but existing fields, codes and meanings won&apos;t change. Breaking changes ship as a <code>v2</code>.
    </P>
  </div>
);

const API_PANELS: Record<ApiTabId, React.FC> = {
  overview: ApiOverviewPanel,
  royalties: ApiRoyaltiesPanel,
  registry: ApiRegistryPanel,
  splitsheet: ApiSplitSheetPanel,
  zoe: ApiZoePanel,
  errors: ApiErrorsPanel,
  billing: ApiBillingPanel,
};

const ApiContent = () => {
  const { tab, select } = useContext(ApiTabContext);
  const Panel = API_PANELS[tab];
  return (
    <div>
      <ApiTabStrip tab={tab} onSelect={(t) => select(t)} />
      <Panel />
    </div>
  );
};

const SECTION_CONTENT: Record<string, React.FC> = {
  "getting-started": GettingStartedContent,
  portfolio: PortfolioContent,
  "project-detail": ProjectDetailContent,
  "work-detail": WorkDetailContent,
  "rights-registry": RightsRegistryContent,
  oneclick: OneClickContent,
  "royalty-tracking": RoyaltyTrackingContent,
  zoe: ZoeContent,
  "split-sheet": SplitSheetContent,
  "artist-management": ArtistManagementContent,
  workspace: WorkspaceContent,
  integrations: IntegrationsContent,
  api: ApiContent,
  credits: CreditsContent,
  "best-practices": BestPracticesContent,
};

// ---------------------------------------------------------------------------
// Page navigation + "was this helpful" footer
// ---------------------------------------------------------------------------

function PageNav({ prev, next, onSelect }: {
  prev: SectionMeta | null; next: SectionMeta | null; onSelect: (id: string) => void;
}) {
  return (
    <div className="mt-10 grid gap-3.5 sm:grid-cols-2">
      {prev ? (
        <button
          onClick={() => onSelect(prev.id)}
          className="rounded-xl border border-border bg-card p-4 text-left transition-colors hover:border-primary/40 hover:bg-muted/40"
        >
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground"><ArrowLeft className="h-3 w-3" /> Previous</div>
          <div className="mt-1.5 text-[15px] font-semibold text-foreground">{prev.label}</div>
        </button>
      ) : <span />}
      {next ? (
        <button
          onClick={() => onSelect(next.id)}
          className="rounded-xl border border-border bg-card p-4 text-right transition-colors hover:border-primary/40 hover:bg-muted/40"
        >
          <div className="flex items-center justify-end gap-1.5 text-xs text-muted-foreground">Next <ArrowRight className="h-3 w-3" /></div>
          <div className="mt-1.5 text-[15px] font-semibold text-foreground">{next.label}</div>
        </button>
      ) : <span />}
    </div>
  );
}

function Helpful() {
  const [voted, setVoted] = useState<"yes" | "no" | null>(null);
  return (
    <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-6">
      <span className="text-sm text-muted-foreground">{voted ? "Thanks for the feedback." : "Was this page helpful?"}</span>
      {!voted && (
        <div className="flex gap-2">
          <button onClick={() => setVoted("yes")} className="inline-flex items-center gap-1.5 rounded-full border border-border px-4 py-1.5 text-[13px] font-medium text-foreground transition-colors hover:border-primary hover:text-primary">
            <ThumbsUp className="h-3.5 w-3.5" /> Yes
          </button>
          <button onClick={() => setVoted("no")} className="inline-flex items-center gap-1.5 rounded-full border border-border px-4 py-1.5 text-[13px] font-medium text-foreground transition-colors hover:border-primary hover:text-primary">
            <ThumbsDown className="h-3.5 w-3.5" /> No
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

interface DocHeading { id: string; label: string; level: number; }

const Documentation = () => {
  const navigate = useNavigate();
  const goBack = useSmartBack("/dashboard");
  const [searchParams] = useSearchParams();
  const { user, signOut } = useAuth();
  // Deep-linkable: /docs?section=oneclick opens straight to that section.
  const [activeSection, setActiveSection] = useState(() => {
    const s = searchParams.get("section");
    return s && SECTION_INDEX.has(s) ? s : "getting-started";
  });
  const [navQuery, setNavQuery] = useState("");
  // API reference: the tab (deep-linkable as ?tab=), the sidebar row that lit
  // it, and the console's key — page-level so they survive tab switches.
  const [apiTab, setApiTab] = useState<ApiTabId>(() => parseApiTab(searchParams.get("tab")));
  const [apiNavKey, setApiNavKey] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const [folds, setFolds] = useState({ platform: true, api: true });
  const wide = useWide();

  // Keep the active section in sync if the ?section= param changes while the
  // page is already mounted (e.g. a footer link clicked from elsewhere).
  useEffect(() => {
    const s = searchParams.get("section");
    if (s && SECTION_INDEX.has(s)) {
      setActiveSection(s);
      if (s === "api") {
        setApiTab(parseApiTab(searchParams.get("tab")));
        setApiNavKey(null);
      }
      window.scrollTo({ top: 0 });
    }
  }, [searchParams]);

  const articleRef = useRef<HTMLDivElement>(null);
  const [headings, setHeadings] = useState<DocHeading[]>([]);
  const [activeHeading, setActiveHeading] = useState("");

  const handleSelectSection = useCallback((id: string) => {
    setActiveSection(id);
    window.scrollTo({ top: 0 });
  }, []);

  const selectApiTab = useCallback((tab: ApiTabId, anchor?: string, navKey?: string) => {
    setActiveSection("api");
    setApiTab(tab);
    setApiNavKey(navKey ?? null);
    if (anchor) {
      // The panel renders on this commit; the anchor exists by the next frame.
      requestAnimationFrame(() => document.getElementById(anchor)?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } else {
      window.scrollTo({ top: 0 });
    }
  }, []);
  const apiTabState = useMemo<ApiTabState>(() => ({ tab: apiTab, select: selectApiTab }), [apiTab, selectApiTab]);
  const isApi = activeSection === "api";

  const currentIndex = useMemo(() => SECTION_INDEX.get(activeSection) ?? 0, [activeSection]);
  const prevSection = currentIndex > 0 ? SECTIONS[currentIndex - 1] : null;
  const nextSection = currentIndex < SECTIONS.length - 1 ? SECTIONS[currentIndex + 1] : null;
  const activeData = SECTIONS[currentIndex];
  const ActiveContent = SECTION_CONTENT[activeData.id];

  // Build the "On this page" rail from the rendered section's headings.
  useLayoutEffect(() => {
    const root = articleRef.current;
    if (!root) return;
    const nodes = Array.from(root.querySelectorAll<HTMLElement>("[data-doc-heading]"));
    const hs = nodes
      .filter((n) => n.id)
      .map((n) => ({ id: n.id, label: (n.textContent || "").trim(), level: Number(n.getAttribute("data-doc-level") || "2") }));
    setHeadings(hs);
    setActiveHeading(hs[0]?.id || "");
  }, [activeSection, apiTab]);

  // Scroll-spy: highlight the heading currently in view.
  useEffect(() => {
    if (headings.length === 0) return;
    const obs = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActiveHeading(visible[0].target.id);
      },
      { rootMargin: "-90px 0px -70% 0px", threshold: 0 }
    );
    headings.forEach((h) => {
      const el = document.getElementById(h.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [headings]);

  // Sidebar folds filtered by the search box.
  const q = navQuery.trim().toLowerCase();
  const platformGroups = useMemo(
    () =>
      PLATFORM_GROUPS
        .map((g) => ({
          group: g.group,
          items: SECTIONS.filter((s) => s.group === g.group && (!q || s.label.toLowerCase().includes(q))),
        }))
        .filter((g) => g.items.length > 0),
    [q]
  );
  const apiGroups = useMemo(
    () =>
      API_NAV
        .map((g) => ({ group: g.group, items: g.items.filter((i) => !q || "api".includes(q) || i.label.toLowerCase().includes(q)) }))
        .filter((g) => g.items.length > 0),
    [q]
  );

  const apiConsole = <PartnerApiConsole kind={CONSOLE_KIND[apiTab]} apiKey={apiKey} onApiKeyChange={setApiKey} />;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          {user ? (
            <>
              <div className="flex items-center gap-3">
                <Button variant="ghost" size="sm" onClick={goBack} className="text-muted-foreground hover:text-foreground">
                  <ArrowLeft className="w-4 h-4 mr-1" /> Back
                </Button>
                <div className="w-px h-5 bg-border" />
                <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => navigate("/")}>
                  <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                    <Music className="w-4 h-4 text-primary-foreground" />
                  </div>
                  <span className="text-base font-bold text-foreground hidden sm:inline">Msanii</span>
                </div>
                <Badge variant="outline" className="gap-1 text-xs hidden sm:flex">
                  <BookOpen className="w-3 h-3" /> Docs
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => navigate("/dashboard")}>Dashboard</Button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-sm font-medium text-primary cursor-pointer hover:bg-primary/30 transition-colors">
                      {(user.email ?? "U")[0].toUpperCase()}
                    </div>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56" align="end" forceMount>
                    <DropdownMenuLabel className="font-normal">
                      <p className="text-xs leading-none text-muted-foreground">{user.email}</p>
                    </DropdownMenuLabel>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={() => navigate("/profile")}>
                      <User className="mr-2 h-4 w-4" />
                      <span>Profile Settings</span>
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem onClick={async () => { await signOut(); navigate("/"); }}>
                      <LogOut className="mr-2 h-4 w-4" />
                      <span>Log out</span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => navigate("/")}>
                <Music className="w-8 h-8" />
                <span className="text-xl font-bold text-foreground">Msanii</span>
              </div>
              <Button onClick={() => navigate("/auth")}>Sign In</Button>
            </>
          )}
        </div>
      </header>

      {/* Mobile section nav (flat, horizontal) */}
      <div className="lg:hidden sticky top-[57px] z-30 border-b border-border bg-background/95 backdrop-blur overflow-x-auto">
        <div className="flex gap-1.5 p-3">
          {SECTIONS.map((s) => {
            const Icon = s.icon;
            const on = activeSection === s.id;
            return (
              <button
                key={s.id}
                onClick={() => handleSelectSection(s.id)}
                className={`flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-xs transition-colors ${on ? "bg-primary/10 font-semibold text-primary" : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"}`}
              >
                <Icon className="h-3.5 w-3.5" /> {s.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* 3-column docs layout */}
      <div className={`mx-auto grid w-full max-w-[1360px] grid-cols-1 lg:grid-cols-[284px_minmax(0,1fr)] ${isApi ? "xl:grid-cols-[284px_minmax(0,1fr)_372px]" : "xl:grid-cols-[284px_minmax(0,1fr)_240px]"}`}>
        {/* Sidebar */}
        <nav className="hidden lg:block sticky top-[57px] h-[calc(100vh-57px)] overflow-y-auto border-r border-border px-3.5 py-5">
          <div className="mb-6 flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3 py-2">
            <Search className="h-3.5 w-3.5 text-muted-foreground" />
            <input
              value={navQuery}
              onChange={(e) => setNavQuery(e.target.value)}
              placeholder="Filter docs"
              className="w-full bg-transparent text-[13px] text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
          </div>
          {platformGroups.length > 0 && (
            <Fold label="Platform" icon={LayoutGrid} open={folds.platform} onToggle={() => setFolds((f) => ({ ...f, platform: !f.platform }))}>
              {platformGroups.map((g) => (
                <NavGroup key={g.group} label={g.group}>
                  {g.items.map((s) => {
                    const Icon = s.icon;
                    const on = activeSection === s.id;
                    return (
                      <button key={s.id} onClick={() => handleSelectSection(s.id)} className={navRowCls(on)}>
                        <Icon className={`h-[15px] w-[15px] shrink-0 ${on ? "" : "opacity-75"}`} /> {s.label}
                      </button>
                    );
                  })}
                </NavGroup>
              ))}
            </Fold>
          )}
          {apiGroups.length > 0 && (
            <Fold label="API" icon={KeyRound} badge="v1" open={folds.api} onToggle={() => setFolds((f) => ({ ...f, api: !f.api }))}>
              {apiGroups.map((g) => (
                <NavGroup key={g.group} label={g.group}>
                  {g.items.map((item) => {
                    const on = isApi && (apiNavKey ?? API_NAV_DEFAULT_KEY[apiTab]) === item.key;
                    const Icon = item.icon;
                    const methodCls = on
                      ? "text-primary"
                      : item.method === "POST"
                        ? "text-[hsl(var(--pay-sched-fg))]"
                        : item.method === "GET"
                          ? "text-primary"
                          : "text-muted-foreground";
                    return (
                      <button key={item.key} onClick={() => selectApiTab(item.tab, item.anchor, item.key)} className={navRowCls(on)}>
                        {item.method ? (
                          <>
                            <span className={`w-[34px] shrink-0 text-right font-mono text-[9.5px] font-semibold ${methodCls}`}>{item.method}</span>
                            <span className="min-w-0 flex-1 truncate font-mono text-[12.5px]">{item.label}</span>
                          </>
                        ) : (
                          <>
                            {Icon && <Icon className={`h-[15px] w-[15px] shrink-0 ${on ? "" : "opacity-75"}`} />} {item.label}
                          </>
                        )}
                      </button>
                    );
                  })}
                </NavGroup>
              ))}
            </Fold>
          )}
          {platformGroups.length === 0 && apiGroups.length === 0 && <div className="px-3 text-sm text-muted-foreground">No matches.</div>}
          <div className="mt-2 border-t border-border px-3 pt-4">
            <button onClick={() => navigate("/tools/zoe")} className="text-[13px] text-muted-foreground transition-colors hover:text-foreground">
              Can't find it? Ask Zoe →
            </button>
          </div>
        </nav>

        {/* Article */}
        <main ref={articleRef} className="min-w-0 px-5 py-10 sm:px-10 lg:px-14">
          <div className="mx-auto max-w-[760px]">
            <nav className="mb-4 flex items-center gap-2 text-[13px] text-muted-foreground">
              <span>Docs</span>
              <span className="opacity-50">/</span>
              <span>{activeData.group}</span>
              <span className="opacity-50">/</span>
              <span className="font-medium text-foreground">{activeData.label}</span>
            </nav>
            <p className="mb-3 font-mono text-[12px] font-semibold uppercase tracking-[0.14em] text-primary">{activeData.group}</p>
            <h1 className="mb-3 text-4xl font-bold tracking-tight text-foreground sm:text-[42px] sm:leading-[1.05]">{activeData.label}</h1>
            <p className="text-[17px] leading-relaxed text-muted-foreground">{SECTION_DESCRIPTIONS[activeData.id]}</p>

            <div className="mt-8">
              <SelectSectionContext.Provider value={handleSelectSection}>
                <ApiTabContext.Provider value={apiTabState}>
                  <ActiveContent />
                </ApiTabContext.Provider>
              </SelectSectionContext.Provider>
            </div>
            {isApi && !wide && <div className="mt-8">{apiConsole}</div>}

            <PageNav prev={prevSection} next={nextSection} onSelect={handleSelectSection} />
            <Helpful />
          </div>
        </main>

        {/* On this page */}
        <aside className={`hidden xl:block sticky top-[57px] h-[calc(100vh-57px)] overflow-y-auto ${isApi ? "border-l border-border px-5 py-6" : "px-6 py-10"}`}>
          {isApi && wide && apiConsole}
          {headings.length > 0 && (
            <>
              <div className={`mb-3 text-[11px] font-bold uppercase tracking-wider text-muted-foreground ${isApi ? "mt-6" : ""}`}>
                {isApi ? "On this tab" : "On this page"}
              </div>
              <div className="grid gap-1 border-l border-border">
                {headings.map((h) => {
                  const on = activeHeading === h.id;
                  return (
                    <a
                      key={h.id}
                      href={`#${h.id}`}
                      onClick={(e) => {
                        e.preventDefault();
                        document.getElementById(h.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
                      }}
                      className={`-ml-px border-l-2 py-1 text-[12.5px] leading-snug transition-colors ${on ? "border-primary font-semibold text-primary" : "border-transparent text-muted-foreground hover:text-foreground"} ${h.level === 3 ? "pl-6" : "pl-3.5"}`}
                    >
                      {h.label}
                    </a>
                  );
                })}
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
};

export default Documentation;
