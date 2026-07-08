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
  DollarSign, Receipt, BarChart3, SplitSquareHorizontal,
} from "lucide-react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
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

const NAV_GROUPS: { group: string; ids: string[] }[] = [
  { group: "Getting started", ids: ["getting-started"] },
  { group: "Roster & projects", ids: ["artist-management", "portfolio", "project-detail", "work-detail", "rights-registry"] },
  { group: "Tools", ids: ["oneclick", "royalty-tracking", "zoe", "split-sheet"] },
  { group: "Platform", ids: ["workspace", "integrations", "best-practices"] },
];

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
  integrations: "Connect Msanii to Google Drive and Slack — and see what's coming next.",
  "best-practices": "Tips for getting the most out of Msanii.",
};

// Lets content-level cards switch the active section (avoids prop drilling).
const SelectSectionContext = createContext<(id: string) => void>(() => {});

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

function Callout({ type = "info", title, children }: {
  type?: "info" | "tip" | "important"; title?: string; children: React.ReactNode;
}) {
  const s = CALLOUT_STYLES[type];
  const IconEl = s.icon;
  return (
    <div className={`my-5 flex gap-3.5 rounded-r-xl border-l-[3px] ${s.bar} ${s.bg} py-4 pl-4 pr-5`}>
      <div className={`mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md ${s.chip}`}>
        <IconEl className="h-3.5 w-3.5" />
      </div>
      <div className="min-w-0">
        <div className={`mb-1 text-[11px] font-bold uppercase tracking-wider ${s.label}`}>{title || s.fallback}</div>
        <div className="text-sm leading-relaxed text-foreground">{children}</div>
      </div>
    </div>
  );
}

function PropTable({ rows }: { rows: [string, string, string][] }) {
  return (
    <div className="my-5 overflow-hidden rounded-xl border border-border">
      <table className="w-full border-collapse text-[13.5px]">
        <thead>
          <tr className="bg-muted/50 text-left">
            {["Item", "Status", "Description"].map((h) => (
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
  { target: "workspace", name: "Workspace", icon: LayoutGrid, desc: "Boards and a calendar wired into Drive and Slack." },
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
        <FeatureCard icon={Pencil} title="Owner actions" description="Register the work, export proof of ownership, invite collaborators, edit metadata, or delete it." color="purple" />
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
        The Metadata Registry is your ownership-tracking dashboard. It shows everything you own and everything you're involved in across all projects. (You create works in Project Detail, not here.)
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "My Works", desc: "Works you own" },
          { label: "Registered", desc: "Fully confirmed" },
          { label: "Pending", desc: "Awaiting response" },
          { label: "Shared with Me", desc: "Works you're on" },
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
        The registry opens on a dashboard of every work you're involved in. Toggle between <strong>My Works</strong> and <strong>Shared with Me</strong>, and use the stat cards across the top to filter the list.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Folder} title="My Works / Shared with Me" description="Switch between works you own and works other people have shared with you." color="purple" />
        <FeatureCard icon={Zap} title="Stat filters" description="Total, Released, Unreleased, Need attention, and Shared with you — click a card to filter the list." color="amber" />
        <FeatureCard icon={Info} title="Search & sort" description="Find a work by title and sort by recently added, title A–Z, or release date." color="teal" />
        <FeatureCard icon={Shield} title="Status at a glance" description="Every row shows its registry status — Draft, Pending, or Registered." color="blue" />
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
      <SectionHeading><Lock className="w-4 h-4" /> Enhanced invite form</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        When inviting a collaborator, the form captures everything needed:
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <FeatureCard icon={Users} title="Email & name" description="Who you're inviting — they'll receive a rich email with all the details." color="blue" />
        <FeatureCard icon={Shield} title="Role & stakes" description="Set master %, publishing %, or both, and choose their role (producer, songwriter, etc.)." color="emerald" />
        <FeatureCard icon={FileText} title="Notes & terms" description="Add context about the arrangement. The collaborator sees this in the invite email." color="amber" />
        <FeatureCard icon={Lock} title="Access control" description="Files become accessible only after acceptance. The invite email contains all decision-making info." color="red" />
      </div>
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
        <Step num={5} title="Export & share" isLast>Download the results as <strong>CSV</strong> or <strong>Excel</strong>, view the payout distribution chart, or send the result straight to <strong>Google Drive</strong> or <strong>Slack</strong>.</Step>
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
        <FeatureCard icon={Plug} title="Integrations" description="Connect Google Drive and Slack to move files, send notifications, and sync tasks." color="blue" />
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
          ["Slack", "Coming soon", "Send task and royalty notifications to a channel, share OneClick breakdowns, and capture @mentions."],
        ]}
      />
    </div>
    <div>
      <SectionHeading>How to connect</SectionHeading>
      <CodeBlock label="Workspace → Settings → Integrations">{`Connect a service in two clicks:

  ✓ Google Drive          connected
  + Slack                 coming soon`}</CodeBlock>
    </div>
    <Callout type="tip" title="Spotify metadata — no setup needed">
      When you mark a work as <strong>Released</strong>, Msanii can pull its ISRC, UPC, release date, and cover art from Spotify automatically. There's nothing to connect — it's built into the Metadata Registry.
    </Callout>
    <Callout type="info" title="What's live today">
      Google Drive is connected today — import files into a project and export them back to Drive. Slack is on the way. There's no public API; everything happens inside Msanii.
    </Callout>
  </div>
);

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
  const [searchParams] = useSearchParams();
  const { user, signOut } = useAuth();
  // Deep-linkable: /docs?section=oneclick opens straight to that section.
  const [activeSection, setActiveSection] = useState(() => {
    const s = searchParams.get("section");
    return s && SECTION_INDEX.has(s) ? s : "getting-started";
  });
  const [navQuery, setNavQuery] = useState("");

  // Keep the active section in sync if the ?section= param changes while the
  // page is already mounted (e.g. a footer link clicked from elsewhere).
  useEffect(() => {
    const s = searchParams.get("section");
    if (s && SECTION_INDEX.has(s)) {
      setActiveSection(s);
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
  }, [activeSection]);

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

  // Sidebar groups filtered by the search box.
  const filteredGroups = useMemo(() => {
    const q = navQuery.trim().toLowerCase();
    return NAV_GROUPS
      .map((g) => ({
        group: g.group,
        items: SECTIONS.filter((s) => s.group === g.group && (!q || s.label.toLowerCase().includes(q))),
      }))
      .filter((g) => g.items.length > 0);
  }, [navQuery]);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          {user ? (
            <>
              <div className="flex items-center gap-3">
                <Button variant="ghost" size="sm" onClick={() => navigate(-1)} className="text-muted-foreground hover:text-foreground">
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
      <div className="mx-auto grid w-full max-w-[1340px] grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)] xl:grid-cols-[260px_minmax(0,1fr)_240px]">
        {/* Sidebar */}
        <nav className="hidden lg:block sticky top-[57px] h-[calc(100vh-57px)] overflow-y-auto border-r border-border px-5 py-7">
          <div className="mb-6 flex items-center gap-2 rounded-lg border border-border bg-muted/50 px-3 py-2">
            <Search className="h-3.5 w-3.5 text-muted-foreground" />
            <input
              value={navQuery}
              onChange={(e) => setNavQuery(e.target.value)}
              placeholder="Filter docs"
              className="w-full bg-transparent text-[13px] text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
          </div>
          {filteredGroups.map((g) => (
            <div key={g.group} className="mb-6">
              <div className="mb-2.5 px-3 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">{g.group}</div>
              <div className="grid gap-0.5">
                {g.items.map((s) => {
                  const Icon = s.icon;
                  const on = activeSection === s.id;
                  return (
                    <button
                      key={s.id}
                      onClick={() => handleSelectSection(s.id)}
                      className={`flex items-center gap-2.5 rounded-lg px-3 py-1.5 text-left text-sm transition-colors ${on ? "bg-primary/10 font-semibold text-primary" : "font-medium text-muted-foreground hover:bg-muted/60 hover:text-foreground"}`}
                    >
                      <Icon className={`h-4 w-4 shrink-0 ${on ? "" : "opacity-75"}`} /> {s.label}
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
          {filteredGroups.length === 0 && <div className="px-3 text-sm text-muted-foreground">No matches.</div>}
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
                <ActiveContent />
              </SelectSectionContext.Provider>
            </div>

            <PageNav prev={prevSection} next={nextSection} onSelect={handleSelectSection} />
            <Helpful />
          </div>
        </main>

        {/* On this page */}
        <aside className="hidden xl:block sticky top-[57px] h-[calc(100vh-57px)] overflow-y-auto px-6 py-10">
          {headings.length > 0 && (
            <>
              <div className="mb-3 text-[11px] font-bold uppercase tracking-wider text-muted-foreground">On this page</div>
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
