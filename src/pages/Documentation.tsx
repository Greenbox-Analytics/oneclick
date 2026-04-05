import { useState, useMemo, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Music, ArrowLeft, ArrowRight, BookOpen, Calculator, Bot, FileText,
  Users, LayoutGrid, Folder, FolderOpen, Shield, Lightbulb, Rocket,
  Info, CheckCircle2, Zap, Volume2, StickyNote, Settings, Lock,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";

// ---------------------------------------------------------------------------
// Types & Constants — hoisted to module scope (rendering-hoist-jsx)
// ---------------------------------------------------------------------------

interface Section {
  id: string;
  label: string;
  icon: React.ElementType;
  color: string; // accent color class per section
}

const SECTIONS: Section[] = [
  { id: "getting-started", label: "Getting Started", icon: Rocket, color: "emerald" },
  { id: "portfolio", label: "Portfolio", icon: Folder, color: "blue" },
  { id: "project-detail", label: "Project Detail", icon: FolderOpen, color: "purple" },
  { id: "rights-registry", label: "Rights Registry", icon: Shield, color: "amber" },
  { id: "oneclick", label: "OneClick", icon: Calculator, color: "teal" },
  { id: "zoe", label: "Zoe AI", icon: Bot, color: "indigo" },
  { id: "split-sheet", label: "Split Sheet", icon: FileText, color: "red" },
  { id: "artist-management", label: "Artist Management", icon: Users, color: "orange" },
  { id: "workspace", label: "Workspace", icon: LayoutGrid, color: "sky" },
  { id: "best-practices", label: "Best Practices", icon: Lightbulb, color: "amber" },
];

// Section index map for O(1) lookup (js-index-maps)
const SECTION_INDEX = new Map(SECTIONS.map((s, i) => [s.id, i]));

const SECTION_DESCRIPTIONS: Record<string, string> = {
  "getting-started": "Get up and running with Msanii in just a few steps.",
  portfolio: "Browse your projects as a card grid grouped by year and artist.",
  "project-detail": "The central hub for a project — works, files, audio, members, notes, and settings.",
  "rights-registry": "Track ownership, manage collaborator invitations, and confirm rights across all your works.",
  oneclick: "Calculate royalty splits and payments from your contracts in one click using AI.",
  zoe: "Your AI-powered contract assistant. Ask questions and get answers with source citations.",
  "split-sheet": "Create professional split sheet agreements and generate clean PDFs ready for signing.",
  "artist-management": "Manage your complete artist roster with profiles, streaming links, and organized projects.",
  workspace: "Your project management hub with Kanban boards, calendar views, and integrations.",
  "best-practices": "Tips for getting the most out of Msanii.",
};

// Accent color map for section headers — avoids dynamic string construction
const ACCENT_STYLES: Record<string, { iconBg: string; iconText: string; bar: string }> = {
  emerald: { iconBg: "bg-emerald-500/10", iconText: "text-emerald-500", bar: "from-emerald-500 via-emerald-500/60" },
  blue: { iconBg: "bg-blue-500/10", iconText: "text-blue-500", bar: "from-blue-500 via-blue-500/60" },
  purple: { iconBg: "bg-purple-500/10", iconText: "text-purple-500", bar: "from-purple-500 via-purple-500/60" },
  amber: { iconBg: "bg-amber-500/10", iconText: "text-amber-500", bar: "from-amber-500 via-amber-500/60" },
  teal: { iconBg: "bg-teal-500/10", iconText: "text-teal-500", bar: "from-teal-500 via-teal-500/60" },
  indigo: { iconBg: "bg-indigo-500/10", iconText: "text-indigo-500", bar: "from-indigo-500 via-indigo-500/60" },
  red: { iconBg: "bg-red-500/10", iconText: "text-red-500", bar: "from-red-500 via-red-500/60" },
  orange: { iconBg: "bg-orange-500/10", iconText: "text-orange-500", bar: "from-orange-500 via-orange-500/60" },
  sky: { iconBg: "bg-sky-500/10", iconText: "text-sky-500", bar: "from-sky-500 via-sky-500/60" },
};

// ---------------------------------------------------------------------------
// Reusable UI primitives — defined OUTSIDE components (rerender-no-inline-components)
// ---------------------------------------------------------------------------

function Step({ num, title, children, isLast = false }: { num: number; title: string; children: React.ReactNode; isLast?: boolean }) {
  return (
    <div className="relative pl-12 pb-8 last:pb-0">
      {/* Connector line — starts below the circle, ends at next circle */}
      {!isLast && (
        <div className="absolute left-[17px] top-[36px] bottom-0 w-0.5 bg-primary/20" />
      )}
      {/* Number circle */}
      <div className="absolute left-0 top-0 w-9 h-9 rounded-full bg-primary/10 border-2 border-primary/30 flex items-center justify-center text-sm font-bold text-primary z-10">
        {num}
      </div>
      <div className="pt-1">
        <h3 className="text-[15px] font-semibold text-foreground mb-1.5">{title}</h3>
        <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
      </div>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description, color = "blue" }: {
  icon: React.ElementType; title: string; description: string; color?: string;
}) {
  const bgClass = `bg-${color}-500/10`;
  const textClass = `text-${color}-600 dark:text-${color}-400`;
  return (
    <div className="flex gap-3 p-3.5 rounded-xl border border-border/50 bg-card/50 hover:bg-muted/40 transition-colors">
      <div className={`w-9 h-9 rounded-lg ${bgClass} flex items-center justify-center shrink-0`}>
        <Icon className={`w-4 h-4 ${textClass}`} />
      </div>
      <div className="min-w-0">
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">{description}</p>
      </div>
    </div>
  );
}

function Callout({ type = "info", title, children }: {
  type?: "info" | "tip" | "important"; title?: string; children: React.ReactNode;
}) {
  const styles = {
    info: { border: "border-blue-500/20", bg: "bg-blue-500/5", icon: Info, iconColor: "text-blue-500", titleColor: "text-blue-600 dark:text-blue-400" },
    tip: { border: "border-emerald-500/20", bg: "bg-emerald-500/5", icon: CheckCircle2, iconColor: "text-emerald-500", titleColor: "text-emerald-600 dark:text-emerald-400" },
    important: { border: "border-amber-500/20", bg: "bg-amber-500/5", icon: Zap, iconColor: "text-amber-500", titleColor: "text-amber-600 dark:text-amber-400" },
  };
  const s = styles[type];
  const IconEl = s.icon;
  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} p-4 flex gap-3`}>
      <IconEl className={`w-5 h-5 ${s.iconColor} shrink-0 mt-0.5`} />
      <div>
        {title ? <p className={`text-sm font-semibold ${s.titleColor} mb-1`}>{title}</p> : null}
        <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
      </div>
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
    { icon: Music, label: "Works" }, { icon: FileText, label: "Files" },
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

function SectionHeading({ children }: { children: React.ReactNode }) {
  return <h3 className="text-base font-semibold text-foreground mb-3 flex items-center gap-2">{children}</h3>;
}

// ---------------------------------------------------------------------------
// Content components — each at module scope (rerender-no-inline-components)
// ---------------------------------------------------------------------------

const GettingStartedContent = () => (
  <div className="space-y-0">
    <Step num={1} title="Create Your Account">
      Sign up using your Google account or email and password. Once signed in, you'll land on your <strong>Dashboard</strong> — your central hub for all tools and features.
    </Step>
    <Step num={2} title="Add Your First Artist">
      Navigate to <strong>Artist Profiles</strong> and click <strong>Add Artist</strong>. Fill in their name, bio, genres, and connect streaming profiles (Spotify, Apple Music, SoundCloud). Add social media links and custom URLs for press kits or EPKs.
    </Step>
    <Step num={3} title="Create a Project and Add Works">
      Create a <strong>Project</strong> in your Portfolio, then add <strong>Works</strong> to it. Upload contracts and audio files, and link them to specific works. Msanii stores your files securely and makes contracts searchable by Zoe AI.
    </Step>
    <Step num={4} title="Explore the Tools" isLast>
      Head to the <strong>Tools</strong> page to access OneClick (royalty calculations), Zoe (AI contract analysis), and the Split Sheet Generator. Each tool reads from your uploaded data — upload once, benefit everywhere.
    </Step>
    <Callout type="tip" title="Quick Start">
      The fastest path: Add Artist → Create Project → Upload a contract → Ask Zoe about it. You'll be productive in under 5 minutes.
    </Callout>
  </div>
);

const PortfolioContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed">
        The Portfolio is your home for all projects. Projects are displayed as a card grid grouped by <strong>Year</strong> then by <strong>Artist</strong>. Each artist section is collapsible with a unique color accent for quick visual identification.
      </p>
    </div>
    <div>
      <SectionHeading>Two Sections</SectionHeading>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Folder} title="My Projects" description="Projects you created and own. Full control over settings, members, and content." color="blue" />
        <FeatureCard icon={Users} title="Shared with Me" description="Projects where someone added you as a member. A role badge shows your access level." color="purple" />
      </div>
    </div>
    <div>
      <SectionHeading>Project Cards</SectionHeading>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <FeatureCard icon={FileText} title="Project Name & Artist" description="Identify the project and its primary artist at a glance." color="emerald" />
        <FeatureCard icon={Music} title="Work Count" description="Number of works (tracks/compositions) in the project." color="amber" />
        <FeatureCard icon={Users} title="Member Count" description="How many collaborators have access to this project." color="teal" />
        <FeatureCard icon={Zap} title="Last Updated" description="Timestamp showing when the project was last modified." color="indigo" />
      </div>
    </div>
    <Callout type="info" title="Navigation">
      Click any project card to open its <strong>Project Detail</strong> page with full access to works, files, members, and more.
    </Callout>
  </div>
);

const ProjectDetailContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        The Project Detail page is the central hub for everything in a project. It features six tabs giving you organized access to every aspect. You can inline-rename the project title and individual work titles by clicking on them.
      </p>
      <TabChips />
    </div>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <FeatureCard icon={Music} title="Works Tab" description="Create and manage works (tracks/compositions). Set type (single, EP track, album track, composition, or custom), ISRC, and link audio files." color="blue" />
      <FeatureCard icon={FileText} title="Files Tab" description="4 folder categories: Contracts, Split Sheets, Royalty Statements, Other. Upload files and link them to specific works with 'Relevant works' labels." color="emerald" />
      <FeatureCard icon={Volume2} title="Audio Tab" description="Upload and manage audio files. Link audio to works. Files are project-scoped — only this project's audio appears in work dropdowns." color="purple" />
      <FeatureCard icon={Users} title="Members Tab" description="Manage project-level access (Owner, Admin, Editor, Viewer) and view work-only collaborators. Invite new members by email." color="amber" />
      <FeatureCard icon={StickyNote} title="Notes Tab" description="Rich text notes scoped to the project using BlockNote editor. Great for meeting notes, strategy docs, or project context." color="teal" />
      <FeatureCard icon={Settings} title="Settings Tab" description="Edit project name and description. View primary artist. Leave project (non-owners) or delete project (owner only)." color="red" />
    </div>
    <div>
      <SectionHeading>Role Permissions</SectionHeading>
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
            {[
              ["See all works & files", true, true, true, true],
              ["Create/edit works", true, true, true, false],
              ["Upload files & audio", true, true, true, false],
              ["Manage members", true, true, false, false],
              ["Edit project settings", true, true, false, false],
              ["Delete project", true, false, false, false],
            ].map(([cap, ...roles], i) => (
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
    <Callout type="important" title="Inline Editing">
      Click any project or work title to rename it directly. Press <strong>Enter</strong> to save, <strong>Escape</strong> to cancel.
    </Callout>
  </div>
);

const RightsRegistryContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>Overview</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        The Rights Registry is your ownership tracking dashboard. It shows everything you own and everything you're involved in across all projects — without creating works here (creation happens in Project Detail).
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "My Works", desc: "Works you own", color: "purple" },
          { label: "Registered", desc: "Fully confirmed", color: "emerald" },
          { label: "Pending", desc: "Awaiting response", color: "amber" },
          { label: "Collaborations", desc: "Works you're on", color: "blue" },
        ].map((card) => (
          <div key={card.label} className={`rounded-xl border border-border p-3 bg-${card.color}-500/5`}>
            <p className={`text-xs font-medium text-${card.color}-600 dark:text-${card.color}-400 uppercase tracking-wide`}>{card.label}</p>
            <p className="text-xs text-muted-foreground mt-1">{card.desc}</p>
          </div>
        ))}
      </div>
    </div>
    <div>
      <SectionHeading>Four Tabs</SectionHeading>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Zap} title="Action Required" description="Pending invites with Accept/Decline buttons. Respond directly from the dashboard without navigating away." color="amber" />
        <FeatureCard icon={Folder} title="My Works" description="All works you own, organized by year then by project. Shows master/publishing percentages and collaborator count." color="purple" />
        <FeatureCard icon={Users} title="Collaborations" description="Works others invited you to. Work-scoped view — you only see your work, not the full project." color="blue" />
        <FeatureCard icon={Info} title="All Activity" description="Chronological feed of invites, acceptances, declines, and status changes across all your works." color="teal" />
      </div>
    </div>
    <div>
      <SectionHeading>Work Statuses</SectionHeading>
      <StatusBadges />
      <p className="text-sm text-muted-foreground mt-3 leading-relaxed">
        Works move from <strong>Draft</strong> → <strong>Pending</strong> (submitted for collaborator approval) → <strong>Registered</strong> (all collaborators confirmed). Only ownership changes (adding/revoking collaborators, changing stakes) revert a registered work back to draft. Metadata edits like renaming are safe.
      </p>
    </div>
    <div>
      <SectionHeading><Lock className="w-4 h-4" /> Enhanced Invite Form</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-3">
        When inviting a collaborator, the form captures everything needed:
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <FeatureCard icon={Users} title="Email & Name" description="Who you're inviting — they'll receive a rich email with all details." color="blue" />
        <FeatureCard icon={Shield} title="Role & Stakes" description="Select master %, publishing %, or both. Choose their role (producer, songwriter, etc.)." color="emerald" />
        <FeatureCard icon={FileText} title="Notes & Terms" description="Add context about the arrangement. The collaborator sees this in the invite email." color="amber" />
        <FeatureCard icon={Lock} title="Access Control" description="Files become accessible only after acceptance. The invite email contains all decision-making info." color="red" />
      </div>
    </div>
  </div>
);

const OneClickContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>How It Works</SectionHeading>
      <div className="space-y-0">
        <Step num={1} title="Select an Artist">Choose the artist whose royalties you want to calculate from your roster.</Step>
        <Step num={2} title="Upload or Select Contracts">Upload contract PDFs or select from existing project files and work-linked files. OneClick's AI parses them to extract parties, works, and royalty split percentages for each revenue type.</Step>
        <Step num={3} title="Upload Royalty Statements">Upload the royalty statement files that contain the actual revenue figures.</Step>
        <Step num={4} title="Calculate">OneClick applies the contract terms to the royalty statements and generates a detailed payment breakdown showing what each party is owed.</Step>
        <Step num={5} title="Export" isLast>Download the results as an Excel spreadsheet with itemized breakdowns and visual charts.</Step>
      </div>
    </div>
    <Callout type="tip" title="Best Results">
      Upload clear, text-based PDF contracts. Scanned images may have lower accuracy. Review extracted splits before calculating to ensure the AI interpreted the contract correctly.
    </Callout>
    <Callout type="info" title="Data Sources">
      OneClick can pull contracts from <strong>project files</strong>, <strong>work-linked files</strong>, and <strong>artist profile documents</strong> — all accessible from the document selection step.
    </Callout>
  </div>
);

const ZoeContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>How to Use Zoe</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">
        Once you've uploaded contracts, Zoe can search across all of them to answer your questions. Zoe uses semantic search to find relevant clauses and provides answers with citations back to the source text.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Bot} title="Ask a Question" description="Type specific questions like 'What is the royalty rate for streaming?' for the best results." color="indigo" />
        <FeatureCard icon={Zap} title="Quick Actions" description="Use suggested buttons for common queries: summarize, find key terms, identify expiration dates." color="amber" />
        <FeatureCard icon={FileText} title="Source Citations" description="Zoe references the specific contract sections so you can verify against original documents." color="emerald" />
        <FeatureCard icon={Shield} title="Shared Works" description="Access contracts from works you're a collaborator on via the 'From Shared Works' source option." color="blue" />
      </div>
    </div>
    <div>
      <SectionHeading>Example Questions</SectionHeading>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {[
          "What are the royalty split percentages?",
          "When does this agreement expire?",
          "What rights does the label have over masters?",
          "Summarize the key terms of this publishing deal.",
          "Are there any exclusivity clauses?",
          "What is the advance recoupment structure?",
        ].map((q) => (
          <div key={q} className="p-3 rounded-lg border border-border/50 bg-muted/20 text-sm text-muted-foreground italic hover:bg-muted/40 transition-colors">
            "{q}"
          </div>
        ))}
      </div>
    </div>
    <Callout type="info" title="Conversation History">
      Your chat history is saved automatically. Return to previous conversations and continue where you left off.
    </Callout>
  </div>
);

const SplitSheetContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>Step-by-Step Guide</SectionHeading>
      <div className="space-y-0">
        <Step num={1} title="Enter Song Details">Enter the song title, date, and any notes. This information appears at the top of the generated split sheet.</Step>
        <Step num={2} title="Define Splits">Add each contributor with their role (songwriter, producer, performer). Set publishing and master ownership percentages. Optionally add IPI numbers, publisher names, and label info.</Step>
        <Step num={3} title="Review & Download" isLast>Verify that publishing and master splits each total 100%. Download as a professionally formatted PDF ready for signing.</Step>
      </div>
    </div>
    <Callout type="tip" title="Pro Tip">
      Create split sheets <strong>before</strong> starting a project to avoid disputes later. Include IPI numbers when available — they're essential for collecting societies to route royalties properly.
    </Callout>
  </div>
);

const ArtistManagementContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>Artist Profiles</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed mb-4">Each artist profile is your private space for managing that artist's information, documents, and notes.</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={Users} title="Basic Info" description="Name, bio, genres, and profile image." color="blue" />
        <FeatureCard icon={Music} title="DSP Links" description="Connect Spotify, Apple Music, and SoundCloud profiles." color="emerald" />
        <FeatureCard icon={Zap} title="Social & Custom Links" description="Instagram, Twitter/X, TikTok, YouTube, EPKs, press kits, linktrees." color="purple" />
        <FeatureCard icon={StickyNote} title="Notes" description="Rich text notes on the profile for meeting notes, strategy docs, or context." color="amber" />
      </div>
    </div>
    <div>
      <SectionHeading>Projects & Documents</SectionHeading>
      <p className="text-sm text-muted-foreground leading-relaxed">
        Organize an artist's work into <strong>Projects</strong>. Each project stores contracts, royalty statements, works, and audio files. Uploaded contracts are automatically available to OneClick and Zoe AI — upload once, benefit everywhere.
      </p>
    </div>
    <Callout type="important" title="Privacy">
      Artist profiles are <strong>private to you</strong>. Only you can see an artist's notes and profile details. Works and their linked contracts can be shared with collaborators through the Rights Registry.
    </Callout>
  </div>
);

const WorkspaceContent = () => (
  <div className="space-y-6">
    <div>
      <SectionHeading>Features</SectionHeading>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <FeatureCard icon={LayoutGrid} title="Kanban Boards" description="Drag-and-drop task boards with columns (To Do, In Progress, Done). Create tasks with titles, descriptions, due dates, and color coding." color="sky" />
        <FeatureCard icon={Zap} title="Calendar View" description="See tasks on a timeline. Track deadlines, release dates, and contract expiration dates at a glance." color="amber" />
        <FeatureCard icon={Info} title="Integrations" description="Connect with Google Drive, Slack, Notion, and Monday.com for seamless workflow." color="blue" />
        <FeatureCard icon={Settings} title="Settings" description="Configure timezone and time format (12h/24h). Preferences apply across Dashboard and Workspace." color="emerald" />
      </div>
    </div>
    <Callout type="info" title="Task Organization">
      Create parent tasks for hierarchical tracking. Add comments for collaboration notes. Filter boards by artist to focus on specific projects.
    </Callout>
  </div>
);

const BestPracticesContent = () => (
  <div className="space-y-6">
    <Callout type="tip" title="Organize with Projects">
      Create a project for each deal, album, or major agreement. Name them descriptively (e.g., "2024 Publishing Deal — Universal" rather than "Deal 1"). Upload both contracts and royalty statements to the same project for seamless OneClick calculations.
    </Callout>
    <Callout type="important" title="Contract Management">
      Upload contracts as soon as they're signed. Use text-based PDFs (not scanned images) for best AI accuracy. After uploading, use Zoe to verify key terms were correctly extracted. Track expiration dates using Workspace boards.
    </Callout>
    <Callout type="info" title="Rights & Ownership">
      Use the Rights Registry to track ownership of every work before distributing or licensing. When inviting collaborators, include detailed stake information so everyone has a clear record. Keep files linked to works — collaborators see linked files once they accept.
    </Callout>
    <Callout type="tip" title="Workflow Recommendations">
      <strong>New artist:</strong> Create profile → Add projects → Upload contracts → Set up Workspace board.<br />
      <strong>Royalty period:</strong> Upload statements → Run OneClick → Export Excel → Record payments.<br />
      <strong>New collaboration:</strong> Generate split sheet → Share PDF → Upload signed copy to project.
    </Callout>
  </div>
);

// ---------------------------------------------------------------------------
// Map section id -> content component (module-level, stable reference)
// ---------------------------------------------------------------------------

const SECTION_CONTENT: Record<string, React.FC> = {
  "getting-started": GettingStartedContent,
  portfolio: PortfolioContent,
  "project-detail": ProjectDetailContent,
  "rights-registry": RightsRegistryContent,
  oneclick: OneClickContent,
  zoe: ZoeContent,
  "split-sheet": SplitSheetContent,
  "artist-management": ArtistManagementContent,
  workspace: WorkspaceContent,
  "best-practices": BestPracticesContent,
};

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

const Documentation = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [activeSection, setActiveSection] = useState("getting-started");

  // Derived values — useMemo for stable reference (rerender-derived-state)
  const currentIndex = useMemo(() => SECTION_INDEX.get(activeSection) ?? 0, [activeSection]);
  const prevSection = currentIndex > 0 ? SECTIONS[currentIndex - 1] : null;
  const nextSection = currentIndex < SECTIONS.length - 1 ? SECTIONS[currentIndex + 1] : null;
  const activeData = SECTIONS[currentIndex];
  const accent = ACCENT_STYLES[activeData.color] ?? ACCENT_STYLES.blue;
  const ActiveContent = SECTION_CONTENT[activeData.id];

  // Stable callbacks (rerender-functional-setstate)
  const goToPrev = useCallback(() => { if (prevSection) setActiveSection(prevSection.id); }, [prevSection]);
  const goToNext = useCallback(() => { if (nextSection) setActiveSection(nextSection.id); }, [nextSection]);

  return (
    <div className="min-h-screen bg-background flex flex-col">
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
                <div className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => navigate("/dashboard")}>
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
                <div
                  className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-sm font-medium text-primary cursor-pointer hover:bg-primary/30 transition-colors"
                  title="Profile Settings"
                  onClick={() => navigate("/profile")}
                >
                  {(user.email ?? "U")[0].toUpperCase()}
                </div>
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

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <nav className="hidden lg:flex flex-col w-64 shrink-0 border-r border-border bg-gradient-to-b from-card/50 to-background p-4 overflow-y-auto">
          <div className="flex items-center gap-2.5 mb-6 px-3 py-2 rounded-lg bg-primary/5">
            <BookOpen className="w-5 h-5 text-primary" />
            <span className="text-sm font-bold text-foreground tracking-tight">Documentation</span>
          </div>
          <div className="space-y-0.5">
            {SECTIONS.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors text-left ${
                    isActive
                      ? "bg-primary/10 text-primary font-medium border-l-2 border-primary pl-[10px]"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  }`}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {section.label}
                </button>
              );
            })}
          </div>
        </nav>

        {/* Mobile nav */}
        <div className="lg:hidden fixed top-[57px] left-0 right-0 z-40 bg-background border-b border-border overflow-x-auto">
          <div className="flex gap-1.5 p-3">
            {SECTIONS.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs whitespace-nowrap transition-colors ${
                    isActive ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {section.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Content */}
        <main className="flex-1 overflow-y-auto p-4 lg:p-8 lg:mt-0 mt-12">
          <div className="max-w-4xl mx-auto">
            <Card className="overflow-hidden border border-border">
              {/* Colored accent bar */}
              <div className={`h-1 bg-gradient-to-r ${accent.bar} to-transparent`} />
              {/* Module header */}
              <div className="p-6 pb-5 border-b border-border">
                <div className="flex items-center gap-3 mb-1">
                  <div className={`w-11 h-11 rounded-xl ${accent.iconBg} flex items-center justify-center ring-1 ring-border/50`}>
                    <activeData.icon className={`w-5 h-5 ${accent.iconText}`} />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-foreground">{activeData.label}</h2>
                    <p className="text-sm text-muted-foreground">{SECTION_DESCRIPTIONS[activeData.id]}</p>
                  </div>
                </div>
              </div>
              {/* Module content */}
              <div className="p-6 overflow-y-auto max-h-[calc(100vh-15rem)]">
                <ActiveContent />
                {/* Prev / Next */}
                <div className="flex items-center justify-between pt-6 border-t border-border mt-8">
                  {prevSection ? (
                    <Button variant="ghost" size="sm" onClick={goToPrev} className="gap-1.5">
                      <ArrowLeft className="w-4 h-4" /> {prevSection.label}
                    </Button>
                  ) : <div />}
                  <span className="text-xs text-muted-foreground">{currentIndex + 1} / {SECTIONS.length}</span>
                  {nextSection ? (
                    <Button variant="ghost" size="sm" onClick={goToNext} className="gap-1.5">
                      {nextSection.label} <ArrowRight className="w-4 h-4" />
                    </Button>
                  ) : <div />}
                </div>
              </div>
            </Card>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Documentation;
