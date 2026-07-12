import {
  Calculator,
  MessageSquare,
  FileSignature,
  Users,
  LayoutGrid,
  FolderOpen,
  Shield,
  Layers,
  UserCog,
  FileText,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

// Flip to false to restore the Metadata Registry walkthrough and works mentions.
const HIDE_REGISTRY_AND_WORKS = false;

export interface ToolWalkthroughStep {
  targetSelector: string;
  title: string;
  description: string;
  placement: "top" | "bottom" | "left" | "right";
  skipIfMissing?: boolean;
}

export interface ToolWalkthroughConfig {
  key: string;
  intro: {
    icon: LucideIcon;
    title: string;
    description: string;
  };
  steps: ToolWalkthroughStep[];
}

const ALL_TOOL_CONFIGS: Record<string, ToolWalkthroughConfig> = {
  oneclick: {
    key: "oneclick",
    intro: {
      icon: Calculator,
      title: "OneClick Royalty Calculator",
      description:
        "Calculate streaming royalty payments by selecting contracts and a royalty statement. OneClick will process them and break down exactly what each party is owed.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="oneclick-contracts"]',
        title: "Contracts",
        description:
          "Upload a new contract or select existing ones from your projects. You can select multiple contracts.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="oneclick-royalty"]',
        title: "Royalty Statement",
        description:
          "Upload or select the royalty statement to calculate against. This is the streaming revenue to split.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="oneclick-calculate"]',
        title: "Calculate",
        description:
          "Once you've selected your documents, hit Calculate. OneClick will process the contracts against the royalty statement and show the payment breakdown.",
        placement: "top",
      },
    ],
  },

  zoe: {
    key: "zoe",
    intro: {
      icon: MessageSquare,
      title: "Zoe AI",
      description:
        "Zoe is your AI contract analyst. Ask any music-business question — or add your contracts and ask about your specific deal. Zoe cites her sources, so you can open the contract at the exact page.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="zoe-context"]',
        title: "Add Your Contracts",
        description:
          "Click here to pick an artist, project, and the contracts you want Zoe to analyze — or upload a new PDF contract. This is also where you'll see what context is currently selected.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="zoe-chat"]',
        title: "Ask Anything",
        description:
          "Type your question here. With no contract added, Zoe answers general music-business questions; add a contract and she answers about your deal — and cites sources you can click to open the contract at the exact page. The attach button (next to the box) is a quick way to add contracts without leaving the chat.",
        placement: "top",
      },
      {
        targetSelector: '[data-walkthrough="zoe-newchat"]',
        title: "New Chat",
        description:
          "Start a fresh conversation anytime. New Chat clears the current session — Zoe keeps context only while you're chatting and doesn't store a long-term archive, so save anything important first.",
        placement: "bottom",
        skipIfMissing: true,
      },
    ],
  },

  splitsheet: {
    key: "splitsheet",
    intro: {
      icon: FileSignature,
      title: "Split Sheet Generator",
      description:
        "Generate professional split sheet agreements to document royalty ownership for music. Follow three steps: work details, ownership splits, then download.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="splitsheet-steps"]',
        title: "Your Progress",
        description:
          "You'll move through three steps: define the work, assign ownership splits, then generate your document.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="splitsheet-royalty-type"]',
        title: "Royalty Type",
        description:
          "Choose Publishing (composition), Master (recording), or Both. Each type tracks splits independently.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="splitsheet-info"]',
        title: "Quick Tip",
        description:
          "Publishing royalties come from the composition. Master royalties come from the recording. This box explains the difference.",
        placement: "bottom",
      },
    ],
  },

  artists: {
    key: "artists",
    intro: {
      icon: Users,
      title: "Artist Profiles",
      description:
        "Your artist roster — create profiles, store DSP links, contracts, and project files all in one place.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="artists-add"]',
        title: "Add Artist",
        description:
          "Add new artists to your roster here. Each artist gets their own profile for contracts, projects, and links.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="artists-search"]',
        title: "Search",
        description: "Quickly find artists by name as your roster grows.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="artists-card"]',
        title: "Artist Card",
        description:
          "Each card shows the artist's name, genres, and contract status. Click 'View Profile' to see everything.",
        placement: "bottom",
        skipIfMissing: true,
      },
    ],
  },

  workspace: {
    key: "workspace",
    intro: {
      icon: LayoutGrid,
      title: "Workspace",
      description:
        "Your project management hub — manage tasks, create campaigns with subtasks, and efficiently organize your projects. Link tasks to artists and projects to keep everything connected.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="workspace-tabs"]',
        title: "Navigation",
        description:
          "Switch between Integrations, Project Boards, Calendar, Notifications, and Settings.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="workspace-boards"]',
        title: "Project Boards",
        description:
          "Manage tasks on Kanban boards that iterate by month. Incomplete tasks carry forward to the next period — completed ones don't. Filter back to find previously completed tasks. You can change the iteration window in Settings.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="workspace-calendar"]',
        title: "Calendar",
        description:
          "View all your tasks and their due dates on a calendar. Flip back to see previous tasks and plan ahead.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="workspace-settings"]',
        title: "Settings",
        description:
          "Configure your workspace — change the board iteration window, set your timezone, and customize how your workspace behaves.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="workspace-integrations"]',
        title: "Integrations",
        description:
          "Connect Google Drive to import contracts, royalty statements, and split sheets directly into your projects — each category only accepts the right file types (e.g., PDFs for contracts, Excel/CSV for royalty statements). Slack notifications are coming soon.",
        placement: "bottom",
      },
    ],
  },

  portfolio: {
    key: "portfolio",
    intro: {
      icon: FolderOpen,
      title: "Portfolio",
      description: HIDE_REGISTRY_AND_WORKS
        ? "Your entire catalog organized by year and artist. Create projects, track files, audio, and members, and see projects shared with you by collaborators — all from one view."
        : "Your entire catalog organized by year and artist. Create projects, track works and members, and see projects shared with you by collaborators — all from one view.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="portfolio-add"]',
        title: "Create a Project",
        description: HIDE_REGISTRY_AND_WORKS
          ? "Start a new project for an artist — albums, EPs, singles, or any body of work. Each project holds documents, audio files, and team members."
          : "Start a new project for an artist — albums, EPs, singles, or any body of work. Each project holds documents, audio files, works, and team members.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="portfolio-filters"]',
        title: "Filter & Search",
        description:
          "Filter by artist, search projects by name, and sort alphabetically or by date. Use these to quickly find what you need as your catalog grows.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="portfolio-year"]',
        title: "Year & Artist Groups",
        description:
          "Projects are grouped by year, then by artist. Each project card shows its file, audio, and member counts. Click any project to open its detail page.",
        placement: "bottom",
        skipIfMissing: true,
      },
      {
        targetSelector: '[data-walkthrough="portfolio-shared"]',
        title: "Shared with Me",
        description:
          "Projects where you've been invited as a member appear here. Your role badge (Admin, Editor, or Viewer) shows your access level in each project.",
        placement: "top",
        skipIfMissing: true,
      },
    ],
  },

  registry: {
    key: "registry",
    intro: {
      icon: Shield,
      title: "Metadata Registry",
      description:
        "Track and protect ownership of your works. Register master and publishing stakes, invite collaborators with defined roles and splits, and keep full traceability of contracts and ownership — all in one place.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="registry-summary"]',
        title: "Your Catalog at a Glance",
        description:
          "Five quick stats — total works you're involved in, released and unreleased tracks, works that need attention, and works shared with you. Click any card to filter the list below.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="registry-tabs"]',
        title: "My Works & Shared with Me",
        description:
          "Toggle between My Works (everything you own) and Shared with Me (works others invited you to). Change the sort order or add a new work from this toolbar.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="registry-search"]',
        title: "Find Any Work",
        description:
          "Search by title, ISRC, artist, or project to quickly locate a specific work across all your projects.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="registry-action"]',
        title: "Your Works",
        description:
          "Works appear here grouped by artist and project — click any one to open it. Switch to Shared with Me to review collaboration invites and accept or decline your stake.",
        placement: "bottom",
        skipIfMissing: true,
      },
    ],
  },

  project_detail: {
    key: "project_detail",
    intro: {
      icon: Layers,
      title: "Project Detail",
      description: HIDE_REGISTRY_AND_WORKS
        ? "Everything about a project in one place — files, audio, team members, notes, and settings. Your role determines what you can do here."
        : "Everything about a project in one place — works (tracks/compositions), files, audio, team members, notes, and settings. Your role determines what you can do here.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="project-role"]',
        title: "Your Role",
        description: HIDE_REGISTRY_AND_WORKS
          ? "Your role badge shows your access level. Owners and admins can manage members and settings. Editors can add files and audio. Viewers have read-only access."
          : "Your role badge shows your access level. Owners and admins can manage members and settings. Editors can add works and files. Viewers have read-only access.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="project-tabs"]',
        title: "Project Tabs",
        description: HIDE_REGISTRY_AND_WORKS
          ? "Files: contracts, split sheets, and documents — upload directly or import from Google Drive (each category accepts specific file types like PDFs for contracts, Excel/CSV for royalty statements). Audio: recordings. Members: who has access. Notes: collaborative notes. Settings: project configuration and Slack channel linking."
          : "Works: tracks and compositions linked to this project. Files: contracts, split sheets, and documents — upload directly or import from Google Drive (each category accepts specific file types like PDFs for contracts, Excel/CSV for royalty statements). Audio: recordings. Members: who has access. Notes: collaborative notes. Settings: project configuration and Slack channel linking.",
        placement: "bottom",
      },
      ...(HIDE_REGISTRY_AND_WORKS
        ? []
        : [
            {
              targetSelector: '[data-walkthrough="project-add-work"]',
              title: "Add Works",
              description:
                "Create a new work (track or composition) inside this project. Works can then be registered in the Metadata Registry with ownership stakes and collaborators.",
              placement: "bottom" as const,
              skipIfMissing: true,
            },
          ]),
      {
        targetSelector: '[data-walkthrough="project-members"]',
        title: "Team Members",
        description: HIDE_REGISTRY_AND_WORKS
          ? "Invite people by email and assign them a role — Admin, Editor, or Viewer. Members see everything in this project."
          : "Invite people by email and assign them a role — Admin, Editor, or Viewer. Members see all works in this project. For work-level access only, invite collaborators from the Metadata Registry instead.",
        placement: "bottom",
      },
    ],
  },

  profile: {
    key: "profile",
    intro: {
      icon: UserCog,
      title: "Profile & TeamCard",
      description:
        "Set up your account details and configure your TeamCard — the identity collaborators see when you're invited to a work or project. Control exactly what information is visible.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="profile-info"]',
        title: "Account Information",
        description:
          "Your name and contact details. The 'Preferred Name' is how Msanii addresses you throughout the app — use a nickname or stage name if you like.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="profile-teamcard"]',
        title: "Your TeamCard",
        description:
          "Your TeamCard is your collaboration identity. When someone invites you to a project or work, this is what they see. Click 'Configure' to edit your details and control which fields are visible.",
        placement: "top",
      },
      {
        targetSelector: '[data-walkthrough="profile-theme"]',
        title: "Appearance",
        description:
          "Toggle between dark and light mode to match your preference.",
        placement: "top",
      },
    ],
  },

  work_detail: {
    key: "work_detail",
    intro: {
      icon: FileText,
      title: "Work Detail",
      description:
        "This is where you manage a single work — a track, composition, or recording. Set its identity codes and release status, record ownership splits, link documents, and export proof of ownership. Everything about this work lives on one page.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="work-header"]',
        title: "Work Identity & Status",
        description:
          "The work's title sits alongside its release tag and registration status — Draft (being set up), Pending (submitted for collaborator approval), or Registered (all collaborators confirmed).",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="work-details"]',
        title: "Track Details",
        description:
          "Industry codes identify the work everywhere royalties are tracked: ISRC (the recording), ISWC (the composition), and UPC (the release). Add the release date and a Spotify link, then use 'Pull from Spotify' to auto-fill metadata once the track is live.",
        placement: "top",
      },
      {
        targetSelector: '[data-walkthrough="work-access"]',
        title: "Your Access",
        description:
          "Your role on this work — Owner, Admin, Editor, or Viewer — decides what you can change. If you're not the owner, this card shows who is and how to reach them for edit access.",
        placement: "left",
      },
      {
        targetSelector: '[data-walkthrough="work-splits"]',
        title: "Ownership Splits",
        description:
          "Master (recording) and publishing (songwriting) percentages for each party. Owners and editors can edit them and see whether each column totals 100%; a work-only collaborator sees just their own share.",
        placement: "left",
      },
      {
        targetSelector: '[data-walkthrough="work-trace"]',
        title: "Traceability & Proof",
        description:
          "A quick audit of what's on file — linked documents, ISRC, and recorded stakes — with a one-click Export Proof of Ownership PDF.",
        placement: "left",
      },
    ],
  },
};

const HIDDEN_TOOL_KEYS = new Set(HIDE_REGISTRY_AND_WORKS ? ["registry", "work_detail"] : []);

export const TOOL_CONFIGS: Record<string, ToolWalkthroughConfig> = Object.fromEntries(
  Object.entries(ALL_TOOL_CONFIGS).filter(([key]) => !HIDDEN_TOOL_KEYS.has(key))
);
