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

export const TOOL_CONFIGS: Record<string, ToolWalkthroughConfig> = {
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
        "Zoe is your AI-powered contract analyst. Select context, upload contracts, and ask questions. The chatbot answers questions based on the selected documents.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="zoe-sidebar"]',
        title: "Select Context",
        description:
          "Start by selecting an artist, project, and the documents you want Zoe to analyze.",
        placement: "right",
      },
      {
        targetSelector: '[data-walkthrough="zoe-upload"]',
        title: "Upload Contracts",
        description:
          "Upload PDF contracts here — Zoe will index them for analysis.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="zoe-chat"]',
        title: "Ask Anything",
        description:
          "Ask Zoe anything about your contracts. She'll answer with source citations you can verify.",
        placement: "top",
      },
      {
        targetSelector: '[data-walkthrough="zoe-newchat"]',
        title: "New Chat",
        description:
          "Start a fresh conversation anytime. Your previous chats are saved automatically.",
        placement: "bottom",
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
          "Choose Publishing (songwriting), Master (recording), or Both. Each type tracks splits independently.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="splitsheet-info"]',
        title: "Quick Tip",
        description:
          "Publishing royalties come from songwriting. Master royalties come from the recording. This box explains the difference.",
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
        "Your project management hub — manage tasks, create epics with subtasks, and efficiently organize your projects. Link tasks to artists and projects to keep everything connected.",
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
          "Connect Google Drive, Slack, Notion, and Monday.com to bring your tools together.",
        placement: "bottom",
      },
    ],
  },

  portfolio: {
    key: "portfolio",
    intro: {
      icon: FolderOpen,
      title: "Portfolio",
      description:
        "Your entire catalog organized by year and artist. Create projects, track works and members, and see projects shared with you by collaborators — all from one view.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="portfolio-add"]',
        title: "Create a Project",
        description:
          "Start a new project for an artist — albums, EPs, singles, or any body of work. Each project holds documents, audio files, works, and team members.",
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
          "Projects are grouped by year, then by artist. Each project card shows work count and member count. Click any project to open its detail page.",
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
      title: "Rights Registry",
      description:
        "Track and protect ownership of your works. Register master and publishing stakes, invite collaborators with defined roles and splits, and manage licensing and agreements — all in one place.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="registry-summary"]',
        title: "Your Catalog at a Glance",
        description:
          "See how many works you own, how many are fully registered, which are pending approval, and how many collaborations you're part of.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="registry-search"]',
        title: "Find Any Work",
        description:
          "Search by title, ISRC, or ISWC to quickly locate a specific work across all your projects.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="registry-tabs"]',
        title: "Organized by Priority",
        description:
          "Action Required shows pending invitations you need to accept or decline. My Works lists everything you own. Collaborations shows works where others invited you. Activity tracks all changes.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="registry-action"]',
        title: "Accept or Decline Invites",
        description:
          "When someone invites you as a collaborator on their work, it appears here. Review the ownership details, then accept to confirm your stake or decline to remove yourself.",
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
      description:
        "Everything about a project in one place — works (tracks/compositions), files, audio, team members, notes, and settings. Your role determines what you can do here.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="project-role"]',
        title: "Your Role",
        description:
          "Your role badge shows your access level. Owners and admins can manage members and settings. Editors can add works and files. Viewers have read-only access.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="project-tabs"]',
        title: "Project Tabs",
        description:
          "Works: tracks and compositions linked to this project. Files: contracts, split sheets, and documents. Audio: recordings. Members: who has access. Notes: collaborative notes. Settings: project configuration.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="project-add-work"]',
        title: "Add Works",
        description:
          "Create a new work (track or composition) inside this project. Works can then be registered in the Rights Registry with ownership stakes and collaborators.",
        placement: "bottom",
        skipIfMissing: true,
      },
      {
        targetSelector: '[data-walkthrough="project-members"]',
        title: "Team Members",
        description:
          "Invite people by email and assign them a role — Admin, Editor, or Viewer. Members see all works in this project. For work-level access only, invite collaborators from the Rights Registry instead.",
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
        "This is where you manage a single work — a track, composition, or recording. Define ownership splits, invite collaborators, set up licensing, and record agreements. Everything about this work lives here.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="work-header"]',
        title: "Work Identity",
        description:
          "ISRC (International Standard Recording Code) identifies a specific recording. ISWC (International Standard Musical Work Code) identifies the underlying composition. UPC (Universal Product Code) identifies the release/product. These codes are used by distributors and royalty collection societies worldwide.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="work-status"]',
        title: "Registration Status",
        description:
          "Draft: still being set up. Pending: submitted for collaborator approval. Registered: all collaborators have accepted their stakes and the work is fully confirmed.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="work-actions"]',
        title: "Owner Actions",
        description:
          "Invite collaborators to claim their ownership stake. Upload proof-of-ownership documents. Edit metadata like ISRC codes and release dates. Register the work once everything is in order.",
        placement: "bottom",
        skipIfMissing: true,
      },
      {
        targetSelector: '[data-walkthrough="work-tabs"]',
        title: "Ownership, Licensing & Agreements",
        description:
          "Ownership: master (recording) and publishing (songwriting) splits with percentages. Licensing: rights granted to third parties (sync, mechanical, performance). Agreements: formal records of ownership transfers, split agreements, and amendments.",
        placement: "bottom",
      },
    ],
  },
};
