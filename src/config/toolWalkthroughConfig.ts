import {
  Calculator,
  MessageSquare,
  FileSignature,
  Users,
  LayoutGrid,
  FolderOpen,
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
          "Upload contracts here — PDF, DOCX, or plain text. Zoe will index them for analysis.",
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
          "Manage tasks with drag-and-drop Kanban boards. Create epics, add subtasks, and link tasks to artists and projects.",
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
        "Manage all your assets in one place. The page is organized per year, with each project created in that year shown per artist — giving you a clear, structured view of everything.",
    },
    steps: [
      {
        targetSelector: '[data-walkthrough="portfolio-filters"]',
        title: "Filter Bar",
        description:
          "Filter by artist, search projects, set date ranges, or change sort order.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="portfolio-add"]',
        title: "Add Project",
        description:
          "Create projects per artist. Each project holds documents, audio files, and linked tasks.",
        placement: "bottom",
      },
      {
        targetSelector: '[data-walkthrough="portfolio-year"]',
        title: "Year Groups",
        description:
          "Projects are organized by year, then by artist. Click to expand any section.",
        placement: "bottom",
        skipIfMissing: true,
      },
      {
        targetSelector: '[data-walkthrough="portfolio-folders"]',
        title: "Project Assets",
        description:
          "Each project has four document folders (Contracts, Split Sheets, Royalty Statements, Other), audio files/folders, and tasks created for the project.",
        placement: "bottom",
        skipIfMissing: true,
      },
    ],
  },
};
