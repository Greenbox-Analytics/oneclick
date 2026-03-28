# Per-Tool Walkthrough System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-time walkthroughs to each of the 6 tool pages, with a new `user_onboarding` table tracking completion per tool.

**Architecture:** Extends the existing Dashboard walkthrough system. Reuses `WalkthroughProvider` and `WalkthroughTooltip` for spotlight. Adds a `ToolIntroModal` (intro screen before spotlight), `ToolHelpButton` (replay), `useToolWalkthrough` (orchestration), `useToolOnboardingStatus` (DB), and a static config file. Each tool page wires these in with `data-walkthrough` attributes.

**Tech Stack:** React 18, TypeScript, Supabase (new `user_onboarding` table), shadcn/ui, Tailwind CSS, lucide-react

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/hooks/useToolOnboardingStatus.ts` | Queries `user_onboarding` row, returns per-tool booleans + `markToolCompleted` |
| `src/hooks/useToolWalkthrough.ts` | Orchestrates modal → spotlight → done, handles skipIfMissing + beforeStep |
| `src/config/toolWalkthroughConfig.ts` | Static config: intro content + spotlight steps for all 6 tools |
| `src/components/walkthrough/ToolIntroModal.tsx` | Full-screen intro overlay with icon/title/description + Start/Skip |
| `src/components/walkthrough/ToolHelpButton.tsx` | `?` button to replay walkthrough |

### Modified Files
| File | Change |
|------|--------|
| `src/integrations/supabase/types.ts` | Add `user_onboarding` table type |
| `src/pages/OneClickDocuments.tsx` | Add `data-walkthrough` attrs + walkthrough wiring |
| `src/pages/Zoe.tsx` | Add `data-walkthrough` attrs + walkthrough wiring |
| `src/pages/SplitSheet.tsx` | Add `data-walkthrough` attrs + walkthrough wiring |
| `src/pages/Artists.tsx` | Add `data-walkthrough` attrs + walkthrough wiring |
| `src/pages/Workspace.tsx` | Convert Tabs to controlled + walkthrough wiring |
| `src/pages/Portfolio.tsx` | Add `data-walkthrough` attrs + walkthrough wiring |

---

### Task 0: Database Schema & TypeScript Types

**Goal:** Create the `user_onboarding` table in Supabase and add the corresponding TypeScript type definitions.

**Files:**
- Modify: `src/integrations/supabase/types.ts:141` (insert after `profiles` section)
- Run SQL in Supabase dashboard

**Acceptance Criteria:**
- [ ] `user_onboarding` table exists with `user_id` PK and 6 boolean `_completed` columns defaulting to `false`
- [ ] FK to `auth.users(id)` with cascade delete
- [ ] RLS policy: users can only read/write their own row
- [ ] TypeScript types reflect the new table in Row, Insert, and Update

**Verify:** Open Supabase Table Editor > user_onboarding > confirm columns. Then `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Run SQL migration in Supabase**

Execute this SQL in the Supabase SQL Editor (Dashboard > SQL Editor > New query):

```sql
-- Create user_onboarding table
CREATE TABLE IF NOT EXISTS public.user_onboarding (
  user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  oneclick_completed boolean NOT NULL DEFAULT false,
  zoe_completed boolean NOT NULL DEFAULT false,
  splitsheet_completed boolean NOT NULL DEFAULT false,
  artists_completed boolean NOT NULL DEFAULT false,
  workspace_completed boolean NOT NULL DEFAULT false,
  portfolio_completed boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.user_onboarding ENABLE ROW LEVEL SECURITY;

-- RLS policy: users can read their own row
CREATE POLICY "Users can read own onboarding status"
  ON public.user_onboarding
  FOR SELECT
  USING (auth.uid() = user_id);

-- RLS policy: users can insert their own row
CREATE POLICY "Users can insert own onboarding status"
  ON public.user_onboarding
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- RLS policy: users can update their own row
CREATE POLICY "Users can update own onboarding status"
  ON public.user_onboarding
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);
```

- [ ] **Step 2: Update TypeScript types**

In `src/integrations/supabase/types.ts`, add the `user_onboarding` table type. Insert after the closing `}` of the `profiles` table definition (after line 141), before `project_files`:

```typescript
      user_onboarding: {
        Row: {
          user_id: string
          oneclick_completed: boolean
          zoe_completed: boolean
          splitsheet_completed: boolean
          artists_completed: boolean
          workspace_completed: boolean
          portfolio_completed: boolean
          created_at: string
        }
        Insert: {
          user_id: string
          oneclick_completed?: boolean
          zoe_completed?: boolean
          splitsheet_completed?: boolean
          artists_completed?: boolean
          workspace_completed?: boolean
          portfolio_completed?: boolean
          created_at?: string
        }
        Update: {
          user_id?: string
          oneclick_completed?: boolean
          zoe_completed?: boolean
          splitsheet_completed?: boolean
          artists_completed?: boolean
          workspace_completed?: boolean
          portfolio_completed?: boolean
          created_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "user_onboarding_user_id_fkey"
            columns: ["user_id"]
            isOneToOne: true
            referencedRelation: "users"
            referencedColumns: ["id"]
          },
        ]
      }
```

- [ ] **Step 3: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No type errors

- [ ] **Step 4: Commit**

```bash
git add src/integrations/supabase/types.ts
git commit -m "feat: add user_onboarding table schema and types"
```

---

### Task 1: useToolOnboardingStatus Hook

**Goal:** Create a hook that queries the `user_onboarding` table and returns per-tool completion status with a function to mark tools as completed.

**Files:**
- Create: `src/hooks/useToolOnboardingStatus.ts`

**Acceptance Criteria:**
- [ ] Returns `{ statuses: Record<string, boolean>, loading: boolean, markToolCompleted: (toolKey: string) => Promise<void> }`
- [ ] Queries `user_onboarding` table by `user.id`
- [ ] If no row exists, returns all `false`
- [ ] `markToolCompleted` upserts the row, setting `{toolKey}_completed = true`

**Verify:** `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Create the hook file**

Create `src/hooks/useToolOnboardingStatus.ts`:

```typescript
import { useEffect, useState, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { supabase } from "@/integrations/supabase/client";

const TOOL_KEYS = [
  "oneclick",
  "zoe",
  "splitsheet",
  "artists",
  "workspace",
  "portfolio",
] as const;

type ToolKey = (typeof TOOL_KEYS)[number];

interface ToolOnboardingStatus {
  statuses: Record<ToolKey, boolean>;
  loading: boolean;
  markToolCompleted: (toolKey: ToolKey) => Promise<void>;
}

const defaultStatuses: Record<ToolKey, boolean> = {
  oneclick: false,
  zoe: false,
  splitsheet: false,
  artists: false,
  workspace: false,
  portfolio: false,
};

export const useToolOnboardingStatus = (): ToolOnboardingStatus => {
  const { user } = useAuth();
  const [statuses, setStatuses] = useState<Record<ToolKey, boolean>>(defaultStatuses);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      if (!user) {
        setLoading(false);
        return;
      }

      try {
        const { data, error } = await supabase
          .from("user_onboarding")
          .select("*")
          .eq("user_id", user.id)
          .single();

        if (error && error.code !== "PGRST116") {
          // PGRST116 = no rows returned — that's fine, means first visit
          console.error("Error fetching tool onboarding status:", error);
        }

        if (data) {
          setStatuses({
            oneclick: data.oneclick_completed,
            zoe: data.zoe_completed,
            splitsheet: data.splitsheet_completed,
            artists: data.artists_completed,
            workspace: data.workspace_completed,
            portfolio: data.portfolio_completed,
          });
        }
        // If no data, statuses stay as defaultStatuses (all false)
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
  }, [user]);

  const markToolCompleted = useCallback(
    async (toolKey: ToolKey) => {
      if (!user) return;

      const columnName = `${toolKey}_completed` as const;

      const { error } = await supabase
        .from("user_onboarding")
        .upsert(
          {
            user_id: user.id,
            [columnName]: true,
          },
          { onConflict: "user_id" }
        );

      if (error) {
        console.error("Error marking tool onboarding complete:", error);
        return;
      }

      setStatuses((prev) => ({ ...prev, [toolKey]: true }));
    },
    [user]
  );

  return { statuses, loading, markToolCompleted };
};
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useToolOnboardingStatus.ts
git commit -m "feat: add useToolOnboardingStatus hook"
```

---

### Task 2: Tool Walkthrough Config & Extended Step Type

**Goal:** Create the static configuration file defining intro content and spotlight steps for all 6 tools, plus the extended step type with `skipIfMissing`.

**Files:**
- Create: `src/config/toolWalkthroughConfig.ts`

**Acceptance Criteria:**
- [ ] `ToolWalkthroughStep` type extends existing `WalkthroughStep` with `skipIfMissing?: boolean`
- [ ] `ToolWalkthroughConfig` type defined with `key`, `intro`, `steps`
- [ ] All 6 tool configs exported as `TOOL_CONFIGS` object
- [ ] Steps that target potentially-absent elements have `skipIfMissing: true`

**Verify:** `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Create the config file**

Create `src/config/toolWalkthroughConfig.ts`:

```typescript
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
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/config/toolWalkthroughConfig.ts
git commit -m "feat: add tool walkthrough config with step definitions"
```

---

### Task 3: ToolIntroModal & ToolHelpButton Components

**Goal:** Build the intro modal overlay that appears before the spotlight walkthrough and the help button for replaying the tour.

**Files:**
- Create: `src/components/walkthrough/ToolIntroModal.tsx`
- Create: `src/components/walkthrough/ToolHelpButton.tsx`

**Acceptance Criteria:**
- [ ] `ToolIntroModal` shows tool icon, title, description with "Start Tour" and "Skip" buttons
- [ ] `ToolIntroModal` renders a dark backdrop with centered content and fade-in animation
- [ ] `ToolHelpButton` renders a `HelpCircle` icon button
- [ ] `ToolHelpButton` shows "Replay tour" tooltip on hover

**Verify:** `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Create ToolIntroModal**

Create `src/components/walkthrough/ToolIntroModal.tsx`:

```typescript
import { Button } from "@/components/ui/button";
import type { ToolWalkthroughConfig } from "@/config/toolWalkthroughConfig";

interface ToolIntroModalProps {
  config: ToolWalkthroughConfig;
  isOpen: boolean;
  onStartTour: () => void;
  onSkip: () => void;
}

const ToolIntroModal = ({ config, isOpen, onStartTour, onSkip }: ToolIntroModalProps) => {
  if (!isOpen) return null;

  const Icon = config.intro.icon;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" />

      {/* Content */}
      <div className="relative z-10 max-w-md w-full mx-4 bg-card border border-border rounded-2xl shadow-2xl p-8 animate-in fade-in zoom-in-95 duration-300">
        <div className="flex flex-col items-center text-center space-y-6">
          <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
            <Icon className="w-8 h-8 text-primary" />
          </div>

          <div className="space-y-2">
            <h2 className="text-2xl font-bold text-foreground">
              {config.intro.title}
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              {config.intro.description}
            </p>
          </div>

          <div className="flex gap-3 w-full">
            <Button variant="ghost" onClick={onSkip} className="flex-1">
              Skip
            </Button>
            <Button onClick={onStartTour} className="flex-1">
              Start Tour
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ToolIntroModal;
```

- [ ] **Step 2: Create ToolHelpButton**

Create `src/components/walkthrough/ToolHelpButton.tsx`:

```typescript
import { HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface ToolHelpButtonProps {
  onClick: () => void;
}

const ToolHelpButton = ({ onClick }: ToolHelpButtonProps) => {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClick}
            className="h-9 w-9"
          >
            <HelpCircle className="w-5 h-5 text-muted-foreground" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Replay tour</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default ToolHelpButton;
```

- [ ] **Step 3: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add src/components/walkthrough/ToolIntroModal.tsx src/components/walkthrough/ToolHelpButton.tsx
git commit -m "feat: add ToolIntroModal and ToolHelpButton components"
```

---

### Task 4: useToolWalkthrough Hook

**Goal:** Create the orchestration hook that manages the modal → spotlight → done flow, including `skipIfMissing` and `onBeforeStep` support.

**Files:**
- Create: `src/hooks/useToolWalkthrough.ts`

**Acceptance Criteria:**
- [ ] Manages phase: `idle` → `modal` → `spotlight` → `done`
- [ ] Auto-starts in `modal` phase when `completed === false` (after 500ms delay)
- [ ] On each spotlight step transition, checks `skipIfMissing` — if target absent, advances automatically
- [ ] Calls `onBeforeStep(stepIndex)` callback before each step if provided
- [ ] `replay()` re-enters `modal` phase with `isReplay = true` (skip/complete don't write DB)
- [ ] `totalSteps` reflects visible steps (excluding skipped)
- [ ] `visibleStepIndex` shows the user-facing step number adjusted for skips

**Verify:** `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Create the hook file**

Create `src/hooks/useToolWalkthrough.ts`:

```typescript
import { useState, useCallback, useEffect, useRef } from "react";
import type { ToolWalkthroughStep, ToolWalkthroughConfig } from "@/config/toolWalkthroughConfig";

type Phase = "idle" | "modal" | "spotlight" | "done";

interface UseToolWalkthroughOptions {
  onBeforeStep?: (stepIndex: number) => void;
}

interface UseToolWalkthroughReturn {
  phase: Phase;
  currentStepIndex: number;
  currentStep: ToolWalkthroughStep | null;
  totalSteps: number;
  visibleStepIndex: number;
  next: () => void;
  skip: () => void;
  replay: () => void;
  startSpotlight: () => void;
}

export const useToolWalkthrough = (
  config: ToolWalkthroughConfig,
  completed: boolean,
  loading: boolean,
  markCompleted: () => Promise<void>,
  options?: UseToolWalkthroughOptions
): UseToolWalkthroughReturn => {
  const [phase, setPhase] = useState<Phase>("idle");
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [visibleStepCount, setVisibleStepCount] = useState(config.steps.length);
  const [skippedBefore, setSkippedBefore] = useState(0);
  const isReplayRef = useRef(false);
  const hasAutoStarted = useRef(false);

  // Auto-start when completed is false and loading is done
  useEffect(() => {
    if (loading || completed || hasAutoStarted.current) return;
    hasAutoStarted.current = true;
    const timer = setTimeout(() => setPhase("modal"), 500);
    return () => clearTimeout(timer);
  }, [loading, completed]);

  const findNextVisibleStep = useCallback(
    (fromIndex: number): number | null => {
      for (let i = fromIndex; i < config.steps.length; i++) {
        const step = config.steps[i];
        if (step.skipIfMissing) {
          const target = document.querySelector(step.targetSelector);
          if (!target) continue;
        }
        return i;
      }
      return null;
    },
    [config.steps]
  );

  const goToStep = useCallback(
    (index: number) => {
      const step = config.steps[index];
      if (options?.onBeforeStep) {
        options.onBeforeStep(index);
      }

      // Wait a tick for DOM to update after beforeStep (e.g., tab switch)
      setTimeout(() => {
        if (step.skipIfMissing) {
          const target = document.querySelector(step.targetSelector);
          if (!target) {
            // Skip this step, try next
            const nextVisible = findNextVisibleStep(index + 1);
            if (nextVisible !== null) {
              goToStep(nextVisible);
            } else {
              // No more visible steps — complete
              setPhase("done");
              if (!isReplayRef.current) markCompleted();
            }
            return;
          }
        }
        setCurrentStepIndex(index);
      }, 50);
    },
    [config.steps, options, findNextVisibleStep, markCompleted]
  );

  const startSpotlight = useCallback(() => {
    // Count visible steps for the counter
    let visible = 0;
    let skippedBeforeFirst = 0;
    let foundFirst = false;
    for (let i = 0; i < config.steps.length; i++) {
      const step = config.steps[i];
      if (step.skipIfMissing) {
        const target = document.querySelector(step.targetSelector);
        if (!target) {
          if (!foundFirst) skippedBeforeFirst++;
          continue;
        }
      }
      visible++;
      foundFirst = true;
    }
    setVisibleStepCount(visible);

    const firstVisible = findNextVisibleStep(0);
    if (firstVisible === null) {
      // No visible steps at all — skip straight to done
      setPhase("done");
      if (!isReplayRef.current) markCompleted();
      return;
    }

    setSkippedBefore(0);
    setPhase("spotlight");
    goToStep(firstVisible);
  }, [config.steps, findNextVisibleStep, goToStep, markCompleted]);

  const next = useCallback(() => {
    const nextVisible = findNextVisibleStep(currentStepIndex + 1);
    if (nextVisible !== null) {
      goToStep(nextVisible);
    } else {
      setPhase("done");
      if (!isReplayRef.current) markCompleted();
    }
  }, [currentStepIndex, findNextVisibleStep, goToStep, markCompleted]);

  const skip = useCallback(() => {
    setPhase("done");
    if (!isReplayRef.current) markCompleted();
  }, [markCompleted]);

  const replay = useCallback(() => {
    isReplayRef.current = true;
    setCurrentStepIndex(0);
    setPhase("modal");
  }, []);

  // Calculate visible step index (for display: "2 / 4" not "3 / 6")
  let visibleStepIndex = 0;
  for (let i = 0; i < currentStepIndex; i++) {
    const step = config.steps[i];
    if (step.skipIfMissing) {
      const target = document.querySelector(step.targetSelector);
      if (!target) continue;
    }
    visibleStepIndex++;
  }

  return {
    phase,
    currentStepIndex,
    currentStep: phase === "spotlight" ? config.steps[currentStepIndex] : null,
    totalSteps: visibleStepCount,
    visibleStepIndex,
    next,
    skip,
    replay,
    startSpotlight,
  };
};
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useToolWalkthrough.ts
git commit -m "feat: add useToolWalkthrough orchestration hook"
```

---

### Task 5: OneClickDocuments Walkthrough Integration

**Goal:** Add the per-tool walkthrough to the OneClickDocuments page.

**Files:**
- Modify: `src/pages/OneClickDocuments.tsx:815,986,1156`

**Acceptance Criteria:**
- [ ] `data-walkthrough="oneclick-contracts"` on the contracts Card (line 815)
- [ ] `data-walkthrough="oneclick-royalty"` on the royalty statement Card (line 986)
- [ ] `data-walkthrough="oneclick-calculate"` on the Calculate Royalties Button (line 1156)
- [ ] Walkthrough auto-triggers on first visit
- [ ] Help button in header area

**Verify:** `npx tsc --noEmit` passes. `npm run dev` — navigate to OneClick > select artist > land on documents page — walkthrough should auto-trigger.

**Steps:**

- [ ] **Step 1: Add imports**

At the top of `src/pages/OneClickDocuments.tsx`, add these imports with the existing ones:

```typescript
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Add hooks inside the component**

Inside the `OneClickDocuments` component, after the existing state declarations (after line ~118), add:

```typescript
  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.oneclick,
    statuses.oneclick,
    onboardingLoading,
    () => markToolCompleted("oneclick")
  );
```

- [ ] **Step 3: Add data-walkthrough attributes**

On the Contracts Card (line 815), change:
```tsx
          <Card>
```
To:
```tsx
          <Card data-walkthrough="oneclick-contracts">
```

On the Royalty Statement Card (line 986), change:
```tsx
          <Card>
```
To:
```tsx
          <Card data-walkthrough="oneclick-royalty">
```

On the Calculate Royalties Button (line 1156), change:
```tsx
          <Button
            onClick={() => handleCalculateRoyalties(false)}
```
To:
```tsx
          <Button
            data-walkthrough="oneclick-calculate"
            onClick={() => handleCalculateRoyalties(false)}
```

- [ ] **Step 4: Add ToolHelpButton to the header**

In the header section (around line 795), find the "Back to Artist Selection" button and add the help button next to it:

```tsx
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => navigate("/tools/oneclick")}
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Artist Selection
            </Button>
            <ToolHelpButton onClick={walkthrough.replay} />
          </div>
```

- [ ] **Step 5: Add walkthrough components to JSX**

Just before the closing `</main>` tag, add:

```tsx
        <ToolIntroModal
          config={TOOL_CONFIGS.oneclick}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
```

- [ ] **Step 6: Commit**

```bash
git add src/pages/OneClickDocuments.tsx
git commit -m "feat: add walkthrough to OneClickDocuments page"
```

---

### Task 6: Zoe Walkthrough Integration

**Goal:** Add the per-tool walkthrough to the Zoe page.

**Files:**
- Modify: `src/pages/Zoe.tsx:813,829,913,1087`

**Acceptance Criteria:**
- [ ] `data-walkthrough="zoe-sidebar"` on the sidebar `<aside>` (line 829)
- [ ] `data-walkthrough="zoe-upload"` on the upload button (line 913)
- [ ] `data-walkthrough="zoe-chat"` on the chat input component (line 1087)
- [ ] `data-walkthrough="zoe-newchat"` on the New Chat button (line 813)
- [ ] Walkthrough auto-triggers on first visit
- [ ] Help button in header

**Verify:** `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Add imports**

At the top of `src/pages/Zoe.tsx`, add:

```typescript
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Add hooks inside the component**

Inside the component, after the existing state declarations, add:

```typescript
  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.zoe,
    statuses.zoe,
    onboardingLoading,
    () => markToolCompleted("zoe")
  );
```

- [ ] **Step 3: Add data-walkthrough attributes**

On the sidebar `<aside>` (line 829), add `data-walkthrough="zoe-sidebar"`:
```tsx
            <aside data-walkthrough="zoe-sidebar" ref={sidebarRef}
```

On the Upload button in the sidebar (line 913), add `data-walkthrough="zoe-upload"`:
```tsx
                    <Button data-walkthrough="zoe-upload" variant="secondary"
```

On the `ZoeInputBar` component (line 1087), wrap it or add the attribute to its parent:
```tsx
              <div data-walkthrough="zoe-chat">
                <ZoeInputBar ... />
              </div>
```

On the New Chat button (line 813), add `data-walkthrough="zoe-newchat"`:
```tsx
                <Button data-walkthrough="zoe-newchat" variant="ghost" size="sm"
```

- [ ] **Step 4: Add ToolHelpButton to the header**

In the header area, next to the "Back to Tools" button, add:

```tsx
            <ToolHelpButton onClick={walkthrough.replay} />
```

- [ ] **Step 5: Add walkthrough components to JSX**

Just before the closing `</div>` of the component return, add:

```tsx
      <ToolIntroModal
        config={TOOL_CONFIGS.zoe}
        isOpen={walkthrough.phase === "modal"}
        onStartTour={walkthrough.startSpotlight}
        onSkip={walkthrough.skip}
      />
      <WalkthroughProvider
        isActive={walkthrough.phase === "spotlight"}
        currentStep={walkthrough.currentStep}
        currentStepIndex={walkthrough.visibleStepIndex}
        totalSteps={walkthrough.totalSteps}
        onNext={walkthrough.next}
        onSkip={walkthrough.skip}
      />
```

- [ ] **Step 6: Commit**

```bash
git add src/pages/Zoe.tsx
git commit -m "feat: add walkthrough to Zoe page"
```

---

### Task 7: SplitSheet Walkthrough Integration

**Goal:** Add the per-tool walkthrough to the SplitSheet page.

**Files:**
- Modify: `src/pages/SplitSheet.tsx:285,295,394`

**Acceptance Criteria:**
- [ ] `data-walkthrough="splitsheet-info"` on the info box div (line 285)
- [ ] `data-walkthrough="splitsheet-steps"` on the step indicator div (line 295)
- [ ] `data-walkthrough="splitsheet-royalty-type"` on the royalty type section (line 394)
- [ ] Walkthrough auto-triggers on first visit
- [ ] Help button in header

**Verify:** `npx tsc --noEmit` passes.

**Steps:**

- [ ] **Step 1: Add imports**

At the top of `src/pages/SplitSheet.tsx`, add:

```typescript
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Add hooks inside the component**

After existing state declarations, add:

```typescript
  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.splitsheet,
    statuses.splitsheet,
    onboardingLoading,
    () => markToolCompleted("splitsheet")
  );
```

- [ ] **Step 3: Add data-walkthrough attributes**

On the info box (line 285), change:
```tsx
        <div className="flex gap-2 items-start mb-6 px-4 py-3 rounded-lg bg-primary/5 border border-primary/20">
```
To:
```tsx
        <div data-walkthrough="splitsheet-info" className="flex gap-2 items-start mb-6 px-4 py-3 rounded-lg bg-primary/5 border border-primary/20">
```

On the step indicator (line 295), change:
```tsx
        <div className="flex items-center justify-center mb-8 gap-2">
```
To:
```tsx
        <div data-walkthrough="splitsheet-steps" className="flex items-center justify-center mb-8 gap-2">
```

On the royalty type section (line 394), change:
```tsx
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Royalty Type <Req />
                  </label>
```
To:
```tsx
                <div data-walkthrough="splitsheet-royalty-type">
                  <label className="text-sm font-medium mb-2 block">
                    Royalty Type <Req />
                  </label>
```

- [ ] **Step 4: Add ToolHelpButton to the header**

In the header area, next to the "Back to Tools" button, add:

```tsx
            <ToolHelpButton onClick={walkthrough.replay} />
```

- [ ] **Step 5: Add walkthrough components to JSX**

Before the closing `</main>` tag, add:

```tsx
        <ToolIntroModal
          config={TOOL_CONFIGS.splitsheet}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
```

- [ ] **Step 6: Commit**

```bash
git add src/pages/SplitSheet.tsx
git commit -m "feat: add walkthrough to SplitSheet page"
```

---

### Task 8: Artists Walkthrough Integration

**Goal:** Add the per-tool walkthrough to the Artists page with `skipIfMissing` on the artist card step.

**Files:**
- Modify: `src/pages/Artists.tsx:121,128,145`

**Acceptance Criteria:**
- [ ] `data-walkthrough="artists-add"` on the Add Artist button (line 121)
- [ ] `data-walkthrough="artists-search"` on the search input wrapper (line 128)
- [ ] `data-walkthrough="artists-card"` on the first artist card — if no artists, step is skipped
- [ ] Walkthrough auto-triggers on first visit
- [ ] Help button in header

**Verify:** `npx tsc --noEmit` passes. Test with empty roster — artist card step should be skipped.

**Steps:**

- [ ] **Step 1: Add imports**

At the top of `src/pages/Artists.tsx`, add:

```typescript
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Add hooks inside the component**

After existing state declarations, add:

```typescript
  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.artists,
    statuses.artists,
    onboardingLoading,
    () => markToolCompleted("artists")
  );
```

- [ ] **Step 3: Add data-walkthrough attributes**

On the Add Artist button (line 121), change:
```tsx
          <Button onClick={() => navigate("/artists/new")}>
```
To:
```tsx
          <Button data-walkthrough="artists-add" onClick={() => navigate("/artists/new")}>
```

On the search input wrapper div (line 128), change:
```tsx
          <div className="relative">
```
To:
```tsx
          <div data-walkthrough="artists-search" className="relative">
```

On the first artist Card in the grid (line 147), change:
```tsx
                <Card key={artist.id} className="hover:shadow-lg transition-shadow">
```
To (add attribute only on the first card):
```tsx
                <Card key={artist.id} data-walkthrough={index === 0 ? "artists-card" : undefined} className="hover:shadow-lg transition-shadow">
```

Note: This requires the `.map()` callback to include the `index` parameter. Change:
```tsx
              {filteredArtists.map((artist) => (
```
To:
```tsx
              {filteredArtists.map((artist, index) => (
```

- [ ] **Step 4: Add ToolHelpButton to the header**

In the header, next to the "Back to Dashboard" button, add:

```tsx
            <ToolHelpButton onClick={walkthrough.replay} />
```

- [ ] **Step 5: Add walkthrough components to JSX**

Before the closing `</main>` tag, add:

```tsx
        <ToolIntroModal
          config={TOOL_CONFIGS.artists}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
```

- [ ] **Step 6: Commit**

```bash
git add src/pages/Artists.tsx
git commit -m "feat: add walkthrough to Artists page"
```

---

### Task 9: Workspace Walkthrough Integration

**Goal:** Add the per-tool walkthrough to the Workspace page. Convert Tabs from uncontrolled to controlled so that the `onBeforeStep` callback can programmatically switch tabs before spotlighting content inside them.

**Files:**
- Modify: `src/pages/Workspace.tsx:104-126`

**Acceptance Criteria:**
- [ ] `<Tabs>` converted from `defaultValue` to controlled `value`/`onValueChange`
- [ ] `data-walkthrough="workspace-tabs"` on the `<TabsList>` (line 105)
- [ ] `data-walkthrough="workspace-boards"` on the boards `<TabsContent>` (line 132)
- [ ] `data-walkthrough="workspace-integrations"` on the integrations `<TabsContent>` (line 128)
- [ ] `onBeforeStep` switches to "boards" tab before step 2, "integrations" tab before step 3
- [ ] Help button in header
- [ ] Existing tab behavior (URL params, default tab) preserved

**Verify:** `npx tsc --noEmit` passes. `npm run dev` — Workspace walkthrough switches tabs automatically during spotlight.

**Steps:**

- [ ] **Step 1: Add imports**

At the top of `src/pages/Workspace.tsx`, add:

```typescript
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Convert Tabs to controlled and add hooks**

Inside the `Workspace` component, replace:
```typescript
  const defaultTab = searchParams.get("tab") || "integrations";
```
With:
```typescript
  const defaultTab = searchParams.get("tab") || "integrations";
  const [activeTab, setActiveTab] = useState(defaultTab);
```

Add the walkthrough hooks after existing state:

```typescript
  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.workspace,
    statuses.workspace,
    onboardingLoading,
    () => markToolCompleted("workspace"),
    {
      onBeforeStep: (stepIndex) => {
        // Step 1 (index 1) targets boards content — switch to boards tab
        if (stepIndex === 1) setActiveTab("boards");
        // Step 2 (index 2) targets integrations content — switch to integrations tab
        if (stepIndex === 2) setActiveTab("integrations");
      },
    }
  );
```

- [ ] **Step 3: Convert Tabs component to controlled**

Change the `<Tabs>` component (line 104) from:
```tsx
        <Tabs defaultValue={defaultTab}>
```
To:
```tsx
        <Tabs value={activeTab} onValueChange={setActiveTab}>
```

- [ ] **Step 4: Add data-walkthrough attributes**

On the `<TabsList>` (line 105), change:
```tsx
          <TabsList className="mb-6">
```
To:
```tsx
          <TabsList data-walkthrough="workspace-tabs" className="mb-6">
```

On the integrations `<TabsContent>` (line 128), change:
```tsx
          <TabsContent value="integrations">
```
To:
```tsx
          <TabsContent data-walkthrough="workspace-integrations" value="integrations">
```

On the boards `<TabsContent>` (line 132), change:
```tsx
          <TabsContent value="boards">
```
To:
```tsx
          <TabsContent data-walkthrough="workspace-boards" value="boards">
```

- [ ] **Step 5: Add ToolHelpButton to the header**

In the header (around line 89), add the help button:

```tsx
          <ToolHelpButton onClick={walkthrough.replay} />
```

- [ ] **Step 6: Add walkthrough components to JSX**

Before the closing `</main>` tag, add:

```tsx
        <ToolIntroModal
          config={TOOL_CONFIGS.workspace}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
```

- [ ] **Step 7: Commit**

```bash
git add src/pages/Workspace.tsx
git commit -m "feat: add walkthrough to Workspace with controlled tab switching"
```

---

### Task 10: Portfolio Walkthrough Integration

**Goal:** Add the per-tool walkthrough to the Portfolio page with `skipIfMissing` on data-dependent steps.

**Files:**
- Modify: `src/pages/Portfolio.tsx:550,557,692,872`

**Acceptance Criteria:**
- [ ] `data-walkthrough="portfolio-add"` on the Add Project button (line 550)
- [ ] `data-walkthrough="portfolio-filters"` on the sticky filter bar (line 557)
- [ ] `data-walkthrough="portfolio-year"` on the first year AccordionItem (line 692) — skipped if no projects
- [ ] `data-walkthrough="portfolio-folders"` on the first file grid (line 872) — skipped if no projects
- [ ] Walkthrough auto-triggers on first visit
- [ ] Help button in header

**Verify:** `npx tsc --noEmit` passes. Test with empty portfolio — year/folder steps skipped gracefully.

**Steps:**

- [ ] **Step 1: Add imports**

At the top of `src/pages/Portfolio.tsx`, add:

```typescript
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
```

- [ ] **Step 2: Add hooks inside the component**

After existing state declarations, add:

```typescript
  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(
    TOOL_CONFIGS.portfolio,
    statuses.portfolio,
    onboardingLoading,
    () => markToolCompleted("portfolio")
  );
```

- [ ] **Step 3: Add data-walkthrough attributes**

On the Add Project button (line 550), change:
```tsx
          <Button onClick={() => handleAddProject()} className="gap-2">
```
To:
```tsx
          <Button data-walkthrough="portfolio-add" onClick={() => handleAddProject()} className="gap-2">
```

On the sticky filter bar (line 557), change:
```tsx
        <div className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 pb-4 mb-6 border-b border-border">
```
To:
```tsx
        <div data-walkthrough="portfolio-filters" className="sticky top-0 z-10 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 pb-4 mb-6 border-b border-border">
```

On the first year AccordionItem (line 692) — add attribute only to the first one. Change:
```tsx
            <AccordionItem key={yearGroup.year} value={`year-${yearGroup.year}`} className="border rounded-lg px-4">
```
To:
```tsx
            <AccordionItem key={yearGroup.year} data-walkthrough={index === 0 ? "portfolio-year" : undefined} value={`year-${yearGroup.year}`} className="border rounded-lg px-4">
```

Note: This requires the `.map()` callback to include the `index` parameter. Change:
```tsx
          {years.map((yearGroup) => (
```
To:
```tsx
          {years.map((yearGroup, index) => (
```

On the first file grid (line 872) — add attribute only to the first project card's grid. The simplest approach is to add it to the grid container. Change:
```tsx
                                    <div className="grid grid-cols-2 gap-2">
```
To (add a tracking variable to mark only the first grid):
```tsx
                                    <div data-walkthrough={isFirstProjectCard ? "portfolio-folders" : undefined} className="grid grid-cols-2 gap-2">
```

To track the first project card, add a `let` variable before the year accordion rendering (before line 690):
```typescript
        let isFirstProjectCard = true;
```

And set it to `false` after the first grid is rendered (inside the project card map, after the grid div closes):
```typescript
        // After the grid div closes, set flag to false
        {(() => { isFirstProjectCard = false; return null; })()}
```

Alternatively, a cleaner approach: use a ref to track if the attribute has been set:

Add near the hooks:
```typescript
  const firstFolderGridRef = useRef(true);
```

Then in the grid:
```tsx
                                    <div
                                      data-walkthrough={firstFolderGridRef.current ? "portfolio-folders" : undefined}
                                      ref={(el) => { if (el && firstFolderGridRef.current) firstFolderGridRef.current = false; }}
                                      className="grid grid-cols-2 gap-2"
                                    >
```

- [ ] **Step 4: Add ToolHelpButton to the header**

In the header area (around line 540), next to the back button, add:

```tsx
          <ToolHelpButton onClick={walkthrough.replay} />
```

- [ ] **Step 5: Add walkthrough components to JSX**

Before the closing `</main>` tag, add:

```tsx
        <ToolIntroModal
          config={TOOL_CONFIGS.portfolio}
          isOpen={walkthrough.phase === "modal"}
          onStartTour={walkthrough.startSpotlight}
          onSkip={walkthrough.skip}
        />
        <WalkthroughProvider
          isActive={walkthrough.phase === "spotlight"}
          currentStep={walkthrough.currentStep}
          currentStepIndex={walkthrough.visibleStepIndex}
          totalSteps={walkthrough.totalSteps}
          onNext={walkthrough.next}
          onSkip={walkthrough.skip}
        />
```

- [ ] **Step 6: Commit**

```bash
git add src/pages/Portfolio.tsx
git commit -m "feat: add walkthrough to Portfolio page"
```

---

## Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 0 | Database schema & TypeScript types | — |
| 1 | useToolOnboardingStatus hook | 0 |
| 2 | Tool walkthrough config & step types | — |
| 3 | ToolIntroModal & ToolHelpButton | 2 |
| 4 | useToolWalkthrough hook | 1, 2, 3 |
| 5 | OneClickDocuments integration | 4 |
| 6 | Zoe integration | 4 |
| 7 | SplitSheet integration | 4 |
| 8 | Artists integration | 4 |
| 9 | Workspace integration | 4 |
| 10 | Portfolio integration | 4 |

Tasks 5-10 are independent of each other and can be done in parallel.
