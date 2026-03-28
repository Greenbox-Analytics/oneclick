# Per-Tool Walkthrough System — Design Spec

**Goal:** Add first-time walkthroughs to each tool page, triggered when a user accesses a tool for the first time. A new `user_onboarding` table tracks completion per tool without modifying existing tables.

**Approach:** Extend the existing Dashboard walkthrough system (Approach A). Reuse `WalkthroughProvider` and `WalkthroughTooltip` as spotlight primitives. Add a new intro modal, per-tool hook orchestration, a DB table, and a help button for replay.

---

## Database Schema

**New table: `user_onboarding`**

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `user_id` | uuid (PK, FK -> auth.users) | -- | One row per user |
| `oneclick_completed` | boolean | `false` | OneClick tool walkthrough |
| `zoe_completed` | boolean | `false` | Zoe AI walkthrough |
| `splitsheet_completed` | boolean | `false` | Split Sheet walkthrough |
| `artists_completed` | boolean | `false` | Artist Profiles walkthrough |
| `workspace_completed` | boolean | `false` | Workspace walkthrough |
| `portfolio_completed` | boolean | `false` | Portfolio walkthrough |
| `created_at` | timestamptz | `now()` | Row creation time |

**Constraints:**
- PK on `user_id`
- FK to `auth.users(id)` with cascade delete
- RLS: users can only read/write their own row

**Behavior:**
- Row is created via upsert on a user's first tool walkthrough completion/skip
- No row = no tools visited yet = all walkthroughs should auto-trigger
- Help button replay is purely client-side (no DB write)
- Scalable: add a column per new tool

**Existing `profiles` table is untouched.** The `onboarding_completed` and `walkthrough_completed` flags on profiles continue to handle the signup flow and Dashboard walkthrough respectively.

---

## Architecture

### New Files

| File | Responsibility |
|------|---------------|
| `src/hooks/useToolOnboardingStatus.ts` | Fetches the `user_onboarding` row, returns per-tool completion booleans. Creates the row on first upsert. |
| `src/hooks/useToolWalkthrough.ts` | Orchestrates modal -> spotlight flow for a given tool. Manages phase: `idle` -> `modal` -> `spotlight` -> `done`. |
| `src/components/walkthrough/ToolIntroModal.tsx` | Full-screen centered intro overlay with tool icon, title, description, "Start Tour" / "Skip" buttons. |
| `src/components/walkthrough/ToolHelpButton.tsx` | Small `?` (HelpCircle) icon button that re-triggers the walkthrough client-side without DB write. |
| `src/config/toolWalkthroughConfig.ts` | Static config defining each tool's intro content and spotlight steps. Single source of truth. |

### Reused Existing Components (No Changes)

- `src/components/walkthrough/WalkthroughProvider.tsx` — spotlight overlay with SVG mask
- `src/components/walkthrough/WalkthroughTooltip.tsx` — positioned tooltip with next/skip

### Modified Files

| File | Change |
|------|--------|
| `src/integrations/supabase/types.ts` | Add `user_onboarding` table type (Row, Insert, Update) |
| `src/pages/OneClickDocuments.tsx` | Add `data-walkthrough` attributes + walkthrough integration |
| `src/pages/Zoe.tsx` | Add `data-walkthrough` attributes + walkthrough integration |
| `src/pages/SplitSheet.tsx` | Add `data-walkthrough` attributes + walkthrough integration |
| `src/pages/Artists.tsx` | Add `data-walkthrough` attributes + walkthrough integration |
| `src/pages/Workspace.tsx` | Add `data-walkthrough` attributes + walkthrough integration |
| `src/pages/Portfolio.tsx` | Add `data-walkthrough` attributes + walkthrough integration |

### No Changes To

- `src/components/ProtectedRoute.tsx` — no gating on tool onboarding
- `src/hooks/useOnboardingStatus.ts` — stays as-is for signup flow
- `src/pages/Dashboard.tsx` — existing Dashboard walkthrough unchanged
- `src/App.tsx` — no new routes

---

## Data Flow

### First tool visit (auto-trigger)

```
Tool page mounts
  -> useToolOnboardingStatus() queries user_onboarding row
  -> tool's completed column is false (or no row exists)
  -> useToolWalkthrough auto-enters "modal" phase
  -> User clicks "Start Tour" -> enters "spotlight" phase
    -> Reuses WalkthroughProvider + WalkthroughTooltip
    -> On last step "Finish" or "Skip tour" -> marks completed in DB via upsert
  -> User clicks "Skip" on modal -> marks completed in DB via upsert
```

### Help button replay

```
User clicks ? button
  -> useToolWalkthrough enters "modal" phase (isReplay = true)
  -> Same modal -> spotlight flow
  -> Skip/complete does NOT write to DB (already marked completed)
```

---

## Component Design

### ToolIntroModal

- Full-screen centered overlay with dark backdrop
- Tool icon (from config), title, description
- Two buttons: "Start Tour" (primary) and "Skip" (ghost)
- Fade-in animation consistent with existing onboarding aesthetic

### ToolHelpButton

- `HelpCircle` icon from lucide-react
- Small circular button, positioned in the top-right area of each tool page header
- Tooltip on hover: "Replay tour"
- On click: triggers walkthrough in replay mode (no DB write)

### useToolOnboardingStatus Hook

- Queries `user_onboarding` for the current user (single row, all columns)
- Returns `{ statuses: Record<string, boolean>, loading: boolean, markToolCompleted: (toolKey: string) => Promise<void> }`
- If no row exists, returns all `false`
- `markToolCompleted` upserts the row, setting the `{toolKey}_completed` column to `true`

### useToolWalkthrough Hook

- Accepts: `toolKey: string`, `config: ToolWalkthroughConfig`, `completed: boolean`
- Manages phase state: `idle` | `modal` | `spotlight` | `done`
- Auto-starts in `modal` phase when `completed === false` (with small delay for DOM readiness)
- Exposes: `phase`, `currentStepIndex`, `currentStep`, `totalSteps`, `next()`, `skip()`, `replay()`
- `replay()` re-enters `modal` phase with `isReplay = true`
- When `isReplay` is true, skip/complete does not call `markToolCompleted`

### Config Shape

```typescript
interface ToolWalkthroughConfig {
  key: string;                    // matches DB column prefix: "oneclick", "zoe", etc.
  intro: {
    icon: LucideIcon;
    title: string;
    description: string;
  };
  steps: WalkthroughStep[];       // reuses existing WalkthroughStep type
}
```

---

## Per-Tool Walkthrough Definitions

### OneClick (`OneClickDocuments.tsx`) — 3 steps

**Intro:** "Calculate streaming royalty payments by selecting contracts and a royalty statement. OneClick will process them and break down exactly what each party is owed."

| # | Target Selector | Title | Description | Placement |
|---|----------------|-------|-------------|-----------|
| 1 | `[data-walkthrough="oneclick-contracts"]` | Contracts | Upload a new contract or select existing ones from your projects. You can select multiple contracts. | bottom |
| 2 | `[data-walkthrough="oneclick-royalty"]` | Royalty Statement | Upload or select the royalty statement to calculate against. This is the streaming revenue to split. | bottom |
| 3 | `[data-walkthrough="oneclick-calculate"]` | Calculate | Once you've selected your documents, hit Calculate. OneClick will process the contracts against the royalty statement and show the payment breakdown. | top |

### Zoe (`Zoe.tsx`) — 4 steps

**Intro:** "Zoe is your AI-powered contract analyst. Select context, upload contracts, and ask questions. The chatbot answers questions based on the selected documents."

| # | Target Selector | Title | Description | Placement |
|---|----------------|-------|-------------|-----------|
| 1 | `[data-walkthrough="zoe-sidebar"]` | Select Context | Start by selecting an artist, project, and the documents you want Zoe to analyze. | right |
| 2 | `[data-walkthrough="zoe-upload"]` | Upload Contracts | Upload contracts here -- PDF, DOCX, or plain text. Zoe will index them for analysis. | bottom |
| 3 | `[data-walkthrough="zoe-chat"]` | Ask Anything | Ask Zoe anything about your contracts. She'll answer with source citations you can verify. | top |
| 4 | `[data-walkthrough="zoe-newchat"]` | New Chat | Start a fresh conversation anytime. Your previous chats are saved automatically. | bottom |

### Split Sheet (`SplitSheet.tsx`) — 3 steps

**Intro:** "Generate professional split sheet agreements to document royalty ownership for music. Follow three steps: work details, ownership splits, then download."

| # | Target Selector | Title | Description | Placement |
|---|----------------|-------|-------------|-----------|
| 1 | `[data-walkthrough="splitsheet-steps"]` | Your Progress | You'll move through three steps: define the work, assign ownership splits, then generate your document. | bottom |
| 2 | `[data-walkthrough="splitsheet-royalty-type"]` | Royalty Type | Choose Publishing (songwriting), Master (recording), or Both. Each type tracks splits independently. | bottom |
| 3 | `[data-walkthrough="splitsheet-info"]` | Quick Tip | Publishing royalties come from songwriting. Master royalties come from the recording. This box explains the difference. | bottom |

### Artists (`Artists.tsx`) — 3 steps

**Intro:** "Your artist roster -- create profiles, store DSP links, contracts, and project files all in one place."

| # | Target Selector | Title | Description | Placement |
|---|----------------|-------|-------------|-----------|
| 1 | `[data-walkthrough="artists-add"]` | Add Artist | Add new artists to your roster here. Each artist gets their own profile for contracts, projects, and links. | bottom |
| 2 | `[data-walkthrough="artists-search"]` | Search | Quickly find artists by name as your roster grows. | bottom |
| 3 | `[data-walkthrough="artists-card"]` | Artist Card | Each card shows the artist's name, genres, and contract status. Click 'View Profile' to see everything. | bottom |

### Workspace (`Workspace.tsx`) — 3 steps

**Intro:** "Your project management hub -- manage tasks, create epics with subtasks, and efficiently organize your projects. Link tasks to artists and projects to keep everything connected."

| # | Target Selector | Title | Description | Placement |
|---|----------------|-------|-------------|-----------|
| 1 | `[data-walkthrough="workspace-tabs"]` | Navigation | Switch between Integrations, Project Boards, Calendar, Notifications, and Settings. | bottom |
| 2 | `[data-walkthrough="workspace-boards"]` | Project Boards | Manage tasks with drag-and-drop Kanban boards. Create epics, add subtasks, and link tasks to artists and projects. | bottom |
| 3 | `[data-walkthrough="workspace-integrations"]` | Integrations | Connect Google Drive, Slack, Notion, and Monday.com to bring your tools together. | bottom |

### Portfolio (`Portfolio.tsx`) — 4 steps

**Intro:** "Manage all your assets in one place. The page is organized per year, with each project created in that year shown per artist -- giving you a clear, structured view of everything."

| # | Target Selector | Title | Description | Placement |
|---|----------------|-------|-------------|-----------|
| 1 | `[data-walkthrough="portfolio-filters"]` | Filter Bar | Filter by artist, search projects, set date ranges, or change sort order. | bottom |
| 2 | `[data-walkthrough="portfolio-add"]` | Add Project | Create projects per artist. Each project holds documents, audio files, and linked tasks. | bottom |
| 3 | `[data-walkthrough="portfolio-year"]` | Year Groups | Projects are organized by year, then by artist. Click to expand any section. | bottom |
| 4 | `[data-walkthrough="portfolio-folders"]` | Project Assets | Each project has four document folders (Contracts, Split Sheets, Royalty Statements, Other), audio files/folders, and tasks created for the project. | bottom |

---

## Integration Pattern

Each tool page adds approximately this code:

```typescript
// Imports
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

// Inside component
const { statuses, markToolCompleted } = useToolOnboardingStatus();
const walkthrough = useToolWalkthrough("zoe", TOOL_CONFIGS.zoe, statuses.zoe);

// In JSX header area
<ToolHelpButton onClick={walkthrough.replay} />

// At bottom of JSX
<ToolIntroModal
  config={TOOL_CONFIGS.zoe}
  isOpen={walkthrough.phase === "modal"}
  onStartTour={walkthrough.startSpotlight}
  onSkip={walkthrough.skip}
/>
<WalkthroughProvider
  isActive={walkthrough.phase === "spotlight"}
  currentStep={walkthrough.currentStep}
  currentStepIndex={walkthrough.currentStepIndex}
  totalSteps={walkthrough.totalSteps}
  onNext={walkthrough.next}
  onSkip={walkthrough.skip}
/>
```

Plus `data-walkthrough` attributes on key DOM elements as defined per tool above.

---

## Summary

| Aspect | Detail |
|--------|--------|
| New DB table | `user_onboarding` -- 1 row per user, 1 boolean column per tool |
| New files | 5 (2 hooks, 2 components, 1 config) |
| Modified files | 8 (types + 7 tool pages) |
| Unchanged files | ProtectedRoute, useOnboardingStatus, Dashboard, App |
| Total spotlight steps | 20 across 6 tools |
| Intro modals | 6 (one per tool) |
| Replay mechanism | HelpCircle button per page, client-side only |
