import { useNavigate, useSearchParams } from "react-router-dom";
import { useEffect, useState, useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { LayoutGrid, HardDrive, CalendarDays, Settings, Loader2 } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { PageHeader } from "@/components/layout/PageHeader";
import { useIsMobile } from "@/hooks/use-mobile";
import { useAuth } from "@/contexts/AuthContext";
import { useWorkspaceSettings } from "@/hooks/useWorkspaceSettings";
import { IntegrationHub } from "@/components/workspace/IntegrationHub";
import { WorkspaceSettings } from "@/components/workspace/WorkspaceSettings";
import { KanbanBoard } from "@/components/workspace/boards/KanbanBoard";
import { BoardSwitcher } from "@/components/workspace/boards/BoardSwitcher";
import { CalendarView } from "@/components/workspace/boards/CalendarView";
import { toast } from "sonner";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";
import { useAnalytics } from "@/hooks/useAnalytics";
import { useWorkspaceScope } from "@/hooks/useWorkspaceScope";
import { useTabParam } from "@/hooks/useTabParam";
import { useBoardsList } from "@/hooks/useBoardsList";

const WORKSPACE_TABS = ["integrations", "boards", "calendar", "settings"] as const;

/** Matches KanbanBoard's own loading state so the two read as one screen. */
const BoardsLoading = () => (
  <div className="flex h-64 items-center justify-center">
    <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
  </div>
);

const Workspace = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const [now, setNow] = useState(new Date());
  const { settings } = useWorkspaceSettings();
  const queryClient = useQueryClient();
  const isMobile = useIsMobile();

  // Retired tabs — redirect legacy deep links rather than rendering a shell
  // with no tab selected. Notifications moved to its own page; the Teams tab
  // was merged into the /teams console (2026-08-16 boards-on-teams).
  useEffect(() => {
    const tab = searchParams.get("tab");
    if (tab === "notifications") {
      navigate("/notifications", { replace: true });
    } else if (tab === "teams") {
      navigate("/teams", { replace: true });
    }
  }, [searchParams, navigate]);

  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formattedDateTime = useMemo(() => {
    const tz = settings?.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    const use24h = settings?.use_24h_time ?? false;
    const dateStr = now.toLocaleDateString("en-US", {
      timeZone: tz,
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    const timeStr = now.toLocaleTimeString("en-US", {
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: !use24h,
    });
    const tzAbbr = now.toLocaleTimeString("en-US", {
      timeZone: tz,
      timeZoneName: "short",
    }).split(" ").pop() || "";
    return `${dateStr} · ${timeStr} ${tzAbbr}`;
  }, [now, settings?.timezone, settings?.use_24h_time]);

  // Handle OAuth callback success
  useEffect(() => {
    const connected = searchParams.get("connected");
    if (connected) {
      const providerNames: Record<string, string> = {
        google_drive: "Google Drive",
        dropbox: "Dropbox",
      };
      toast.success(`${providerNames[connected] || connected} connected successfully!`);
      queryClient.invalidateQueries({ queryKey: ["integrations"] });
      // Drop only `connected` (keep `tab`/`taskId`), and `replace` so Back
      // doesn't land on the callback URL and replay the toast — which pushed
      // another entry, making Back look broken.
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.delete("connected");
          return next;
        },
        { replace: true },
      );
    }
  }, [searchParams, setSearchParams]);

  const [activeTab, setActiveTab] = useTabParam(WORKSPACE_TABS, "integrations");
  const initialTaskId = searchParams.get("taskId") || undefined;

  // Board switcher selection (Personal vs. a team + a board within it).
  const [selectedBoardId, setSelectedBoardId] = useState<string | undefined>(undefined);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);

  // Same query key BoardSwitcher uses, so this is deduped, not a second fetch.
  const { isLoading: boardsLoading } = useBoardsList(selectedTeamId);

  // When workspace scoping is live, the header "Working as" pill IS the
  // context switcher: boards follow it, and the per-page context dropdown in
  // BoardSwitcher goes away. Reset the board selection on a switch so a board
  // from the previous workspace is never rendered under the new one.
  const { scopeId, enabled: scopingEnabled } = useWorkspaceScope();
  useEffect(() => {
    if (!scopingEnabled) return;
    if (selectedTeamId !== scopeId) {
      setSelectedBoardId(undefined);
      setSelectedTeamId(scopeId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopingEnabled, scopeId]);

  // Fire tool_opened when the active tab corresponds to a tool surface.
  // integrations / settings are NOT tools — skip them.
  const { captureToolOpened } = useAnalytics();
  useEffect(() => {
    if (activeTab === "boards") captureToolOpened("boards");
    else if (activeTab === "calendar") captureToolOpened("calendar");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab]);

  // Tool walkthrough
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.workspace, {
    onComplete: () => markToolCompleted("workspace"),
    onBeforeStep: (stepIndex) => {
      if (stepIndex === 1) setActiveTab("boards");
      if (stepIndex === 2) setActiveTab("calendar");
      if (stepIndex === 3) setActiveTab("settings");
      if (stepIndex === 4) setActiveTab("integrations");
    },
  });

  useEffect(() => {
    if (!onboardingLoading && !statuses.workspace && walkthrough.phase === "idle") {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.workspace]);

  return (
    <div className="min-h-screen bg-background">
      <PageHeader
        actions={
          <>
            <ToolHelpButton onClick={walkthrough.replay} />
          </>
        }
      />

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h2 className="text-3xl font-bold text-foreground mb-2">Workspace</h2>
          <p className="text-sm text-muted-foreground">
            {formattedDateTime}
          </p>
          <p className="text-muted-foreground mt-1">
            Manage integrations, project boards, and calendar
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          {isMobile ? (
            <Select value={activeTab} onValueChange={setActiveTab}>
              <SelectTrigger className="mb-6 w-full" data-walkthrough="workspace-tabs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="integrations">
                  <span className="inline-flex items-center gap-2">
                    <HardDrive className="w-4 h-4" /> Integrations
                  </span>
                </SelectItem>
                <SelectItem value="boards">
                  <span className="inline-flex items-center gap-2">
                    <LayoutGrid className="w-4 h-4" /> Project Boards
                  </span>
                </SelectItem>
                <SelectItem value="calendar">
                  <span className="inline-flex items-center gap-2">
                    <CalendarDays className="w-4 h-4" /> Calendar
                  </span>
                </SelectItem>
                <SelectItem value="settings">
                  <span className="inline-flex items-center gap-2">
                    <Settings className="w-4 h-4" /> Settings
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>
          ) : (
            <TabsList className="mb-6" data-walkthrough="workspace-tabs">
              <TabsTrigger value="integrations" className="gap-2">
                <HardDrive className="w-4 h-4" />
                Integrations
              </TabsTrigger>
              <TabsTrigger value="boards" className="gap-2">
                <LayoutGrid className="w-4 h-4" />
                Project Boards
              </TabsTrigger>
              <TabsTrigger value="calendar" className="gap-2">
                <CalendarDays className="w-4 h-4" />
                Calendar
              </TabsTrigger>
              <TabsTrigger value="settings" className="gap-2">
                <Settings className="w-4 h-4" />
                Settings
              </TabsTrigger>
            </TabsList>
          )}

          <TabsContent value="integrations" data-walkthrough="workspace-integrations">
            <IntegrationHub />
          </TabsContent>

          <TabsContent value="boards" data-walkthrough="workspace-boards">
            {/* One loading screen for the whole tab. The switcher, the board and
                the task overview each used to resolve on their own clock, so a
                slow load rendered as a stack of spinners popping in and out in
                three different places. Hold the switcher back until its board
                list is in; from there KanbanBoard's spinner (which now also
                waits on the task overview's data) carries the same shape in the
                same place, so it reads as one continuous load. */}
            {boardsLoading && <BoardsLoading />}
            {/* HIDDEN, never unmounted, while the board list loads. Gating this
                subtree on `boardsLoading` turned a parallel fetch into a
                waterfall: KanbanBoard's columns/tasks/parents queries only
                started once /boards/boards had come back, adding a whole round
                trip to the critical path. Kept mounted, all four requests leave
                together and the single spinner just covers the slowest.

                No wasted fetch in a team context: KanbanBoard renders null
                there until a board is picked, which is the one case where the
                board id genuinely depends on the list. */}
            <div className={boardsLoading ? "hidden" : undefined}>
              <BoardSwitcher
                teamId={selectedTeamId}
                boardId={selectedBoardId}
                onBoardChange={(b, t) => {
                  setSelectedBoardId(b);
                  setSelectedTeamId(t);
                }}
              />
              {/* Under a team context with no board selected, don't fall through to the
                  personal-boards union — the switcher shows its "No boards yet" state instead. */}
              {selectedTeamId && !selectedBoardId ? null : (
                <KanbanBoard
                  key={selectedBoardId ?? "personal"}
                  boardId={selectedBoardId}
                  teamId={selectedTeamId}
                  initialSelectedTaskId={initialTaskId}
                />
              )}
            </div>
          </TabsContent>

          <TabsContent value="calendar" data-walkthrough="workspace-calendar">
            <CalendarView />
          </TabsContent>

          <TabsContent value="settings" data-walkthrough="workspace-settings">
            <WorkspaceSettings />
          </TabsContent>
        </Tabs>

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
      </main>
    </div>
  );
};

export default Workspace;
