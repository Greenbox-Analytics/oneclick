import { useNavigate, useSearchParams } from "react-router-dom";
import { useEffect, useState, useMemo } from "react";
import { Music, ArrowLeft, LayoutGrid, HardDrive, Bell, CalendarDays, Settings } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useWorkspaceSettings } from "@/hooks/useWorkspaceSettings";
import { IntegrationHub } from "@/components/workspace/IntegrationHub";
import { WorkspaceSettings } from "@/components/workspace/WorkspaceSettings";
import { KanbanBoard } from "@/components/workspace/boards/KanbanBoard";
import { CalendarView } from "@/components/workspace/boards/CalendarView";
import { toast } from "sonner";
import { RegistryNotifications } from "@/components/workspace/RegistryNotifications";
import { useUnreadCount } from "@/hooks/useRegistryNotifications";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

const Workspace = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const [now, setNow] = useState(new Date());
  const { settings } = useWorkspaceSettings();
  const unreadNotifications = useUnreadCount();

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
        slack: "Slack",
        notion: "Notion",
        monday: "Monday.com",
      };
      toast.success(`${providerNames[connected] || connected} connected successfully!`);
      setSearchParams({});
    }
  }, [searchParams, setSearchParams]);

  const defaultTab = searchParams.get("tab") || "integrations";
  const [activeTab, setActiveTab] = useState(defaultTab);
  const initialTaskId = searchParams.get("taskId") || undefined;

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
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate("/dashboard")}
            >
              <ArrowLeft className="w-5 h-5" />
            </Button>
            <div
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => navigate("/dashboard")}
            >
              <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
                <Music className="w-6 h-6 text-primary-foreground" />
              </div>
              <h1 className="text-2xl font-bold text-foreground">Msanii</h1>
            </div>
          </div>
          <ToolHelpButton onClick={walkthrough.replay} />
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h2 className="text-3xl font-bold text-foreground mb-2">Workspace</h2>
          <p className="text-sm text-muted-foreground">
            {formattedDateTime}
          </p>
          <p className="text-muted-foreground mt-1">
            Manage integrations, project boards, and notifications
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
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
            <TabsTrigger value="notifications" className="gap-2">
              <Bell className="w-4 h-4" />
              Notifications
              {unreadNotifications > 0 && (
                <span className="ml-1 px-1.5 py-0.5 text-[10px] font-bold bg-destructive text-destructive-foreground rounded-full">
                  {unreadNotifications}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="settings" className="gap-2">
              <Settings className="w-4 h-4" />
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="integrations" data-walkthrough="workspace-integrations">
            <IntegrationHub />
          </TabsContent>

          <TabsContent value="boards" data-walkthrough="workspace-boards">
            <KanbanBoard initialSelectedTaskId={initialTaskId} />
          </TabsContent>

          <TabsContent value="calendar" data-walkthrough="workspace-calendar">
            <CalendarView />
          </TabsContent>

          <TabsContent value="notifications">
            <RegistryNotifications />
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
