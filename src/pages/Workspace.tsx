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

const Workspace = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const [now, setNow] = useState(new Date());
  const { settings } = useWorkspaceSettings();

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

        <Tabs defaultValue={defaultTab}>
          <TabsList className="mb-6">
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
            </TabsTrigger>
            <TabsTrigger value="settings" className="gap-2">
              <Settings className="w-4 h-4" />
              Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="integrations">
            <IntegrationHub />
          </TabsContent>

          <TabsContent value="boards">
            <KanbanBoard />
          </TabsContent>

          <TabsContent value="calendar">
            <CalendarView />
          </TabsContent>

          <TabsContent value="notifications">
            <div className="text-center py-12 text-muted-foreground">
              <Bell className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <h3 className="text-lg font-semibold mb-2">Notification Settings</h3>
              <p>Connect Slack, Notion, or Monday.com to configure notifications</p>
            </div>
          </TabsContent>

          <TabsContent value="settings">
            <WorkspaceSettings />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Workspace;
