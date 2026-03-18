import { useNavigate, useSearchParams } from "react-router-dom";
import { useEffect } from "react";
import { Music, ArrowLeft, LayoutGrid, HardDrive, Bell, CalendarDays } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { IntegrationHub } from "@/components/workspace/IntegrationHub";
import { KanbanBoard } from "@/components/workspace/boards/KanbanBoard";
import { CalendarView } from "@/components/workspace/boards/CalendarView";
import { toast } from "sonner";

const Workspace = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();

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
          <p className="text-muted-foreground">
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
        </Tabs>
      </main>
    </div>
  );
};

export default Workspace;
