import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { useMyRole } from "@/hooks/useProjectMembers";
import { InlineEdit } from "@/components/InlineEdit";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Music, ArrowLeft, Loader2,
  FileText, Volume2, Users, Settings, StickyNote, BookOpen, MessageSquare,
} from "lucide-react";
import { toast } from "sonner";

import WorksTab from "@/components/project/WorksTab";
import FilesTab from "@/components/project/FilesTab";
import AudioTab from "@/components/project/AudioTab";
import MembersTab from "@/components/project/MembersTab";
import SettingsTab from "@/components/project/SettingsTab";
import NotesView from "@/components/notes/NotesView";
import { useToolOnboardingStatus } from "@/hooks/useToolOnboardingStatus";
import { useProjectSlackChannel } from "@/hooks/useProjectIntegrations";
import { useSlackChannels } from "@/hooks/useSlackSettings";
import { useIntegrations } from "@/hooks/useIntegrations";
import { useToolWalkthrough } from "@/hooks/useToolWalkthrough";
import { TOOL_CONFIGS } from "@/config/toolWalkthroughConfig";
import ToolIntroModal from "@/components/walkthrough/ToolIntroModal";
import ToolHelpButton from "@/components/walkthrough/ToolHelpButton";
import WalkthroughProvider from "@/components/walkthrough/WalkthroughProvider";

const ROLE_COLORS: Record<string, string> = {
  owner: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  admin: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  editor: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  viewer: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const canEdit = (role: string | null) => role === "owner" || role === "admin" || role === "editor";

const ProjectDetail = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const userRole = useMyRole(projectId);

  const [activeTab, setActiveTab] = useState("files");

  const { connections } = useIntegrations();
  const slackConnected = connections.some(c => c.provider === "slack" && c.status === "active");
  const { channelId } = useProjectSlackChannel(projectId);
  const { data: channels } = useSlackChannels(slackConnected && !!channelId);
  const linkedChannel = channels?.find(c => c.id === channelId);

  // Tour
  const { statuses, loading: onboardingLoading, markToolCompleted } = useToolOnboardingStatus();
  const walkthrough = useToolWalkthrough(TOOL_CONFIGS.project_detail, {
    onComplete: () => markToolCompleted("project_detail"),
    onBeforeStep: (stepIndex) => {
      // Step 3 targets project-members tab trigger — switch to members so user sees the content
      if (stepIndex === 3) setActiveTab("members");
    },
  });

  // Fetch project with artist info
  const projectQuery = useQuery({
    queryKey: ["project-detail", projectId],
    queryFn: async () => {
      if (!projectId) return null;
      const { data } = await supabase
        .from("projects")
        .select("*, artists(name, user_id)")
        .eq("id", projectId)
        .single();
      return data;
    },
    enabled: !!projectId,
  });

  const project = projectQuery.data;

  useEffect(() => {
    if (!onboardingLoading && !statuses.project_detail && walkthrough.phase === "idle" && project) {
      const timer = setTimeout(() => walkthrough.startModal(), 500);
      return () => clearTimeout(timer);
    }
  }, [onboardingLoading, statuses.project_detail, project]);

  const handleRenameProject = async (newName: string) => {
    if (!projectId) return;
    const { error } = await supabase
      .from("projects")
      .update({ name: newName })
      .eq("id", projectId);
    if (error) {
      toast.error("Failed to rename project");
      throw error;
    }
    queryClient.invalidateQueries({ queryKey: ["project-detail", projectId] });
    toast.success("Project renamed");
  };

  if (projectQuery.isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4">
        <p className="text-muted-foreground">Project not found or access denied.</p>
        <Button variant="outline" onClick={() => navigate("/portfolio")}>
          <ArrowLeft className="w-4 h-4 mr-2" /> Back to Portfolio
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <Button
                variant="ghost"
                size="sm"
                className="shrink-0 text-muted-foreground hover:text-foreground"
                onClick={() => navigate("/portfolio")}
              >
                <ArrowLeft className="w-4 h-4 mr-1" /> Portfolio
              </Button>
              <div
                className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() => navigate("/dashboard")}
              >
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                  <Music className="w-5 h-5 text-primary-foreground" />
                </div>
              </div>
              <div className="w-px h-6 bg-border" />
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  {canEdit(userRole) ? (
                    <InlineEdit
                      value={project.name}
                      onSave={handleRenameProject}
                      className="text-xl font-bold text-foreground"
                      inputClassName="text-xl font-bold h-9"
                    />
                  ) : (
                    <h1 className="text-xl font-bold text-foreground truncate">{project.name}</h1>
                  )}
                  {userRole && (
                    <Badge
                      variant="outline"
                      data-walkthrough="project-role"
                      className={`text-xs ${ROLE_COLORS[userRole] || ""}`}
                    >
                      {userRole}
                    </Badge>
                  )}
                  {slackConnected && linkedChannel && (
                    <a
                      href={`slack://channel?id=${channelId}`}
                      className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-[#4A154B]/10 text-[#4A154B] hover:bg-[#4A154B]/20 transition-colors"
                      title={`Open #${linkedChannel.name} in Slack`}
                    >
                      <MessageSquare className="w-3 h-3" />
                      #{linkedChannel.name}
                    </a>
                  )}
                </div>
                <p className="text-sm text-muted-foreground">
                  {project.artists?.name || "Unknown Artist"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <ToolHelpButton onClick={() => walkthrough.replay()} />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/docs")}
                title="Documentation"
                className="text-muted-foreground hover:text-foreground"
              >
                <BookOpen className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main content with tabs */}
      <main className="container mx-auto px-4 py-6 max-w-5xl">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList data-walkthrough="project-tabs" className="mb-6">
            <TabsTrigger value="works" className="gap-1.5" disabled>
              <Music className="w-4 h-4" /> Works
            </TabsTrigger>
            <TabsTrigger value="files" className="gap-1.5">
              <FileText className="w-4 h-4" /> Files
            </TabsTrigger>
            <TabsTrigger value="audio" className="gap-1.5">
              <Volume2 className="w-4 h-4" /> Audio
            </TabsTrigger>
            <TabsTrigger data-walkthrough="project-members" value="members" className="gap-1.5">
              <Users className="w-4 h-4" /> Members
            </TabsTrigger>
            <TabsTrigger value="notes" className="gap-1.5">
              <StickyNote className="w-4 h-4" /> Notes
            </TabsTrigger>
            <TabsTrigger value="settings" className="gap-1.5">
              <Settings className="w-4 h-4" /> Settings
            </TabsTrigger>
          </TabsList>

          <TabsContent value="works">
            {projectId && (
              <WorksTab
                projectId={projectId}
                userRole={userRole}
                artistId={project.artist_id}
              />
            )}
          </TabsContent>

          <TabsContent value="files">
            {projectId && <FilesTab projectId={projectId} userRole={userRole} />}
          </TabsContent>

          <TabsContent value="audio">
            {projectId && (
              <AudioTab
                projectId={projectId}
                userRole={userRole}
                artistId={project.artist_id}
              />
            )}
          </TabsContent>

          <TabsContent value="members">
            {projectId && <MembersTab projectId={projectId} userRole={userRole} />}
          </TabsContent>

          <TabsContent value="notes">
            {projectId && <NotesView scope={{ projectId }} />}
          </TabsContent>

          <TabsContent value="settings">
            {projectId && (
              <SettingsTab
                projectId={projectId}
                userRole={userRole}
                project={project}
              />
            )}
          </TabsContent>
        </Tabs>
      </main>

      <ToolIntroModal
        config={TOOL_CONFIGS.project_detail}
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
    </div>
  );
};

export default ProjectDetail;
