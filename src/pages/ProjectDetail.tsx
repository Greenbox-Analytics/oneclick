import { useState } from "react";
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
  Music, ArrowLeft, Loader2, Plus,
  FileText, Volume2, Users, Settings, StickyNote, BookOpen,
} from "lucide-react";
import { toast } from "sonner";

import WorksTab from "@/components/project/WorksTab";
import FilesTab from "@/components/project/FilesTab";
import AudioTab from "@/components/project/AudioTab";
import MembersTab from "@/components/project/MembersTab";
import SettingsTab from "@/components/project/SettingsTab";
import AddWorkDialog from "@/components/project/AddWorkDialog";
import NotesView from "@/components/notes/NotesView";

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

  const [addWorkOpen, setAddWorkOpen] = useState(false);

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
                      className={`text-xs ${ROLE_COLORS[userRole] || ""}`}
                    >
                      {userRole}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground">
                  {project.artists?.name || "Unknown Artist"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/docs")}
                title="Documentation"
                className="text-muted-foreground hover:text-foreground"
              >
                <BookOpen className="w-4 h-4" />
              </Button>
              {canEdit(userRole) && (
                <Button size="sm" onClick={() => setAddWorkOpen(true)}>
                  <Plus className="w-4 h-4 mr-2" /> Add Work
                </Button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content with tabs */}
      <main className="container mx-auto px-4 py-6 max-w-5xl">
        <Tabs defaultValue="works">
          <TabsList className="mb-6">
            <TabsTrigger value="works" className="gap-1.5">
              <Music className="w-4 h-4" /> Works
            </TabsTrigger>
            <TabsTrigger value="files" className="gap-1.5">
              <FileText className="w-4 h-4" /> Files
            </TabsTrigger>
            <TabsTrigger value="audio" className="gap-1.5">
              <Volume2 className="w-4 h-4" /> Audio
            </TabsTrigger>
            <TabsTrigger value="members" className="gap-1.5">
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

      {/* Add Work Dialog (accessible from header button) */}
      {canEdit(userRole) && projectId && (
        <AddWorkDialog
          open={addWorkOpen}
          onOpenChange={setAddWorkOpen}
          projectId={projectId}
          artistId={project.artist_id}
        />
      )}
    </div>
  );
};

export default ProjectDetail;
