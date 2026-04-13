import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { supabase } from "@/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Loader2, Trash2, LogOut, Settings, UserMinus, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { useRemoveProjectMember, useProjectMembers } from "@/hooks/useProjectMembers";
import { useIntegrations } from "@/hooks/useIntegrations";
import { ProjectSlackSettings } from "./integrations/ProjectSlackSettings";

interface SettingsTabProps {
  projectId: string;
  userRole: string | null;
  project: {
    id: string;
    name: string;
    description: string | null;
    artist_id: string;
    artists?: { name: string; user_id: string } | null;
  };
}

const canEdit = (role: string | null) => role === "owner" || role === "admin";

export default function SettingsTab({ projectId, userRole, project }: SettingsTabProps) {
  const navigate = useNavigate();
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const { data: members } = useProjectMembers(projectId);
  const removeMember = useRemoveProjectMember();

  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description || "");
  const [saving, setSaving] = useState(false);

  // Update project mutation
  const updateProject = useMutation({
    mutationFn: async ({ name, description }: { name: string; description: string }) => {
      const { error } = await supabase
        .from("projects")
        .update({ name: name.trim(), description: description.trim() || null })
        .eq("id", projectId);
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project-detail", projectId] });
      toast.success("Project updated");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  // Delete project mutation
  const deleteProject = useMutation({
    mutationFn: async () => {
      // 1. Delete files from storage (best-effort — storage doesn't have RLS)
      const { data: files } = await supabase
        .from("project_files")
        .select("id, file_path")
        .eq("project_id", projectId);

      if (files && files.length > 0) {
        const paths = files.map((f) => f.file_path).filter(Boolean);
        if (paths.length > 0) {
          try {
            await supabase.storage.from("project-files").remove(paths);
          } catch {
            // Storage cleanup is best-effort
          }
        }
        // 2. Delete file records
        const { error: filesError } = await supabase
          .from("project_files")
          .delete()
          .eq("project_id", projectId);
        if (filesError) {
          console.error("Failed to delete project files:", filesError);
        }
      }

      // 3. Delete related records that may block cascade
      await supabase.from("drive_sync_mappings").delete().eq("project_id", projectId);
      await supabase.from("project_notification_settings").delete().eq("project_id", projectId);

      // 4. Delete the project (cascades to project_members, pending_invites, etc.)
      const { error } = await supabase.from("projects").delete().eq("id", projectId);
      if (error) throw error;
    },
    onSuccess: () => {
      toast.success("Project deleted");
      navigate("/portfolio");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error("Project name is required");
      return;
    }
    setSaving(true);
    await updateProject.mutateAsync({ name, description });
    setSaving(false);
  };

  const handleLeave = () => {
    const myMember = members?.find((m) => m.user_id === user?.id);
    if (!myMember) return;
    removeMember.mutate(
      { projectId, memberId: myMember.id },
      {
        onSuccess: () => {
          toast.success("You left the project");
          navigate("/portfolio");
        },
      }
    );
  };

  const isOwner = userRole === "owner";
  const { connections } = useIntegrations();
  const slackConnected = connections.some(c => c.provider === "slack" && c.status === "active");

  return (
    <div className="space-y-6 max-w-2xl">
      {/* General Settings */}
      <Card className="p-6 space-y-5">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Settings className="w-4 h-4 text-muted-foreground" />
          General
        </h3>

        <div className="space-y-2">
          <Label htmlFor="project-name">Project Name</Label>
          <Input
            id="project-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={!canEdit(userRole)}
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="project-description">Description</Label>
          <Textarea
            id="project-description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Add a description..."
            rows={3}
            disabled={!canEdit(userRole)}
          />
        </div>

        <div className="space-y-2">
          <Label>Primary Artist</Label>
          <Badge variant="outline" className="text-sm px-3 py-1">
            {project.artists?.name || "Unknown"}
          </Badge>
        </div>

        {canEdit(userRole) && (
          <Button onClick={handleSave} disabled={saving || !name.trim()}>
            {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Save Changes
          </Button>
        )}
      </Card>

      {slackConnected && (
        <>
          <Separator />
          <ProjectSlackSettings projectId={projectId} />
        </>
      )}

      <Separator />

      {/* Leave Project (non-owners only) */}
      {!isOwner && userRole && (
        <>
          <Card className="p-6 space-y-3">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <UserMinus className="w-4 h-4 text-muted-foreground" />
              Leave Project
            </h3>
            <p className="text-xs text-muted-foreground">
              You will lose access to this project. This action cannot be undone.
            </p>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <LogOut className="w-4 h-4 mr-2" /> Leave Project
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Leave project?</AlertDialogTitle>
                  <AlertDialogDescription>
                    You will be removed from "{project.name}" and lose access. You'll need to be re-invited to rejoin.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleLeave}>Leave</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </Card>
          <Separator />
        </>
      )}

      {/* Danger Zone (owner only) */}
      {isOwner && (
        <Card className="p-6 space-y-3 border-destructive/30 bg-gradient-to-r from-destructive/5 to-transparent">
          <h3 className="text-sm font-semibold text-destructive flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            Danger Zone
          </h3>
          <p className="text-xs text-muted-foreground">
            Deleting this project will permanently remove all associated files and data.
          </p>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="sm">
                <Trash2 className="w-4 h-4 mr-2" /> Delete Project
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete project?</AlertDialogTitle>
                <AlertDialogDescription>
                  This will permanently delete "{project.name}" and all its files. This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={() => deleteProject.mutate()}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  {deleteProject.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                  Delete
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </Card>
      )}
    </div>
  );
}
