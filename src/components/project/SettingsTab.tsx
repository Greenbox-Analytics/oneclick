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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
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
import { Loader2, Trash2, LogOut, Settings, UserMinus, AlertTriangle, BarChart2, Building2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { useRemoveProjectMember, useProjectMembers } from "@/hooks/useProjectMembers";
import { useIntegrations } from "@/hooks/useIntegrations";
import { ProjectSlackSettings } from "./integrations/ProjectSlackSettings";
import { useDeleteProjectRoyalties } from "@/hooks/useRoyalties";
import { useEntitlements, type OrgBillingContext } from "@/hooks/useEntitlements";
import { useProjectOrgLink, useLinkProjectToOrg, useUnlinkProjectFromOrg } from "@/hooks/useOrgs";

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

// Licensing Phase C (spec §6, rules 10-11, plan Task 8) — EXACT consent copy,
// shown at the moment an owner links a project. Every clause is load-bearing:
// admins managing team access, credits (including the owner's OWN work)
// always billing the org while linked, storage counting against the org's
// allowance, and ownership never changing.
const ORG_LINK_CONSENT_COPY =
  "Your organization's admins will be able to manage who on your team can access this project, and work anyone with a seat does here will use the organization's credits — including yours: while linked, this project always bills the organization, even if it runs out of credits (you can unlink anytime). Storage you use here counts against the organization's larger allowance. Your ownership never changes.";

// Phase B rule 13's block-don't-bill covers the billing side of unlinking;
// this is the storage-specific warning the plan requires verbatim.
const ORG_UNLINK_STORAGE_WARNING =
  "If you're using more storage than your personal plan includes, you won't be able to upload new files after unlinking until you're back under your limit.";

/** Owner-only "Link to organization" control (plan Task 8). Gated on the
 * owner holding >=1 ACTIVE seat in an ACTIVE org (probed via
 * `availableContexts`, same signal the billing-context switcher uses) OR
 * the project already being linked — an owner who later loses their seat
 * can still see and undo an existing link, since unlinking never re-checks
 * seat status (only ownership, `orgs/projects.py::unlink_project`). */
function OrganizationLinkSection({ projectId }: { projectId: string }) {
  const { data: entitlements } = useEntitlements();
  const { data: link, isLoading: linkLoading } = useProjectOrgLink(projectId);
  const linkProject = useLinkProjectToOrg();
  const unlinkProject = useUnlinkProjectFromOrg();
  const [selectedOrgId, setSelectedOrgId] = useState("");

  const eligibleOrgs = (entitlements?.availableContexts ?? []).filter(
    (c) => c.type === "org" && !c.pending,
  ) as (OrgBillingContext & { type: "org"; pending: boolean })[];

  if (linkLoading) return null;
  if (!link && eligibleOrgs.length === 0) return null;

  const handleLink = () => {
    if (!selectedOrgId) return;
    linkProject.mutate({ orgId: selectedOrgId, projectId }, { onSuccess: () => setSelectedOrgId("") });
  };

  const handleUnlink = () => {
    if (!link) return;
    unlinkProject.mutate({ orgId: link.orgId, projectId });
  };

  return (
    <>
      <Card className="p-6 space-y-4">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Building2 className="w-4 h-4 text-muted-foreground" />
          Organization
        </h3>

        {link ? (
          <>
            <p className="text-xs text-muted-foreground">
              Linked to <span className="font-medium text-foreground">{link.orgName ?? "your organization"}</span>
              {link.linkedAt &&
                ` since ${new Date(link.linkedAt).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}`}
              . Its admins manage who on your team can access this project, and work here bills the
              organization's credits.
            </p>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" disabled={unlinkProject.isPending}>
                  {unlinkProject.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                  Unlink
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Unlink this project?</AlertDialogTitle>
                  <AlertDialogDescription>
                    Your organization's admins will no longer manage access to this project, and new work here
                    will bill your own plan instead. {ORG_UNLINK_STORAGE_WARNING}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel disabled={unlinkProject.isPending}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    disabled={unlinkProject.isPending}
                    onClick={(e) => {
                      e.preventDefault();
                      handleUnlink();
                    }}
                  >
                    {unlinkProject.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Unlink
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </>
        ) : (
          <>
            <p className="text-xs text-muted-foreground">{ORG_LINK_CONSENT_COPY}</p>
            <div className="flex items-center gap-2">
              <Select value={selectedOrgId} onValueChange={setSelectedOrgId}>
                <SelectTrigger className="w-56" aria-label="Choose an organization to link">
                  <SelectValue placeholder="Choose an organization" />
                </SelectTrigger>
                <SelectContent>
                  {eligibleOrgs.map((o) => (
                    <SelectItem key={o.orgId} value={o.orgId}>
                      {o.orgName}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button size="sm" onClick={handleLink} disabled={!selectedOrgId || linkProject.isPending}>
                {linkProject.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                Link
              </Button>
            </div>
          </>
        )}
      </Card>
      <Separator />
    </>
  );
}

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

  const deleteProjectRoyalties = useDeleteProjectRoyalties();

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

      {isOwner && <OrganizationLinkSection projectId={projectId} />}

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
                <Button variant="outline" size="sm" disabled={removeMember.isPending}>
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
                  <AlertDialogCancel disabled={removeMember.isPending}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    disabled={removeMember.isPending}
                    onClick={(e) => { e.preventDefault(); handleLeave(); }}
                  >
                    {removeMember.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
                    Leave
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </Card>
          <Separator />
        </>
      )}

      {/* Royalty tracking (owner only) */}
      {isOwner && (
        <>
          <Card className="p-6 space-y-3">
            <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-muted-foreground" />
              Royalty tracking
            </h3>
            <p className="text-xs text-muted-foreground">
              Remove all royalty entries for this project. The next OneClick run will recompute them from scratch.
            </p>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={deleteProjectRoyalties.isPending}
                  className="text-destructive border-destructive/40 hover:bg-destructive/10 hover:text-destructive"
                >
                  {deleteProjectRoyalties.isPending && (
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  )}
                  Delete royalty entries
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete royalty entries for this project?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This removes all royalty tracking entries <strong>and the cached OneClick calculations</strong> for
                    this project. The next OneClick run will recompute. Already-issued invoices are kept but may show as
                    orphaned. This can't be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel disabled={deleteProjectRoyalties.isPending}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    disabled={deleteProjectRoyalties.isPending}
                    onClick={(e) => {
                      e.preventDefault();
                      deleteProjectRoyalties.mutate(projectId, {
                        onSuccess: (res) => {
                          toast.success(
                            `Removed ${res.deleted_calculations} calculation(s).`,
                          );
                        },
                        onError: (err: Error) => toast.error(err.message),
                      });
                    }}
                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  >
                    {deleteProjectRoyalties.isPending && (
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    )}
                    Delete entries
                  </AlertDialogAction>
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
              <Button variant="destructive" size="sm" disabled={deleteProject.isPending}>
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
                <AlertDialogCancel disabled={deleteProject.isPending}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  disabled={deleteProject.isPending}
                  onClick={(e) => {
                    e.preventDefault();
                    deleteProject.mutate();
                  }}
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
