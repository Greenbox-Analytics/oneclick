import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Loader2, Plus, X, UserPlus, Users, Link as LinkIcon } from "lucide-react";
import {
  useProjectMembers,
  useAddProjectMember,
  useUpdateMemberRole,
  useRemoveProjectMember,
  usePendingInvites,
  useCancelPendingInvite,
  type ProjectMember,
} from "@/hooks/useProjectMembers";
import { useWorksByProject } from "@/hooks/useRegistry";
import { supabase } from "@/integrations/supabase/client";

interface MembersTabProps {
  projectId: string;
  userRole: string | null;
}

interface WorkCollaborator {
  id: string;
  name: string | null;
  email: string;
  status: string;
  works_registry?: { id: string; title: string } | null;
}

const ROLE_COLORS: Record<string, string> = {
  owner: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  admin: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  editor: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  viewer: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

const COLLAB_STATUS_COLORS: Record<string, string> = {
  confirmed: "bg-emerald-500/15 text-emerald-400",
  invited: "bg-amber-500/15 text-amber-400",
  declined: "bg-red-500/15 text-red-400",
  revoked: "bg-gray-500/15 text-gray-400",
};

const canManageMembers = (role: string | null) => role === "owner" || role === "admin";

function getInitials(name: string): string {
  return name
    .split(" ")
    .map((w) => w[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export default function MembersTab({ projectId, userRole }: MembersTabProps) {
  const { data: members, isLoading: membersLoading, isError: membersError } = useProjectMembers(projectId);
  const { data: pendingInvites } = usePendingInvites(projectId);
  const { data: works } = useWorksByProject(projectId);
  const addMember = useAddProjectMember();
  const updateRole = useUpdateMemberRole();
  const removeMember = useRemoveProjectMember();
  const cancelInvite = useCancelPendingInvite();

  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("editor");

  // Fetch work-level collaborators for this project's works
  const workIds = (works || []).map((w) => w.id);
  const { data: workCollaborators } = useQuery({
    queryKey: ["project-work-collaborators", projectId, workIds],
    queryFn: async () => {
      if (workIds.length === 0) return [];
      const { data, error } = await supabase
        .from("registry_collaborators" as never)
        .select("*, works_registry(id, title)")
        .in("work_id", workIds);
      if (error) return [];
      return (data as WorkCollaborator[] | null) || [];
    },
    enabled: workIds.length > 0,
  });

  // Fetch user profiles for members to get names
  // Note: profiles table has full_name, first_name, last_name, given_name — no email or avatar_url
  // Email is on team_cards table (if exists) or auth.users (not queryable from client)
  const memberUserIds = (members || []).map((m) => m.user_id);
  const { data: memberProfiles, isLoading: profilesLoading } = useQuery({
    queryKey: ["member-profiles", memberUserIds],
    queryFn: async () => {
      if (memberUserIds.length === 0) return [];
      const { data, error } = await supabase
        .from("profiles")
        .select("id, full_name, first_name, last_name, given_name")
        .in("id", memberUserIds);
      if (error) { console.error("Error fetching profiles:", error); return []; }
      return data || [];
    },
    enabled: memberUserIds.length > 0,
  });

  // Fetch emails from team_cards (has email column)
  const { data: teamCards, isLoading: teamCardsLoading } = useQuery({
    queryKey: ["member-teamcards", memberUserIds],
    queryFn: async () => {
      if (memberUserIds.length === 0) return [];
      const { data, error } = await supabase
        .from("team_cards")
        .select("user_id, email, avatar_url")
        .in("user_id", memberUserIds);
      if (error) return [];
      return data || [];
    },
    enabled: memberUserIds.length > 0,
  });

  const memberDetailsLoading =
    memberUserIds.length > 0 && (profilesLoading || teamCardsLoading);

  const profileMap = new Map<string, { full_name: string | null; first_name: string | null; last_name: string | null; given_name: string | null }>();
  for (const p of memberProfiles || []) {
    profileMap.set(p.id, p);
  }

  const teamCardMap = new Map<string, { email: string | null; avatar_url: string | null }>();
  for (const tc of teamCards || []) {
    teamCardMap.set(tc.user_id, tc);
  }

  const handleInvite = () => {
    if (!inviteEmail.trim()) return;
    addMember.mutate(
      { projectId, email: inviteEmail.trim(), role: inviteRole },
      {
        onSuccess: () => {
          setInviteEmail("");
          setInviteRole("editor");
          setInviteDialogOpen(false);
        },
      }
    );
  };

  if (membersLoading || memberDetailsLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (membersError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Users className="w-10 h-10 text-destructive/40 mb-3" />
        <p className="text-sm text-muted-foreground">Failed to load members</p>
        <p className="text-xs text-muted-foreground/60 mt-1">Please try refreshing the page</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Project Members */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Users className="w-4 h-4" /> Project Members
          </h3>
          {canManageMembers(userRole) && (
            <Button size="sm" variant="outline" onClick={() => setInviteDialogOpen(true)}>
              <UserPlus className="w-4 h-4 mr-2" /> Invite Member
            </Button>
          )}
        </div>

        {(!members || members.length === 0) ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Users className="w-10 h-10 text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground">No members yet</p>
            <p className="text-xs text-muted-foreground/60 mt-1">Invite collaborators to work together on this project</p>
          </div>
        ) : (
          <div className="grid gap-2">
            {members.map((member: ProjectMember) => {
              const profile = profileMap.get(member.user_id);
              const tc = teamCardMap.get(member.user_id);
              const displayName =
                profile?.full_name ||
                profile?.given_name ||
                [profile?.first_name, profile?.last_name].filter(Boolean).join(" ") ||
                "Member";
              const email = tc?.email || "";
              return (
                <Card key={member.id} className="p-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 text-xs font-medium text-primary">
                      {getInitials(displayName)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">{displayName}</p>
                      {email && (
                        <p className="text-xs text-muted-foreground truncate">{email}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {canManageMembers(userRole) && member.role !== "owner" ? (
                        <Select
                          value={member.role}
                          onValueChange={(newRole) =>
                            updateRole.mutate({ projectId, memberId: member.id, role: newRole })
                          }
                        >
                          <SelectTrigger className="h-7 w-24 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="admin">Admin</SelectItem>
                            <SelectItem value="editor">Editor</SelectItem>
                            <SelectItem value="viewer">Viewer</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <Badge
                          variant="outline"
                          className={`text-xs ${ROLE_COLORS[member.role] || ""}`}
                        >
                          {member.role}
                        </Badge>
                      )}
                      {canManageMembers(userRole) && member.role !== "owner" && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                          onClick={() => removeMember.mutate({ projectId, memberId: member.id })}
                        >
                          <X className="w-3.5 h-3.5" />
                        </Button>
                      )}
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* Pending Invites */}
        {pendingInvites && pendingInvites.length > 0 && (
          <div className="space-y-2 mt-4">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Pending Invites
            </h4>
            {pendingInvites.map((inv) => (
              <Card key={inv.id} className="p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-foreground">{inv.email}</p>
                    <p className="text-xs text-muted-foreground">
                      Invited as {inv.role} — expires{" "}
                      {new Date(inv.expires_at).toLocaleDateString()}
                    </p>
                  </div>
                  {canManageMembers(userRole) && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                      onClick={() => cancelInvite.mutate({ projectId, inviteId: inv.id })}
                    >
                      <X className="w-3.5 h-3.5" />
                    </Button>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Work-Only Collaborators */}
      {workCollaborators && workCollaborators.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <LinkIcon className="w-4 h-4" /> Work-Only Collaborators
          </h3>
          <div className="grid gap-2">
            {workCollaborators.map((collab: WorkCollaborator) => {
              const workTitle = collab.works_registry?.title || "Unknown Work";
              return (
                <Card key={collab.id} className="p-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center shrink-0 text-xs font-medium text-muted-foreground">
                      {getInitials(collab.name || collab.email || "?")}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {collab.name || collab.email}
                      </p>
                      <p className="text-xs text-muted-foreground truncate">
                        {collab.email}
                        {" — on "}
                        <span className="text-primary">{workTitle}</span>
                      </p>
                    </div>
                    <Badge
                      className={`text-xs border-0 shrink-0 ${
                        COLLAB_STATUS_COLORS[collab.status] || COLLAB_STATUS_COLORS.invited
                      }`}
                    >
                      {collab.status === "confirmed"
                        ? "Confirmed"
                        : collab.status === "declined"
                        ? "Declined"
                        : collab.status === "revoked"
                        ? "Revoked"
                        : "Invited"}
                    </Badge>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>
      )}

      {/* Invite Dialog */}
      <Dialog open={inviteDialogOpen} onOpenChange={setInviteDialogOpen}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Invite Member</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="invite-email">Email</Label>
              <Input
                id="invite-email"
                type="email"
                placeholder="name@example.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={inviteRole} onValueChange={setInviteRole}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="editor">Editor</SelectItem>
                  <SelectItem value="viewer">Viewer</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setInviteDialogOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleInvite}
              disabled={!inviteEmail.trim() || addMember.isPending}
            >
              {addMember.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Send Invite
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
