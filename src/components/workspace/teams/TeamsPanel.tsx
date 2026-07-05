import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
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
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Loader2,
  Plus,
  X,
  UserPlus,
  Users,
  LogOut,
  Archive,
  ChevronDown,
  RotateCcw,
  Trash2,
} from "lucide-react";
import {
  useTeams,
  useCreateTeam,
  useArchiveTeam,
  useTeamMembers,
  useTeamInvites,
  useInviteTeamMember,
  useUpdateTeamMemberRole,
  useRemoveTeamMember,
  useCancelTeamInvite,
  useArchivedTeams,
  useRestoreTeam,
  useDeleteTeam,
} from "@/hooks/useTeams";
import type { TeamMember } from "@/types/teams";
import { DeleteConfirmDialog } from "@/components/workspace/boards/DeleteConfirmDialog";
import { useAuth } from "@/contexts/AuthContext";

const ROLE_COLORS: Record<string, string> = {
  admin: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  member: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
};

function getInitials(name: string): string {
  return name
    .split(" ")
    .map((w) => w[0])
    .filter(Boolean)
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export default function TeamsPanel() {
  const { user } = useAuth();
  const { data: teams, isLoading: teamsLoading, isError: teamsError } = useTeams();
  const createTeam = useCreateTeam();
  const archiveTeam = useArchiveTeam();
  const inviteMember = useInviteTeamMember();
  const updateRole = useUpdateTeamMemberRole();
  const removeMember = useRemoveTeamMember();
  const cancelInvite = useCancelTeamInvite();
  const { data: archivedTeams } = useArchivedTeams();
  const restoreTeam = useRestoreTeam();
  const deleteTeam = useDeleteTeam();

  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newTeamName, setNewTeamName] = useState("");
  const [newTeamDescription, setNewTeamDescription] = useState("");
  const [inviteDialogOpen, setInviteDialogOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("member");
  const [archivedOpen, setArchivedOpen] = useState(false);
  const [deleteTeamId, setDeleteTeamId] = useState<string | null>(null);

  // Derive the team targeted for permanent deletion from the archived list. If the list
  // refetches away this id while the dialog is open, target becomes undefined and onConfirm
  // no-ops (guarded below).
  const deleteTarget = archivedTeams?.find((t) => t.id === deleteTeamId);

  // Fall back to the first team when nothing is selected (or the selection went away
  // after a leave/archive).
  const teamId =
    selectedTeamId && teams?.some((t) => t.id === selectedTeamId)
      ? selectedTeamId
      : teams?.[0]?.id;
  const selectedTeam = teams?.find((t) => t.id === teamId);
  const isAdmin = selectedTeam?.my_role === "admin";

  const { data: members, isLoading: membersLoading, isError: membersError } = useTeamMembers(teamId);
  // Backend only allows admins to list invites — don't fetch as a member.
  const { data: pendingInvites } = useTeamInvites(isAdmin ? teamId : undefined);

  const myMembership = members?.find((m) => m.user_id === user?.id);
  // Sole admin AND sole member → leaving would be rejected; offer Archive instead.
  const isSoleAdminSoleMember =
    !!myMembership && myMembership.role === "admin" && (members?.length ?? 0) === 1;

  // Reset drafts whenever a dialog closes (Cancel, Esc, overlay click).
  const handleCreateDialogOpenChange = (open: boolean) => {
    setCreateDialogOpen(open);
    if (!open) {
      setNewTeamName("");
      setNewTeamDescription("");
    }
  };

  const handleInviteDialogOpenChange = (open: boolean) => {
    setInviteDialogOpen(open);
    if (!open) setInviteEmail("");
  };

  const handleCreateTeam = () => {
    if (!newTeamName.trim()) return;
    createTeam.mutate(
      {
        name: newTeamName.trim(),
        description: newTeamDescription.trim() || undefined,
      },
      {
        onSuccess: (team) => {
          setNewTeamName("");
          setNewTeamDescription("");
          setCreateDialogOpen(false);
          if (team?.id) setSelectedTeamId(team.id);
        },
      }
    );
  };

  const handleInvite = () => {
    if (!inviteEmail.trim() || !teamId) return;
    inviteMember.mutate(
      { teamId, email: inviteEmail.trim(), role: inviteRole },
      {
        onSuccess: () => {
          setInviteEmail("");
          setInviteRole("member");
          setInviteDialogOpen(false);
        },
      }
    );
  };

  const handleLeave = () => {
    if (!teamId || !myMembership) return;
    removeMember.mutate(
      { teamId, memberId: myMembership.id, successMessage: "You left the team" },
      { onSuccess: () => setSelectedTeamId(null) }
    );
  };

  const handleArchive = () => {
    if (!teamId) return;
    archiveTeam.mutate(teamId, { onSuccess: () => setSelectedTeamId(null) });
  };

  if (teamsLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (teamsError) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Users className="w-10 h-10 text-destructive/40 mb-3" />
        <p className="text-sm text-muted-foreground">Couldn't load your teams</p>
        <p className="text-xs text-muted-foreground/60 mt-1">Please try refreshing the page</p>
      </div>
    );
  }

  const createTeamDialog = (
    <Dialog open={createDialogOpen} onOpenChange={handleCreateDialogOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Create Team</DialogTitle>
          <DialogDescription>
            Name your team — you can invite people right after.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="team-name">Team name</Label>
            <Input
              id="team-name"
              placeholder="e.g. My Band"
              value={newTeamName}
              onChange={(e) => setNewTeamName(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="team-description">Description (optional)</Label>
            <Textarea
              id="team-description"
              placeholder="What is this team for?"
              value={newTeamDescription}
              onChange={(e) => setNewTeamDescription(e.target.value)}
              rows={3}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleCreateDialogOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleCreateTeam} disabled={!newTeamName.trim() || createTeam.isPending}>
            {createTeam.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Create Team
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );

  // Empty state — no teams yet
  if (!teams || teams.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <Users className="w-12 h-12 text-muted-foreground/40 mb-4" />
        <h3 className="text-lg font-semibold text-foreground mb-1">No teams yet</h3>
        <p className="text-sm text-muted-foreground max-w-sm mb-4">
          Teams let you share boards and tasks with the people you work with. Create one to get
          started.
        </p>
        <Button onClick={() => setCreateDialogOpen(true)}>
          <Plus className="w-4 h-4 mr-2" /> Create Team
        </Button>
        {createTeamDialog}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Team selector + create */}
      <div className="flex flex-wrap items-center gap-2">
        <Select value={teamId} onValueChange={(v) => setSelectedTeamId(v)}>
          <SelectTrigger className="w-56" aria-label="Select team">
            <SelectValue placeholder="Select a team" />
          </SelectTrigger>
          <SelectContent>
            {teams.map((team) => (
              <SelectItem key={team.id} value={team.id}>
                {team.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button size="sm" variant="outline" onClick={() => setCreateDialogOpen(true)}>
          <Plus className="w-4 h-4 mr-2" /> Create Team
        </Button>
      </div>

      {selectedTeam?.description && (
        <p className="text-sm text-muted-foreground -mt-3">{selectedTeam.description}</p>
      )}

      {/* Members */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Users className="w-4 h-4" /> Team Members
          </h3>
          {isAdmin && (
            <Button size="sm" variant="outline" onClick={() => setInviteDialogOpen(true)}>
              <UserPlus className="w-4 h-4 mr-2" /> Invite Member
            </Button>
          )}
        </div>

        {membersLoading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : membersError ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <Users className="w-8 h-8 text-destructive/40 mb-2" />
            <p className="text-sm text-muted-foreground">Couldn't load members</p>
          </div>
        ) : !members || members.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Users className="w-10 h-10 text-muted-foreground/40 mb-3" />
            <p className="text-sm text-muted-foreground">No members yet</p>
            <p className="text-xs text-muted-foreground/60 mt-1">
              Invite people to work together on this team's boards
            </p>
          </div>
        ) : (
          <div className="grid gap-2">
            {members.map((member: TeamMember) => {
              const displayName = member.full_name || "Member";
              const isSelf = member.user_id === user?.id;
              return (
                <Card key={member.id} className="p-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 text-xs font-medium text-primary overflow-hidden">
                      {member.avatar_url ? (
                        <img
                          src={member.avatar_url}
                          alt={displayName}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        getInitials(displayName)
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-foreground truncate">
                        {displayName}
                        {isSelf && <span className="text-muted-foreground font-normal"> (you)</span>}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      {isAdmin && !isSelf ? (
                        <Select
                          value={member.role}
                          onValueChange={(newRole) =>
                            updateRole.mutate({ teamId: teamId!, memberId: member.id, role: newRole })
                          }
                        >
                          <SelectTrigger
                            className="h-7 w-24 text-xs"
                            aria-label={`Change role for ${displayName}`}
                          >
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="admin">Admin</SelectItem>
                            <SelectItem value="member">Member</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <Badge variant="outline" className={`text-xs ${ROLE_COLORS[member.role] || ""}`}>
                          {member.role}
                        </Badge>
                      )}
                      {isAdmin && !isSelf && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                          aria-label={`Remove ${displayName}`}
                          title={`Remove ${displayName}`}
                          onClick={() => removeMember.mutate({ teamId: teamId!, memberId: member.id })}
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

        {/* Pending Invites (admins only — backend restricts the list) */}
        {isAdmin && pendingInvites && pendingInvites.length > 0 && (
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
                      Invited as {inv.role}
                      {inv.expires_at &&
                        ` — expires ${new Date(inv.expires_at).toLocaleDateString()}`}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                    aria-label={`Cancel invite to ${inv.email}`}
                    title={`Cancel invite to ${inv.email}`}
                    onClick={() => cancelInvite.mutate({ teamId: teamId!, inviteId: inv.id })}
                  >
                    <X className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Leave / Archive footer */}
      {myMembership && (
        <div className="pt-4 border-t border-border">
          {isSoleAdminSoleMember ? (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="text-destructive hover:text-destructive">
                  <Archive className="w-4 h-4 mr-2" /> Archive Team
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Archive this team?</AlertDialogTitle>
                  <AlertDialogDescription>
                    You're the only person in "{selectedTeam?.name}". Archiving will hide the team
                    and its boards from your workspace. This can't be undone from the app.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleArchive}>Archive Team</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          ) : (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" size="sm" className="text-destructive hover:text-destructive">
                  <LogOut className="w-4 h-4 mr-2" /> Leave Team
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Leave this team?</AlertDialogTitle>
                  <AlertDialogDescription>
                    You'll lose access to "{selectedTeam?.name}" and its boards. A team admin can
                    invite you back later.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleLeave}>Leave Team</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      )}

      {/* Archived teams (admin-only — endpoint returns only teams the caller admins).
          Hidden entirely when the caller has no archived teams; auto-hides once the last
          archived team is restored/deleted (driven by the useArchivedTeams refetch). */}
      {archivedTeams && archivedTeams.length > 0 && (
        <Collapsible
          open={archivedOpen}
          onOpenChange={setArchivedOpen}
          className="pt-4 border-t border-border"
        >
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="text-muted-foreground">
              <Archive className="w-4 h-4 mr-2" /> Archived
              <span className="ml-1.5 text-xs text-muted-foreground/70">
                ({archivedTeams.length})
              </span>
              <ChevronDown
                className={`w-4 h-4 ml-2 transition-transform ${archivedOpen ? "rotate-180" : ""}`}
              />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 space-y-2">
            {archivedTeams.map((t) => (
              <Card key={t.id} className="p-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">{t.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {t.boards ?? 0} boards · {t.tasks ?? 0} tasks · {t.members ?? 0} members
                    </p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => restoreTeam.mutate(t.id)}
                      disabled={restoreTeam.isPending}
                    >
                      <RotateCcw className="w-3.5 h-3.5 mr-2" /> Restore
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-destructive hover:text-destructive"
                      onClick={() => setDeleteTeamId(t.id)}
                    >
                      <Trash2 className="w-3.5 h-3.5 mr-2" /> Delete team permanently…
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </CollapsibleContent>
        </Collapsible>
      )}

      {/* Single permanent-delete dialog, bound to the row-selected archived team.
          confirmName is the BARE team name — the dialog gates the typed delete-<name> locally,
          the server re-compares normalized names. */}
      <DeleteConfirmDialog
        open={!!deleteTeamId}
        onOpenChange={(o) => !o && setDeleteTeamId(null)}
        name={deleteTarget?.name ?? ""}
        resourceType="team"
        impact={
          deleteTarget
            ? `${deleteTarget.boards ?? 0} boards, ${deleteTarget.tasks ?? 0} tasks, ${deleteTarget.members ?? 0} members`
            : undefined
        }
        isPending={deleteTeam.isPending}
        onConfirm={() =>
          deleteTarget &&
          deleteTeam.mutate(
            { teamId: deleteTarget.id, confirmName: deleteTarget.name },
            { onSuccess: () => setDeleteTeamId(null) }
          )
        }
      />

      {/* Invite Dialog */}
      <Dialog open={inviteDialogOpen} onOpenChange={handleInviteDialogOpenChange}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Invite Member</DialogTitle>
            <DialogDescription>They'll get an invite by email or in-app.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="team-invite-email">Email</Label>
              <Input
                id="team-invite-email"
                type="email"
                placeholder="name@example.com"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={inviteRole} onValueChange={setInviteRole}>
                <SelectTrigger aria-label="Invite role">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="member">Member</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => handleInviteDialogOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleInvite} disabled={!inviteEmail.trim() || inviteMember.isPending}>
              {inviteMember.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Send Invite
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {createTeamDialog}
    </div>
  );
}
