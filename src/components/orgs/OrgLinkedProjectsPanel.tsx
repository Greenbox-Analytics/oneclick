// src/components/orgs/OrgLinkedProjectsPanel.tsx
// Admin console (Licensing Phase C, spec §6, plan Task 8): linked projects +
// per-project member matrix. Linking/unlinking is the project OWNER's alone
// (rule 1, managed from the project's own settings tab) — this panel is
// VIEW + manage-seat-access only: admins grant/adjust/remove which of their
// org's seats can reach a linked project, driving Task 3's endpoints. They
// can never create or remove the link itself here.
//
// The per-member role select has no "current role" to preload from: no
// endpoint exposes a linked project's existing project_members rows to the
// org admin (only the aggregate `orgGrantedMemberCount`), so each row starts
// at "viewer" and shows the OUTCOME of the admin's last action (granted as
// X / has independent access / access removed) rather than a persisted
// state — an honest reflection of what data is actually available here.
import { useState } from "react";
import { ChevronDown, ChevronRight, FolderKanban, Loader2, UserMinus } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  useOrgLinkedProjects,
  useSetOrgProjectMemberRole,
  useRemoveOrgProjectMember,
  type OrgLinkedProject,
  type OrgSeatUsage,
  type OrgProjectRole,
} from "@/hooks/useOrgs";
import { fmtDate } from "@/lib/utils";

type RowFeedback =
  | { kind: "granted"; role: OrgProjectRole }
  | { kind: "organic"; detail: string }
  | { kind: "removed" };

function MemberMatrixRow({
  orgId,
  projectId,
  seat,
}: {
  orgId: string;
  projectId: string;
  seat: OrgSeatUsage;
}) {
  const [role, setRole] = useState<OrgProjectRole>("viewer");
  const [feedback, setFeedback] = useState<RowFeedback | null>(null);
  const setMemberRole = useSetOrgProjectMemberRole();
  const removeMember = useRemoveOrgProjectMember();
  const busy = setMemberRole.isPending || removeMember.isPending;

  const handleSetRole = () => {
    setMemberRole.mutate(
      { orgId, projectId, memberId: seat.orgMemberId, role },
      {
        onSuccess: (res) => {
          if (res.status === "organic") {
            setFeedback({ kind: "organic", detail: res.detail || "This member already has access on their own." });
          } else {
            setFeedback({ kind: "granted", role: (res.member?.role as OrgProjectRole | undefined) ?? role });
          }
        },
      },
    );
  };

  const handleRemove = () => {
    removeMember.mutate(
      { orgId, projectId, memberId: seat.orgMemberId },
      {
        onSuccess: (res) => {
          if (res.status === "organic") {
            setFeedback({ kind: "organic", detail: res.detail || "This member already has access on their own." });
          } else {
            setFeedback({ kind: "removed" });
          }
        },
      },
    );
  };

  return (
    <div className="flex items-center gap-2.5 px-4 py-2.5 border-t border-border/60">
      <div className="min-w-0 flex-1">
        <div className="text-sm truncate">{seat.email ?? "Unknown"}</div>
        {feedback?.kind === "organic" && (
          <div className="text-[11px] text-muted-foreground mt-0.5">{feedback.detail}</div>
        )}
        {feedback?.kind === "granted" && (
          <div className="text-[11px] text-emerald-600 dark:text-emerald-400 mt-0.5 capitalize">
            Granted as {feedback.role}
          </div>
        )}
        {feedback?.kind === "removed" && (
          <div className="text-[11px] text-muted-foreground mt-0.5">Access removed</div>
        )}
      </div>
      <Select value={role} onValueChange={(v) => setRole(v as OrgProjectRole)} disabled={busy}>
        <SelectTrigger className="h-7 w-24 text-xs flex-none" aria-label={`Role for ${seat.email ?? "member"}`}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="viewer">Viewer</SelectItem>
          <SelectItem value="editor">Editor</SelectItem>
          <SelectItem value="admin">Admin</SelectItem>
        </SelectContent>
      </Select>
      <Button size="sm" variant="outline" className="h-7 text-xs flex-none" onClick={handleSetRole} disabled={busy}>
        {setMemberRole.isPending && <Loader2 className="w-3 h-3 mr-1 animate-spin" />}
        Set role
      </Button>
      <Button
        size="sm"
        variant="ghost"
        className="h-7 w-7 p-0 flex-none text-muted-foreground hover:text-destructive"
        aria-label={`Remove ${seat.email ?? "member"}'s access`}
        title="Remove access"
        onClick={handleRemove}
        disabled={busy}
      >
        {removeMember.isPending ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <UserMinus className="w-3.5 h-3.5" />}
      </Button>
    </div>
  );
}

function LinkedProjectRow({
  orgId,
  project,
  seats,
}: {
  orgId: string;
  project: OrgLinkedProject;
  seats: OrgSeatUsage[];
}) {
  const [expanded, setExpanded] = useState(false);
  const activeSeats = seats.filter((s) => s.status === "active");

  return (
    <div className="border border-border rounded-xl overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-muted/40"
      >
        {expanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground flex-none" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-none" />
        )}
        <div className="min-w-0 flex-1">
          <div className="text-sm font-medium truncate">{project.name ?? "Untitled project"}</div>
          <div className="text-xs text-muted-foreground truncate">
            Owned by {project.ownerEmail ?? "unknown"} · Linked {fmtDate(project.linkedAt)}
          </div>
        </div>
        <Badge variant="outline" className="flex-none">
          {project.orgGrantedMemberCount} {project.orgGrantedMemberCount === 1 ? "seat" : "seats"} granted
        </Badge>
      </button>
      {expanded && (
        <div>
          {activeSeats.length === 0 ? (
            <div className="px-4 py-4 text-sm text-muted-foreground text-center border-t border-border/60">
              No active seats to manage yet — invite someone to the organization first.
            </div>
          ) : (
            activeSeats.map((seat) => (
              <MemberMatrixRow key={seat.orgMemberId} orgId={orgId} projectId={project.projectId} seat={seat} />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function OrgLinkedProjectsPanel({ orgId, seats }: { orgId: string; seats: OrgSeatUsage[] }) {
  const { data: projects, isLoading, isError } = useOrgLinkedProjects(orgId);

  return (
    <Card className="p-6">
      <div className="flex items-center gap-2">
        <FolderKanban className="w-4 h-4 text-muted-foreground" />
        <div className="text-[15px] font-semibold">Linked projects</div>
      </div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Projects an owner has linked to this organization. Grant or remove seat access below — a member with
        access on their own is left alone, with a note instead of an error.
      </div>

      <div className="mt-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-10">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : isError ? (
          <div className="text-sm text-muted-foreground text-center py-8">
            Couldn&apos;t load linked projects. Please try refreshing.
          </div>
        ) : !projects || projects.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-8">
            No projects are linked yet. Project owners can link from their project&apos;s settings.
          </div>
        ) : (
          <div className="space-y-2">
            {projects.map((p) => (
              <LinkedProjectRow key={p.projectId} orgId={orgId} project={p} seats={seats} />
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
