import { useMemo, useState } from "react";
import { Shield, Eye, Users, UserPlus, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { RegistryAvatar } from "./RegistryAvatar";
import CollaboratorAccessPanel from "./CollaboratorAccessPanel";
import { useWorkGrants, type GrantMatrixCollaborator } from "@/hooks/useRegistry";

function StatusBadge({ status }: { status: string }) {
  const confirmed = status === "confirmed" || status === "accepted";
  return (
    <Badge
      variant="outline"
      className={cn(
        "rounded px-1.5 py-0 text-[10px] font-medium capitalize",
        confirmed
          ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/5"
          : "border-muted-foreground/30 text-muted-foreground"
      )}
    >
      {status}
    </Badge>
  );
}

function AccessLevelBadge({ accessLevel }: { accessLevel: string }) {
  const admin = accessLevel === "admin";
  return (
    <Badge
      variant="secondary"
      className={cn(
        "inline-flex items-center gap-1 rounded px-1.5 py-0 text-[10px] font-semibold",
        admin && "bg-primary/15 text-primary"
      )}
    >
      {admin ? <Shield className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
      {admin ? "Admin" : "Viewer"}
    </Badge>
  );
}

/**
 * Owner/admin management surface for a single work's collaborators. Renders the
 * roster (non-revoked) as selectable rows plus an "Invite collaborator" action;
 * both the per-person editor and the invite flow open the shared
 * `CollaboratorAccessPanel`. Mounted in WorkEditor behind `access.can_manage`.
 */
export function PermissionsPanel({
  workId,
  projectName,
}: {
  workId: string;
  projectName?: string;
}) {
  const grantsQuery = useWorkGrants(workId);

  // Exclude revoked collaborators from the manageable roster.
  const collaborators: GrantMatrixCollaborator[] = useMemo(
    () =>
      (grantsQuery.data?.collaborators || []).filter(
        (c) => c.status !== "revoked"
      ),
    [grantsQuery.data?.collaborators]
  );

  // Which panel is open: invite, or edit for a specific collaborator id.
  const [inviteOpen, setInviteOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);

  const editing = useMemo(
    () => collaborators.find((c) => c.id === editingId) || null,
    [collaborators, editingId]
  );

  if (grantsQuery.isLoading) {
    return (
      <Card className="p-6 flex items-center justify-center text-sm text-muted-foreground">
        <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading people & access…
      </Card>
    );
  }

  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-muted text-muted-foreground flex items-center justify-center shrink-0">
            <Users className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold tracking-tight">People &amp; access</h3>
            <p className="text-xs text-muted-foreground mt-0.5">
              Control each collaborator's role and exactly what they can see.
            </p>
          </div>
        </div>
        <Button size="sm" variant="outline" onClick={() => setInviteOpen(true)}>
          <UserPlus className="w-3.5 h-3.5 mr-1.5" /> Invite collaborator
        </Button>
      </div>

      {/* Owner — read-only */}
      <div className="flex items-center gap-2.5 rounded-lg border border-border bg-muted/30 px-3 py-2.5 mb-2">
        <RegistryAvatar name="You" size={32} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold truncate">You · Owner</div>
          <div className="text-xs text-muted-foreground">Full control</div>
        </div>
        <Shield className="w-4 h-4 text-primary shrink-0" />
      </div>

      {collaborators.length === 0 ? (
        <div className="py-8 text-center text-sm text-muted-foreground border border-dashed rounded-lg">
          No collaborators yet —{" "}
          <button
            type="button"
            className="text-primary hover:underline"
            onClick={() => setInviteOpen(true)}
          >
            invite someone
          </button>{" "}
          to this work to manage their access here.
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          {collaborators.map((c) => (
            <button
              key={c.id}
              type="button"
              onClick={() => setEditingId(c.id)}
              className="flex items-center gap-2.5 rounded-lg border border-border px-3 py-2.5 text-left transition-colors hover:bg-muted/40 hover:border-primary/40"
            >
              <RegistryAvatar name={c.name || c.email} size={32} />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold truncate">
                  {c.name || c.email}
                </div>
                <div className="text-xs text-muted-foreground truncate capitalize">
                  {c.role || "Collaborator"}
                </div>
              </div>
              <div className="flex flex-col items-end gap-1 shrink-0">
                <AccessLevelBadge accessLevel={c.access_level} />
                <StatusBadge status={c.status} />
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Invite flow */}
      {inviteOpen && (
        <CollaboratorAccessPanel
          mode="invite"
          workId={workId}
          projectName={projectName}
          open={inviteOpen}
          onOpenChange={setInviteOpen}
        />
      )}

      {/* Per-person edit flow */}
      {editing && (
        <CollaboratorAccessPanel
          mode="edit"
          workId={workId}
          collaborator={editing}
          projectName={projectName}
          open={!!editing}
          onOpenChange={(open) => {
            if (!open) setEditingId(null);
          }}
        />
      )}
    </Card>
  );
}

export default PermissionsPanel;
