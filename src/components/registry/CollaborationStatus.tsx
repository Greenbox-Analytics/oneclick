import { useAuth } from "@/contexts/AuthContext";
import { type Collaborator, useSubmitForApproval, useConfirmStake, useDeclineInvitation, useResendInvitation, useRevokeCollaborator } from "@/hooks/useRegistry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Send, Loader2, CheckCircle, Clock, X, RefreshCw, Trash2 } from "lucide-react";

const COLLAB_STATUS_ICON: Record<string, typeof CheckCircle> = {
  confirmed: CheckCircle,
  invited: Clock,
  declined: X,
};

const COLLAB_STATUS_COLOR: Record<string, string> = {
  confirmed: "bg-green-100 text-green-800",
  invited: "bg-amber-100 text-amber-800",
  declined: "bg-gray-100 text-gray-800",
};

const COLLAB_STATUS_LABEL: Record<string, string> = {
  confirmed: "Accepted",
  invited: "Pending",
  declined: "Declined",
};

interface Props {
  workId: string;
  workStatus: string;
  collaborators: Collaborator[];
  isOwner: boolean;
}

export default function CollaborationStatus({ workId, workStatus, collaborators, isOwner }: Props) {
  const { user } = useAuth();
  const submitForApproval = useSubmitForApproval();
  const confirmStake = useConfirmStake();
  const declineInvitation = useDeclineInvitation();
  const resendInvitation = useResendInvitation();
  const revokeCollaborator = useRevokeCollaborator();

  const confirmed = collaborators.filter((c) => c.status === "confirmed").length;
  const total = collaborators.length;
  const pct = total > 0 ? (confirmed / total) * 100 : 0;

  const canSubmit = isOwner && workStatus === "draft" && total > 0;

  return (
    <Card className="mb-6">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Collaboration Status</CardTitle>
          {canSubmit && (
            <Button size="sm" onClick={() => submitForApproval.mutate(workId)}
              disabled={submitForApproval.isPending}>
              {submitForApproval.isPending ? (
                <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              ) : (
                <Send className="w-3 h-3 mr-1" />
              )}
              Submit for Approval
            </Button>
          )}
        </div>
        {total > 0 && (
          <div className="mt-2">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>{confirmed} of {total} confirmed</span>
              <span>{pct.toFixed(0)}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-green-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )}
      </CardHeader>
      <CardContent>
        {total === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-2">
            No collaborators invited yet. Add ownership stakes and invite stakeholders.
          </p>
        ) : (
          <div className="space-y-2">
            {collaborators.map((c) => {
              const Icon = COLLAB_STATUS_ICON[c.status] || Clock;
              const isExpired = c.status === "invited" && new Date(c.expires_at) < new Date();
              const isMyCollab = c.collaborator_user_id === user?.id;
              const canAct = isMyCollab && !isOwner && c.status === "invited";
              return (
                <div key={c.id} className="flex items-center justify-between p-2 rounded-lg border">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" />
                    <div>
                      <span className="text-sm font-medium">{c.name}</span>
                      <span className="text-xs text-muted-foreground ml-2">({c.role})</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">{c.email}</span>
                    <Badge className={COLLAB_STATUS_COLOR[c.status] || "bg-gray-100 text-gray-800"}>
                      {COLLAB_STATUS_LABEL[c.status] || c.status}
                    </Badge>
                    {canAct && (
                      <>
                        <Button size="sm" variant="outline" className="h-7 px-2 text-xs"
                          onClick={() => confirmStake.mutate(c.id)}
                          disabled={confirmStake.isPending}>
                          {confirmStake.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : "Accept"}
                        </Button>
                        <Button size="sm" variant="outline" className="h-7 px-2 text-xs text-destructive"
                          onClick={() => declineInvitation.mutate(c.id)}
                          disabled={declineInvitation.isPending}>
                          {declineInvitation.isPending ? <Loader2 className="w-3 h-3 animate-spin" /> : "Decline"}
                        </Button>
                      </>
                    )}
                    {isExpired && isOwner && (
                      <Button size="icon" variant="outline" className="h-7 w-7"
                        onClick={() => resendInvitation.mutate(c.id)}
                        disabled={resendInvitation.isPending}
                        title="Resend invitation">
                        {resendInvitation.isPending ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <RefreshCw className="w-3 h-3" />
                        )}
                      </Button>
                    )}
                    {isOwner && (
                      <Button size="icon" variant="ghost" className="h-7 w-7 text-destructive"
                        onClick={() => revokeCollaborator.mutate(c.id)}
                        disabled={revokeCollaborator.isPending}
                        title="Revoke collaborator">
                        {revokeCollaborator.isPending ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : (
                          <Trash2 className="w-3 h-3" />
                        )}
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
