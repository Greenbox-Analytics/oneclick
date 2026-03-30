import { type Collaborator, useSubmitForApproval, useResendInvitation, useRevokeCollaborator } from "@/hooks/useRegistry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Send, Loader2, CheckCircle, AlertTriangle, Clock, RefreshCw, Trash2 } from "lucide-react";

const COLLAB_STATUS_ICON: Record<string, typeof CheckCircle> = {
  confirmed: CheckCircle,
  disputed: AlertTriangle,
  invited: Clock,
};

const COLLAB_STATUS_COLOR: Record<string, string> = {
  confirmed: "bg-green-100 text-green-800",
  disputed: "bg-red-100 text-red-800",
  invited: "bg-amber-100 text-amber-800",
};

interface Props {
  workId: string;
  workStatus: string;
  collaborators: Collaborator[];
  isOwner: boolean;
}

export default function CollaborationStatus({ workId, workStatus, collaborators, isOwner }: Props) {
  const submitForApproval = useSubmitForApproval();
  const resendInvitation = useResendInvitation();
  const revokeCollaborator = useRevokeCollaborator();

  const confirmed = collaborators.filter((c) => c.status === "confirmed").length;
  const total = collaborators.length;
  const pct = total > 0 ? (confirmed / total) * 100 : 0;

  const canSubmit = isOwner && (workStatus === "draft" || workStatus === "disputed") && total > 0;

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
                    <Badge className={COLLAB_STATUS_COLOR[c.status] || ""}>{c.status}</Badge>
                    {isExpired && (
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
            {collaborators.some((c) => c.status === "disputed" && c.dispute_reason) && (
              <div className="mt-2 p-3 bg-red-50 rounded-lg border border-red-200">
                <p className="text-sm font-medium text-red-800 mb-1">Dispute Reasons:</p>
                {collaborators.filter((c) => c.status === "disputed" && c.dispute_reason).map((c) => (
                  <p key={c.id} className="text-xs text-red-700">
                    <strong>{c.name}:</strong> {c.dispute_reason}
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
