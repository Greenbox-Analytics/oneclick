import { useNavigate } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useMarkNotificationRead, type RegistryNotification } from "@/hooks/useRegistryNotifications";
import { useAcceptOrgInvite, useDeclineOrgInvite } from "@/hooks/useOrgs";

export const TYPE_COLORS: Record<string, string> = {
  invitation: "bg-blue-100 text-blue-800",
  confirmation: "bg-green-100 text-green-800",
  dispute: "bg-red-100 text-red-800",
  status_change: "bg-purple-100 text-purple-800",
};

export function NotificationRow({ n }: { n: RegistryNotification }) {
  const navigate = useNavigate();
  const markRead = useMarkNotificationRead();
  const accept = useAcceptOrgInvite();
  const decline = useDeclineOrgInvite();

  // An org invite is type='invitation' + entity_type='org' — the only
  // actionable notification. Registry's own 'invitation' rows carry
  // entity_type 'work'/null and stay button-less.
  const isOrgInvite = n.type === "invitation" && n.entity_type === "org";
  const isInvite = isOrgInvite;

  // The org hooks stay silent by design (OrgInviteClaim owns that page's copy),
  // so the feedback for this row belongs here.
  const runInvite = (mutation: typeof accept, successMsg: string) => () => {
    const token = n.metadata?.token;
    if (!token) return;
    mutation.mutate(String(token), {
      onSuccess: () => {
        markRead.mutate(n.id);
        toast.success(successMsg);
      },
      onError: (e: Error) => toast.error(e.message),
    });
  };

  const handleClick = () => {
    // Invites: Accept/Decline are gated on !n.read, so a row click must NOT mark it read.
    if (!n.read && !isInvite) markRead.mutate(n.id);
    if (n.work_id) navigate(`/tools/registry/${n.work_id}`);
  };

  return (
    <div
      className={`p-3 rounded-lg border cursor-pointer transition-colors hover:bg-muted/50 ${
        !n.read ? "bg-primary/5 border-primary/20" : ""
      }`}
      onClick={handleClick}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            {!n.read && <div className="w-2 h-2 rounded-full bg-primary shrink-0" />}
            <span className="text-sm font-medium">{n.title}</span>
            <Badge className={TYPE_COLORS[n.type] || "bg-gray-100 text-gray-800"}>
              {isOrgInvite ? "org invite" : n.type.replace("_", " ")}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-1 ml-4">{n.message}</p>
          {isInvite && !n.read && (
            <div className="mt-2 ml-4 flex gap-2" onClick={(e) => e.stopPropagation()}>
              <Button
                size="sm"
                disabled={accept.isPending || decline.isPending}
                onClick={runInvite(accept, "Invite accepted")}
              >
                Accept
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={accept.isPending || decline.isPending}
                onClick={runInvite(decline, "Invitation declined")}
              >
                Decline
              </Button>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
          <span>{new Date(n.created_at).toLocaleDateString()}</span>
          {n.work_id && <ExternalLink className="w-3 h-3" />}
        </div>
      </div>
    </div>
  );
}
