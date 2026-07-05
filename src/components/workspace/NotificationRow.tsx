import { useNavigate } from "react-router-dom";
import { ExternalLink } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useMarkNotificationRead, type RegistryNotification } from "@/hooks/useRegistryNotifications";
import { useAcceptTeamInvite, useDeclineTeamInvite } from "@/hooks/useTeams";

export const TYPE_COLORS: Record<string, string> = {
  invitation: "bg-blue-100 text-blue-800",
  confirmation: "bg-green-100 text-green-800",
  dispute: "bg-red-100 text-red-800",
  status_change: "bg-purple-100 text-purple-800",
  team_invite: "bg-indigo-100 text-indigo-700",
};

export function NotificationRow({ n }: { n: RegistryNotification }) {
  const navigate = useNavigate();
  const markRead = useMarkNotificationRead();
  const accept = useAcceptTeamInvite();
  const decline = useDeclineTeamInvite();

  const handleClick = () => {
    // team_invite: Accept/Decline are gated on !n.read, so a row click must NOT mark it read.
    if (!n.read && n.type !== "team_invite") markRead.mutate(n.id);
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
              {n.type.replace("_", " ")}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-1 ml-4">{n.message}</p>
          {n.type === "team_invite" && !n.read && (
            <div className="mt-2 ml-4 flex gap-2" onClick={(e) => e.stopPropagation()}>
              <Button
                size="sm"
                disabled={accept.isPending || decline.isPending}
                onClick={() => {
                  const token = n.metadata?.token;
                  if (!token) return;
                  accept.mutate(String(token), { onSuccess: () => markRead.mutate(n.id) });
                }}
              >
                Accept
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={accept.isPending || decline.isPending}
                onClick={() => {
                  const token = n.metadata?.token;
                  if (!token) return;
                  decline.mutate(String(token), { onSuccess: () => markRead.mutate(n.id) });
                }}
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
