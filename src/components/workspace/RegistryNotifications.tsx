import { useNavigate } from "react-router-dom";
import {
  useRegistryNotifications,
  useMarkNotificationRead,
  useMarkAllRead,
  type RegistryNotification,
} from "@/hooks/useRegistryNotifications";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, CheckCheck, Shield, ExternalLink } from "lucide-react";

const TYPE_COLORS: Record<string, string> = {
  invitation: "bg-blue-100 text-blue-800",
  confirmation: "bg-green-100 text-green-800",
  dispute: "bg-red-100 text-red-800",
  status_change: "bg-purple-100 text-purple-800",
};

export function RegistryNotifications() {
  const navigate = useNavigate();
  const { data: notifications, isLoading } = useRegistryNotifications();
  const markRead = useMarkNotificationRead();
  const markAllRead = useMarkAllRead();

  const handleClick = (n: RegistryNotification) => {
    if (!n.read) markRead.mutate(n.id);
    if (n.work_id) navigate(`/tools/registry/${n.work_id}`);
  };

  const unreadCount = (notifications || []).filter((n) => !n.read).length;

  if (isLoading) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Bell className="w-8 h-8 mx-auto mb-3 animate-pulse" />
        <p>Loading notifications...</p>
      </div>
    );
  }

  if (!notifications?.length) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Bell className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <h3 className="text-lg font-semibold mb-2">No notifications yet</h3>
        <p>Collaboration updates from the Rights Registry will appear here</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Shield className="w-5 h-5 text-primary" />
          Rights Registry Notifications
          {unreadCount > 0 && (
            <Badge variant="destructive" className="ml-1">{unreadCount}</Badge>
          )}
        </h3>
        {unreadCount > 0 && (
          <Button variant="ghost" size="sm" onClick={() => markAllRead.mutate()}>
            <CheckCheck className="w-4 h-4 mr-1" /> Mark all read
          </Button>
        )}
      </div>
      <div className="space-y-2">
        {notifications.map((n) => (
          <div
            key={n.id}
            className={`p-3 rounded-lg border cursor-pointer transition-colors hover:bg-muted/50 ${
              !n.read ? "bg-primary/5 border-primary/20" : ""
            }`}
            onClick={() => handleClick(n)}
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
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
                <span>{new Date(n.created_at).toLocaleDateString()}</span>
                {n.work_id && <ExternalLink className="w-3 h-3" />}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
