import { useRegistryNotifications, useMarkAllRead } from "@/hooks/useRegistryNotifications";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, CheckCheck, Shield } from "lucide-react";
import { SlackMentions } from "./SlackMentions";
import { NotificationRow } from "./NotificationRow";

export function RegistryNotifications() {
  const { data: notifications, isLoading } = useRegistryNotifications();
  const markAllRead = useMarkAllRead();

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
        <p>Collaboration updates from the Metadata Registry will appear here</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <SlackMentions />
      <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Shield className="w-5 h-5 text-primary" />
          Metadata Registry Notifications
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
          <NotificationRow key={n.id} n={n} />
        ))}
      </div>
      </div>
    </div>
  );
}
