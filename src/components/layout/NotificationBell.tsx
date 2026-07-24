import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bell, CheckCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { AutoHideTooltip } from "@/components/layout/AutoHideTooltip";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { NotificationRow } from "@/components/workspace/NotificationRow";
import { useRegistryNotifications, useMarkAllRead } from "@/hooks/useRegistryNotifications";

export function NotificationBell() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const { data: notifications } = useRegistryNotifications();
  const markAllRead = useMarkAllRead();

  const items = notifications || [];
  const unread = items.filter((n) => !n.read).length;
  const recent = items.slice(0, 10);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      {/* Auto-fading hover tooltip replaces the native `title` attr (the two
          would double up). Suppressed while the panel itself is open. */}
      <AutoHideTooltip
        label={unread > 0 ? `Notifications (${unread} unread)` : "Notifications"}
        disabled={open}
      >
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon" className="relative text-muted-foreground hover:text-foreground"
                  aria-label={unread > 0 ? `Notifications, ${unread} unread` : "Notifications"}>
            <Bell className="w-4 h-4" />
            {unread > 0 && (
              <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 px-1 rounded-full bg-destructive
                               text-destructive-foreground text-[10px] font-bold flex items-center justify-center">
                {unread > 9 ? "9+" : unread}
              </span>
            )}
          </Button>
        </PopoverTrigger>
      </AutoHideTooltip>
      <PopoverContent align="end" className="w-80 p-0">
        <div className="flex items-center justify-between px-3 py-2 border-b">
          <span className="text-sm font-semibold">Notifications</span>
          {unread > 0 && (
            <Button variant="ghost" size="sm" className="h-7 px-2" onClick={() => markAllRead.mutate()}>
              <CheckCheck className="w-3.5 h-3.5 mr-1" /> Mark all read
            </Button>
          )}
        </div>
        <div className="max-h-96 overflow-y-auto p-2 space-y-2">
          {recent.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
              You're all caught up.
            </div>
          ) : (
            // Wrapper closes the popover when a row click navigates. Accept/Decline
            // stopPropagation, so acting on an invite keeps the popover open to show the result.
            recent.map((n) => (
              <div key={n.id} onClick={() => setOpen(false)}>
                <NotificationRow n={n} />
              </div>
            ))
          )}
        </div>
        <div className="border-t px-3 py-2">
          <Button variant="ghost" size="sm" className="w-full justify-center"
                  onClick={() => { setOpen(false); navigate("/notifications"); }}>
            See all
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}
