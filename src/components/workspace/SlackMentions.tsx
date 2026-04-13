import { useNavigate } from "react-router-dom";
import { MessageSquare, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useSlackNotifications,
  useSlackUnreadCount,
  useMarkSlackRead,
  useMarkAllSlackRead,
} from "@/hooks/useSlackNotifications";

export function SlackMentions() {
  const navigate = useNavigate();
  const { data: notifications, isLoading } = useSlackNotifications();
  const unreadCount = useSlackUnreadCount();
  const markRead = useMarkSlackRead();
  const markAllRead = useMarkAllSlackRead();

  if (isLoading || !notifications?.length) return null;

  const handleClick = (n: (typeof notifications)[0]) => {
    if (!n.is_read) markRead.mutate(n.id);
    if (n.project_id) navigate(`/projects/${n.project_id}`);
  };

  const timeAgo = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-[#4A154B]" />
          <h4 className="font-medium text-sm">Slack Mentions</h4>
          {unreadCount > 0 && (
            <Badge variant="secondary" className="text-xs">
              {unreadCount}
            </Badge>
          )}
        </div>
        {unreadCount > 0 && (
          <Button variant="ghost" size="sm" onClick={() => markAllRead.mutate()}>
            <Check className="w-3 h-3 mr-1" /> Mark all read
          </Button>
        )}
      </div>
      <div className="space-y-2">
        {notifications.map((n) => (
          <div
            key={n.id}
            onClick={() => handleClick(n)}
            className={`p-3 rounded-lg border cursor-pointer transition-colors ${
              n.is_read
                ? "hover:bg-muted/50"
                : "bg-primary/5 border-primary/20 hover:bg-primary/10"
            }`}
          >
            <div className="flex items-start gap-2">
              {!n.is_read && (
                <div className="w-2 h-2 rounded-full bg-primary mt-1.5 shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{n.sender_name}</span>
                  {n.project && (
                    <Badge variant="outline" className="text-[10px]">
                      {n.project.name}
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-muted-foreground mt-0.5 line-clamp-2">
                  {n.message_text}
                </p>
                <span className="text-xs text-muted-foreground mt-1 block">
                  {timeAgo(n.created_at)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
