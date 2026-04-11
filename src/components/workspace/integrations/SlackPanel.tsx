import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Hash, Lock } from "lucide-react";
import { useSlackChannels, useSlackSettings } from "@/hooks/useSlackSettings";

const EVENT_TYPES = [
  {
    key: "task_created",
    label: "Task Created",
    description: "When a new task is created on the board",
  },
  {
    key: "task_updated",
    label: "Task Updated",
    description: "When a task is modified or moved",
  },
  {
    key: "task_completed",
    label: "Task Completed",
    description: "When a task is marked as done",
  },
  {
    key: "contract_uploaded",
    label: "Contract Uploaded",
    description: "When a new contract is uploaded to a project",
  },
  {
    key: "royalty_calculated",
    label: "Royalty Calculated",
    description: "When a royalty calculation is completed",
  },
];

interface SlackPanelProps {
  onClose?: () => void;
}

export function SlackPanel({ onClose }: SlackPanelProps) {
  const { data: channels, isLoading: channelsLoading } = useSlackChannels();
  const { settings, isLoading: settingsLoading, updateSetting } = useSlackSettings();

  const getSettingForEvent = (eventType: string) =>
    settings.find((s) => s.event_type === eventType);

  const handleToggle = (eventType: string, enabled: boolean) => {
    const existing = getSettingForEvent(eventType);
    updateSetting({
      event_type: eventType,
      enabled,
      channel_id: existing?.channel_id || undefined,
    });
  };

  const handleChannelChange = (eventType: string, channelId: string) => {
    const existing = getSettingForEvent(eventType);
    updateSetting({
      event_type: eventType,
      enabled: existing?.enabled ?? true,
      channel_id: channelId,
    });
  };

  const isLoading = channelsLoading || settingsLoading;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div>
          <CardTitle className="text-base">Slack Notifications</CardTitle>
          <p className="text-sm text-muted-foreground mt-1">
            Choose which events send notifications and to which channels
          </p>
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        )}
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {EVENT_TYPES.map((event) => {
              const setting = getSettingForEvent(event.key);
              const isEnabled = setting?.enabled ?? false;

              return (
                <div
                  key={event.key}
                  className="flex items-center justify-between gap-4 rounded-lg border p-3"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{event.label}</span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {event.description}
                    </p>
                    {isEnabled && (
                      <div className="mt-2">
                        <Select
                          value={setting?.channel_id || ""}
                          onValueChange={(val) =>
                            handleChannelChange(event.key, val)
                          }
                        >
                          <SelectTrigger className="h-8 text-xs w-56">
                            <SelectValue placeholder="Select channel..." />
                          </SelectTrigger>
                          <SelectContent>
                            {(channels || []).map((ch) => (
                              <SelectItem key={ch.id} value={ch.id}>
                                <div className="flex items-center gap-1.5">
                                  {ch.is_private ? (
                                    <Lock className="w-3 h-3" />
                                  ) : (
                                    <Hash className="w-3 h-3" />
                                  )}
                                  {ch.name}
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                  </div>
                  <Switch
                    checked={isEnabled}
                    onCheckedChange={(checked) =>
                      handleToggle(event.key, checked)
                    }
                  />
                </div>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
