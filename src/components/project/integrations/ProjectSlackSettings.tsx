import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { useSlackChannels } from "@/hooks/useSlackSettings";
import {
  useProjectSlackChannel,
  useProjectNotificationSettings,
} from "@/hooks/useProjectIntegrations";

const EVENT_TYPES = [
  { key: "task_created", label: "Task Created" },
  { key: "task_updated", label: "Task Updated" },
  { key: "task_completed", label: "Task Completed" },
  { key: "contract_uploaded", label: "Contract Uploaded" },
  { key: "contract_deleted", label: "Contract Deleted" },
  { key: "royalty_calculated", label: "Royalty Calculated" },
];

interface ProjectSlackSettingsProps {
  projectId: string;
}

export function ProjectSlackSettings({ projectId }: ProjectSlackSettingsProps) {
  const { data: channels, isLoading: channelsLoading } = useSlackChannels();
  const { channelId, isLoading: channelLoading, updateChannel } = useProjectSlackChannel(projectId);
  const { isEventEnabled, toggleEvent, isLoading: settingsLoading } = useProjectNotificationSettings(projectId);

  const isLoading = channelsLoading || channelLoading || settingsLoading;

  if (isLoading) {
    return <Skeleton className="h-48 w-full" />;
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Slack Notifications</CardTitle>
        <p className="text-sm text-muted-foreground">
          Link a Slack channel to receive notifications for this project
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="text-sm font-medium mb-1.5 block">Channel</label>
          <Select
            value={channelId || ""}
            onValueChange={(val) => updateChannel({ channelId: val || null })}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select a channel..." />
            </SelectTrigger>
            <SelectContent>
              {(channels || []).map((ch) => (
                <SelectItem key={ch.id} value={ch.id}>
                  <div className="flex items-center gap-1.5">
                    {ch.is_private ? <Lock className="w-3 h-3" /> : <Hash className="w-3 h-3" />}
                    {ch.name}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {channelId && (
          <div className="space-y-2">
            <label className="text-sm font-medium">Notify on</label>
            {EVENT_TYPES.map((event) => (
              <div key={event.key} className="flex items-center justify-between py-1">
                <span className="text-sm">{event.label}</span>
                <Switch
                  checked={isEventEnabled(event.key)}
                  onCheckedChange={(checked) =>
                    toggleEvent({ eventType: event.key, enabled: checked })
                  }
                />
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
