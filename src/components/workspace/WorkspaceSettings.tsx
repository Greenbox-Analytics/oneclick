import { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Loader2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useWorkspaceSettings } from "@/hooks/useWorkspaceSettings";

const COMMON_TIMEZONES = [
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Los_Angeles",
  "America/Anchorage",
  "Pacific/Honolulu",
  "America/Toronto",
  "America/Vancouver",
  "Europe/London",
  "Europe/Paris",
  "Europe/Berlin",
  "Europe/Moscow",
  "Asia/Dubai",
  "Asia/Kolkata",
  "Asia/Shanghai",
  "Asia/Tokyo",
  "Australia/Sydney",
  "Pacific/Auckland",
  "Africa/Nairobi",
  "Africa/Lagos",
  "Africa/Cairo",
  "Africa/Johannesburg",
];

export function WorkspaceSettings() {
  const { settings, isLoading, updateSettings } = useWorkspaceSettings();

  const allTimezones = useMemo(() => {
    try {
      const all = Intl.supportedValuesOf("timeZone");
      // Put common ones first, then the rest
      const commonSet = new Set(COMMON_TIMEZONES);
      const rest = all.filter((tz) => !commonSet.has(tz));
      return { common: COMMON_TIMEZONES, rest };
    } catch {
      return { common: COMMON_TIMEZONES, rest: [] };
    }
  }, []);

  const systemTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

  if (isLoading || !settings) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Board Settings</CardTitle>
          <CardDescription>Configure how your project boards behave</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label>Board Period</Label>
            <Select
              value={settings.board_period}
              onValueChange={(value) => updateSettings({ board_period: value as any })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="weekly">Weekly</SelectItem>
                <SelectItem value="biweekly">Bi-weekly</SelectItem>
                <SelectItem value="monthly">Monthly</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {settings.board_period === "custom" && (
            <div className="space-y-2">
              <Label htmlFor="custom-days">Custom Period (days)</Label>
              <Input
                id="custom-days"
                type="number"
                min={1}
                max={365}
                value={settings.custom_period_days || ""}
                onChange={(e) =>
                  updateSettings({ custom_period_days: parseInt(e.target.value) || undefined })
                }
                placeholder="e.g. 14"
              />
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Calendar</CardTitle>
          <CardDescription>Configure default calendar display</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-2">
            <Label>Default View</Label>
            <Select
              value={settings.calendar_view || "month"}
              onValueChange={(value) => updateSettings({ calendar_view: value as any })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="day">Day</SelectItem>
                <SelectItem value="week">Week</SelectItem>
                <SelectItem value="month">Month</SelectItem>
                <SelectItem value="year">Year</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Display Preferences</CardTitle>
          <CardDescription>Customize how information is displayed</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label>24-hour Time</Label>
              <p className="text-sm text-muted-foreground">Use 24-hour format instead of AM/PM</p>
            </div>
            <Switch
              checked={settings.use_24h_time}
              onCheckedChange={(checked) => updateSettings({ use_24h_time: checked })}
            />
          </div>

          <div className="space-y-2">
            <Label>Timezone</Label>
            <Select
              value={settings.timezone || "system"}
              onValueChange={(value) => updateSettings({ timezone: value === "system" ? undefined : value } as any)}
            >
              <SelectTrigger>
                <SelectValue placeholder={`System (${systemTimezone})`} />
              </SelectTrigger>
              <SelectContent className="max-h-[300px]">
                <SelectItem value="system">System ({systemTimezone})</SelectItem>
                {allTimezones.common.map((tz) => (
                  <SelectItem key={tz} value={tz}>
                    {tz.replace(/_/g, " ")}
                  </SelectItem>
                ))}
                {allTimezones.rest.map((tz) => (
                  <SelectItem key={tz} value={tz}>
                    {tz.replace(/_/g, " ")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Leave as "System" to use your device timezone
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
