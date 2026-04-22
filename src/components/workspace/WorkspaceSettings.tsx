import { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Loader2, Check, ChevronsUpDown } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
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

const REGION_LABELS: Record<string, string> = {
  America: "Americas",
  Europe: "Europe",
  Asia: "Asia",
  Africa: "Africa",
  Pacific: "Pacific",
  Australia: "Australia",
  Indian: "Indian Ocean",
  Atlantic: "Atlantic",
  Arctic: "Arctic",
  Antarctica: "Antarctica",
};

export function WorkspaceSettings() {
  const { settings, isLoading, updateSettings } = useWorkspaceSettings();
  const [tzOpen, setTzOpen] = useState(false);
  const [regionFilter, setRegionFilter] = useState<string | null>(null);

  const { regions, filteredTimezones } = useMemo(() => {
    let all: string[];
    try {
      all = Intl.supportedValuesOf("timeZone");
    } catch {
      all = COMMON_TIMEZONES;
    }

    // Extract unique regions
    const regionSet = new Set<string>();
    for (const tz of all) {
      const region = tz.split("/")[0];
      if (REGION_LABELS[region]) regionSet.add(region);
    }
    const regions = Array.from(regionSet).sort();

    // Filter by region if one is selected
    const filtered = regionFilter
      ? all.filter((tz) => tz.startsWith(regionFilter + "/"))
      : all;

    // When no region filter, group common first then rest
    if (!regionFilter) {
      const commonSet = new Set(COMMON_TIMEZONES);
      const common = COMMON_TIMEZONES.filter((tz) => all.includes(tz));
      const rest = filtered.filter((tz) => !commonSet.has(tz));
      return { regions, filteredTimezones: { common, rest } };
    }

    return { regions, filteredTimezones: { common: [], rest: filtered } };
  }, [regionFilter]);

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
              onValueChange={(value) => updateSettings({ board_period: value as "weekly" | "biweekly" | "monthly" | "custom" })}
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
              onValueChange={(value) => updateSettings({ calendar_view: value as "day" | "week" | "month" | "year" })}
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
            <Popover open={tzOpen} onOpenChange={setTzOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={tzOpen}
                  className="w-full justify-between font-normal"
                >
                  {settings.timezone
                    ? settings.timezone.replace(/_/g, " ")
                    : `System (${systemTimezone})`}
                  <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
                <Command>
                  <CommandInput placeholder="Search timezone..." />
                  <div className="flex flex-wrap gap-1 px-2 py-2 border-b">
                    <Button
                      variant={regionFilter === null ? "secondary" : "ghost"}
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={() => setRegionFilter(null)}
                    >
                      All
                    </Button>
                    {regions.map((region) => (
                      <Button
                        key={region}
                        variant={regionFilter === region ? "secondary" : "ghost"}
                        size="sm"
                        className="h-6 px-2 text-xs"
                        onClick={() => setRegionFilter(regionFilter === region ? null : region)}
                      >
                        {REGION_LABELS[region] || region}
                      </Button>
                    ))}
                  </div>
                  <CommandList>
                    <CommandEmpty>No timezone found.</CommandEmpty>
                    {!regionFilter && (
                      <CommandGroup heading="System">
                        <CommandItem
                          value={`system ${systemTimezone}`}
                          onSelect={() => {
                            updateSettings({ timezone: undefined });
                            setTzOpen(false);
                          }}
                        >
                          <Check className={cn("mr-2 h-4 w-4", !settings.timezone ? "opacity-100" : "opacity-0")} />
                          System ({systemTimezone})
                        </CommandItem>
                      </CommandGroup>
                    )}
                    {filteredTimezones.common.length > 0 && (
                      <CommandGroup heading="Common">
                        {filteredTimezones.common.map((tz) => (
                          <CommandItem
                            key={tz}
                            value={tz}
                            onSelect={() => {
                              updateSettings({ timezone: tz });
                              setTzOpen(false);
                            }}
                          >
                            <Check className={cn("mr-2 h-4 w-4", settings.timezone === tz ? "opacity-100" : "opacity-0")} />
                            {tz.replace(/_/g, " ")}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    )}
                    {filteredTimezones.rest.length > 0 && (
                      <CommandGroup heading={regionFilter ? (REGION_LABELS[regionFilter] || regionFilter) : "All Timezones"}>
                        {filteredTimezones.rest.map((tz) => (
                          <CommandItem
                            key={tz}
                            value={tz}
                            onSelect={() => {
                              updateSettings({ timezone: tz });
                              setTzOpen(false);
                            }}
                          >
                            <Check className={cn("mr-2 h-4 w-4", settings.timezone === tz ? "opacity-100" : "opacity-0")} />
                            {tz.replace(/_/g, " ")}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    )}
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
            <p className="text-xs text-muted-foreground">
              Leave as "System" to use your device timezone
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
