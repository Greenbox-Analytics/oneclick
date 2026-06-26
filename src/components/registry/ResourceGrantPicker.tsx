import { useState } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export interface ResourceItem {
  id: string;
  label: string;
  meta?: string; // e.g. "4 pages", "48 MB", "Worldwide · active"
  badge?: string; // e.g. "CONTRACT", "SPLIT", "LICENSE", "AGREEMENT"
}

export interface GrantSelection {
  project_file: string[];
  audio_file: string[];
  license: string[];
  agreement: string[];
  ownership_breakdown: boolean;
}

export interface ResourceGrantPickerProps {
  documents: ResourceItem[]; // -> project_file
  audio: ResourceItem[]; // -> audio_file
  licenses: ResourceItem[]; // -> license
  agreements: ResourceItem[]; // -> agreement
  value: GrantSelection;
  onChange: (next: GrantSelection) => void;
  disabled?: boolean; // when true (e.g. collaborator is Admin), render read-only/greyed
}

// Resource-type keys that map to a string[] field on GrantSelection.
type ResourceKey = "project_file" | "audio_file" | "license" | "agreement";

interface SectionDef {
  key: ResourceKey;
  title: string;
  items: ResourceItem[];
}

function toggleId(list: string[], id: string): string[] {
  return list.includes(id) ? list.filter((x) => x !== id) : [...list, id];
}

interface SectionProps {
  def: SectionDef;
  selected: string[];
  disabled: boolean;
  onToggle: (id: string) => void;
}

function ResourceSection({ def, selected, disabled, onToggle }: SectionProps) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const filtered = q ? def.items.filter((i) => i.label.toLowerCase().includes(q)) : def.items;
  const sharedCount = def.items.filter((i) => selected.includes(i.id)).length;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between px-1">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          {def.title}
        </span>
        <span className="text-[11px] font-medium text-muted-foreground">{sharedCount} shared</span>
      </div>

      <div className="relative">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={disabled}
          placeholder={`Search ${def.title.toLowerCase()}…`}
          className="h-8 pl-8 text-sm"
        />
      </div>

      <div className="flex max-h-[140px] flex-col gap-1 overflow-auto pr-1">
        {filtered.length === 0 ? (
          <p className="px-2 py-2 text-center text-xs text-muted-foreground">No matches</p>
        ) : (
          filtered.map((item) => {
            const checked = selected.includes(item.id);
            return (
              <label
                key={item.id}
                className={cn(
                  "flex items-center gap-2.5 rounded-md px-2 py-1.5 transition-colors",
                  disabled ? "cursor-not-allowed" : "cursor-pointer hover:bg-muted/50",
                  checked && "bg-emerald-50 dark:bg-emerald-950/40",
                )}
              >
                <Checkbox
                  checked={checked}
                  disabled={disabled}
                  onCheckedChange={() => onToggle(item.id)}
                  className={cn(
                    checked &&
                      "border-emerald-600 bg-emerald-600 text-white data-[state=checked]:bg-emerald-600 data-[state=checked]:text-white",
                  )}
                />
                <span className="flex min-w-0 flex-1 flex-col">
                  <span className="truncate text-sm font-medium text-foreground" title={item.label}>
                    {item.label}
                  </span>
                  {item.meta && (
                    <span className="truncate text-xs text-muted-foreground">{item.meta}</span>
                  )}
                </span>
                {item.badge && (
                  <Badge
                    variant="secondary"
                    className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
                  >
                    {item.badge}
                  </Badge>
                )}
              </label>
            );
          })
        )}
      </div>
    </div>
  );
}

export function ResourceGrantPicker({
  documents,
  audio,
  licenses,
  agreements,
  value,
  onChange,
  disabled = false,
}: ResourceGrantPickerProps) {
  const sections: SectionDef[] = [
    { key: "project_file", title: "Documents", items: documents },
    { key: "audio_file", title: "Audio", items: audio },
    { key: "license", title: "Licenses", items: licenses },
    { key: "agreement", title: "Agreements", items: agreements },
  ].filter((s) => s.items.length > 0);

  const handleToggle = (key: ResourceKey, id: string) => {
    if (disabled) return;
    onChange({ ...value, [key]: toggleId(value[key], id) });
  };

  const handleOwnershipChange = (checked: boolean) => {
    if (disabled) return;
    onChange({ ...value, ownership_breakdown: checked });
  };

  return (
    <div className={cn("space-y-5", disabled && "opacity-60")}>
      {/* Full ownership breakdown toggle */}
      <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-3 py-2.5">
        <div className="flex flex-col">
          <span className="text-sm font-medium text-foreground">Full ownership breakdown</span>
          <span className="text-xs text-muted-foreground">Let them see everyone's %</span>
        </div>
        <Switch
          checked={value.ownership_breakdown}
          disabled={disabled}
          onCheckedChange={handleOwnershipChange}
          className="data-[state=checked]:bg-emerald-600"
        />
      </div>

      {sections.map((section) => (
        <ResourceSection
          key={section.key}
          def={section}
          selected={value[section.key]}
          disabled={disabled}
          onToggle={(id) => handleToggle(section.key, id)}
        />
      ))}
    </div>
  );
}

export default ResourceGrantPicker;
