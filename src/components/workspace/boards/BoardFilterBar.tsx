import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { MultiSelectCombobox } from "./MultiSelectCombobox";
import { useTeamMembers } from "@/hooks/useTeams";
import type { BoardTask } from "@/types/integrations";

export interface BoardFilterValue {
  createdBy: string[];
  artist: string[];
}

interface BoardFilterBarProps {
  teamId: string | null;
  tasks: BoardTask[];
  value: BoardFilterValue;
  onChange: (value: BoardFilterValue) => void;
}

export function BoardFilterBar({ teamId, tasks, value, onChange }: BoardFilterBarProps) {
  // Team members power the "Created by" filter (only relevant on team boards).
  const { data: members } = useTeamMembers(teamId ?? undefined);

  const memberOptions = useMemo(
    () => (members ?? []).map((m) => ({ id: m.user_id, label: m.full_name || m.user_id })),
    [members],
  );

  // "By artist" options are the distinct artists present on the board, keyed by
  // artist_id and displaying the snapshot name from the enriched task payload.
  const artistOptions = useMemo(() => {
    const byId = new Map<string, string>();
    for (const t of tasks) {
      for (const a of t.artists ?? []) {
        if (!byId.has(a.id)) byId.set(a.id, a.name);
      }
    }
    return Array.from(byId, ([id, label]) => ({ id, label }));
  }, [tasks]);

  const activeCount = value.createdBy.length + value.artist.length;

  return (
    <div className="flex flex-wrap items-start gap-3 mb-4">
      {teamId && (
        <div className="w-56">
          <MultiSelectCombobox
            options={memberOptions}
            selected={value.createdBy}
            onChange={(createdBy) => onChange({ ...value, createdBy })}
            placeholder="Created by"
            aria-label="Created by"
          />
        </div>
      )}

      <div className="w-56">
        <MultiSelectCombobox
          options={artistOptions}
          selected={value.artist}
          onChange={(artist) => onChange({ ...value, artist })}
          placeholder="By artist"
          aria-label="By artist"
        />
      </div>

      {activeCount > 0 && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground pt-1.5">
          <span>
            {activeCount} filter{activeCount === 1 ? "" : "s"} active
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 px-2"
            onClick={() => onChange({ createdBy: [], artist: [] })}
          >
            Clear
          </Button>
        </div>
      )}
    </div>
  );
}
