import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar } from "@/components/ui/calendar";
import { CalendarIcon, X, ArrowUpRight } from "lucide-react";
import { format } from "date-fns";
import { cn } from "@/lib/utils";
import { ColorPicker } from "./ColorPicker";
import { MultiSelectCombobox } from "./MultiSelectCombobox";
import { useTeamMembers } from "@/hooks/useTeams";
import { useAddAssignee, useRemoveAssignee } from "@/hooks/useTaskAssignees";
import { useAuth } from "@/contexts/AuthContext";
import type { BoardColumn, BoardTaskDetail, ParentTaskWithChildren } from "@/types/integrations";

interface TaskFieldsProps {
  task: BoardTaskDetail;
  description: string;
  setDescription: (v: string) => void;
  priority: string;
  setPriority: (v: string) => void;
  startDate: Date | undefined;
  setStartDate: (v: Date | undefined) => void;
  dueDate: Date | undefined;
  setDueDate: (v: Date | undefined) => void;
  color: string;
  setColor: (v: string) => void;
  // Team context from the board switcher (get_task_detail returns board_id only).
  // Set → team board (member multi-select); null/undefined → personal board (self-assign only).
  teamId?: string | null;
  statusColumnId: string;
  setStatusColumnId: (v: string) => void;
  columns: BoardColumn[];
  parents: ParentTaskWithChildren[];
  artists: { id: string; name: string }[];
  projects: { id: string; name: string }[];
  contracts: { id: string; file_name: string }[];
  selectedArtistIds: string[];
  setSelectedArtistIds: (ids: string[]) => void;
  selectedProjectIds: string[];
  setSelectedProjectIds: (ids: string[]) => void;
  selectedContractIds: string[];
  setSelectedContractIds: (ids: string[]) => void;
  saveField: (field: string, value: unknown) => void;
  // Batches multiple fields into a single updateTask mutation so the backend
  // merge sees <kind>_ids and <kind>_labels together. No-op in create mode.
  saveFields: (fields: Record<string, unknown>) => void;
  onNavigateToTask?: (taskId: string) => void;
}

export function TaskFields({
  task,
  description,
  setDescription,
  priority,
  setPriority,
  startDate,
  setStartDate,
  dueDate,
  setDueDate,
  color,
  setColor,
  teamId,
  statusColumnId,
  setStatusColumnId,
  columns,
  parents,
  artists,
  projects,
  contracts,
  selectedArtistIds,
  setSelectedArtistIds,
  selectedProjectIds,
  setSelectedProjectIds,
  selectedContractIds,
  setSelectedContractIds,
  saveField,
  saveFields,
  onNavigateToTask,
}: TaskFieldsProps) {
  const { user } = useAuth();
  // Only fetches when teamId is set (hook is enabled-gated on teamId).
  const { data: members } = useTeamMembers(teamId ?? undefined);
  const addAssignee = useAddAssignee();
  const removeAssignee = useRemoveAssignee();

  // Assignment mutations need a real task id — in create mode (task.id is
  // empty) the section is hidden; assignment happens after creation.
  const canAssign = !!task.id;
  const assignPending = addAssignee.isPending || removeAssignee.isPending;
  const assigneeIds = task.assignees?.map((a) => a.user_id) ?? [];

  const handleAssigneesChange = (ids: string[]) => {
    ids
      .filter((id) => !assigneeIds.includes(id))
      .forEach((userId) => addAssignee.mutate({ taskId: task.id, userId }));
    assigneeIds
      .filter((id) => !ids.includes(id))
      .forEach((userId) => removeAssignee.mutate({ taskId: task.id, userId }));
  };

  // For each linked kind, split the server's linked set into:
  //   - options: the viewer's own list + synthetic options for can_open linked
  //     ids not already in the own list (e.g. a project the viewer is a member
  //     of). Non-can_open ids are intentionally excluded so they can't be
  //     toggled in the dropdown.
  //   - readonly: non-can_open linked entities, shown as plain (unremovable)
  //     chips. The backend merge preserves these server-side.
  // `nameFor` resolves a selected id's label from the option set (own +
  // synthetic) so the write can snapshot <kind>_labels.
  const buildLinkedField = (
    ownOptions: { id: string; label: string }[],
    linked: { id: string; name: string; can_open: boolean }[]
  ) => {
    const ownIds = new Set(ownOptions.map((o) => o.id));
    const synthetic = linked
      .filter((l) => l.can_open && !ownIds.has(l.id))
      .map((l) => ({ id: l.id, label: l.name }));
    const options = [...ownOptions, ...synthetic];
    const nameByIdEntries = new Map(options.map((o) => [o.id, o.label]));
    return {
      options,
      readonly: linked.filter((l) => !l.can_open),
      nameFor: (id: string) =>
        nameByIdEntries.get(id) ??
        linked.find((l) => l.id === id)?.name ??
        id,
    };
  };

  const artistField = buildLinkedField(
    artists.map((a) => ({ id: a.id, label: a.name })),
    task.artists ?? []
  );
  const projectField = buildLinkedField(
    projects.map((p) => ({ id: p.id, label: p.name })),
    task.projects ?? []
  );
  const documentField = buildLinkedField(
    contracts.map((c) => ({ id: c.id, label: c.file_name })),
    task.documents ?? []
  );

  const labelsFor = (ids: string[], nameFor: (id: string) => string) =>
    Object.fromEntries(ids.map((id) => [id, nameFor(id)]));

  return (
    <>
      {/* Status (for parent tasks) */}
      {task.is_parent && columns.length > 0 && (
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Status
          </Label>
          <Select
            value={statusColumnId || "__backlog__"}
            onValueChange={(val) => {
              if (val === "__backlog__") {
                setStatusColumnId("");
                saveField("column_id", "");
              } else {
                setStatusColumnId(val);
                saveField("column_id", val);
              }
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Set status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__backlog__">
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: "#9ca3af" }} />
                  Backlog
                </div>
              </SelectItem>
              {columns
                .filter((c) => c.title.toLowerCase() !== "review")
                .sort((a, b) => a.position - b.position)
                .map((col) => (
                  <SelectItem key={col.id} value={col.id}>
                    <div className="flex items-center gap-2">
                      {col.color && (
                        <div
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: col.color }}
                        />
                      )}
                      {col.title}
                    </div>
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Priority */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Priority
        </Label>
        <Select
          value={priority}
          onValueChange={(val) => {
            setPriority(val);
            saveField("priority", val);
          }}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Set priority" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="low">Low</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="urgent">Urgent</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Parent Task Link */}
      {task.parent && (
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Parent Task
          </Label>
          <button
            className="flex items-center gap-2 text-sm text-primary hover:underline"
            onClick={() => {
              if (onNavigateToTask && task.parent) {
                onNavigateToTask(task.parent.id);
              }
            }}
          >
            <ArrowUpRight className="h-3.5 w-3.5" />
            {task.parent.title}
          </button>
        </div>
      )}

      {/* Link to Parent (for non-parent tasks without a parent) */}
      {!task.is_parent && !task.parent && parents.length > 0 && (
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Link to Parent Task
          </Label>
          <Select
            value=""
            onValueChange={(val) => {
              saveField("parent_task_id", val);
            }}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select parent task..." />
            </SelectTrigger>
            <SelectContent>
              {parents.map((p) => (
                <SelectItem key={p.id} value={p.id}>
                  {p.title}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      {/* Dates */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Start Date
          </Label>
          <div className="relative">
            <Popover>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  className={cn(
                    "w-full justify-start text-left font-normal",
                    !startDate && "text-muted-foreground",
                    startDate && "pr-8"
                  )}
                >
                  <CalendarIcon className="mr-2 h-4 w-4" />
                  {startDate ? format(startDate, "MMM d, yyyy") : "Pick date"}
                </Button>
              </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={startDate}
                onSelect={(date) => {
                  setStartDate(date);
                  saveField(
                    "start_date",
                    date ? format(date, "yyyy-MM-dd") : null
                  );
                }}
              />
            </PopoverContent>
            </Popover>
            {startDate && (
              <button
                type="button"
                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-sm opacity-50 hover:opacity-100"
                onClick={() => {
                  setStartDate(undefined);
                  saveField("start_date", null);
                }}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Due Date
          </Label>
          <div className="relative">
          <Popover>
            <PopoverTrigger asChild>
              <Button
                variant="outline"
                className={cn(
                  "w-full justify-start text-left font-normal",
                  !dueDate && "text-muted-foreground",
                  dueDate && "pr-8"
                )}
              >
                <CalendarIcon className="mr-2 h-4 w-4" />
                {dueDate ? format(dueDate, "MMM d, yyyy") : "Pick date"}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <Calendar
                mode="single"
                selected={dueDate}
                onSelect={(date) => {
                  setDueDate(date);
                  saveField(
                    "due_date",
                    date ? format(date, "yyyy-MM-dd") : null
                  );
                }}
              />
            </PopoverContent>
          </Popover>
          {dueDate && (
            <button
              type="button"
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-sm opacity-50 hover:opacity-100"
              onClick={() => {
                setDueDate(undefined);
                saveField("due_date", null);
              }}
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Description
        </Label>
        <Textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          onBlur={() => saveField("description", description)}
          placeholder="Add a description..."
          className="min-h-[100px]"
        />
      </div>

      {/* Color */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Color
        </Label>
        <ColorPicker
          value={color}
          onChange={(c) => {
            setColor(c);
            saveField("color", c || null);
          }}
        />
      </div>

      {/* Assignees */}
      {canAssign && (
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground uppercase tracking-wider">
            Assignees
          </Label>
          {teamId ? (
            // MultiSelectCombobox has no `disabled` prop — soft-disable while
            // assignee mutations are in flight so the picker doesn't read as broken.
            <div className={assignPending ? "opacity-60 pointer-events-none" : ""}>
              <MultiSelectCombobox
                options={[
                  ...(members ?? []).map((m) => ({
                    id: m.user_id,
                    label: m.full_name || "Unnamed member",
                  })),
                  // Assignees who have since left the team: keep them in the
                  // options so their chip renders and can be deselected.
                  ...(task.assignees ?? [])
                    .filter((a) => !(members ?? []).some((m) => m.user_id === a.user_id))
                    .map((a) => ({
                      id: a.user_id,
                      label: `${a.full_name || "Unnamed member"} (no longer in team)`,
                    })),
                ]}
                selected={assigneeIds}
                onChange={handleAssigneesChange}
                placeholder="Assign members..."
              />
            </div>
          ) : (
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <Checkbox
                disabled={assignPending}
                checked={!!user?.id && assigneeIds.includes(user.id)}
                onCheckedChange={(checked) => {
                  if (!user?.id) return;
                  if (checked) {
                    addAssignee.mutate({ taskId: task.id, userId: user.id });
                  } else {
                    removeAssignee.mutate({ taskId: task.id, userId: user.id });
                  }
                }}
              />
              Assign to me
            </label>
          )}
          {task.assignee_name && !task.assignees?.length && (
            <p className="text-xs text-muted-foreground">
              Previously assigned to: {task.assignee_name}
            </p>
          )}
        </div>
      )}

      <Separator />

      {/* Artists */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Involved Artists
        </Label>
        <MultiSelectCombobox
          options={artistField.options}
          selected={selectedArtistIds}
          onChange={(ids) => {
            setSelectedArtistIds(ids);
            saveFields({
              artist_ids: ids,
              artist_labels: labelsFor(ids, artistField.nameFor),
            });
          }}
          placeholder="Select artists..."
        />
        <ReadonlyLinkChips items={artistField.readonly} />
      </div>

      {/* Projects */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Linked Projects
        </Label>
        <MultiSelectCombobox
          options={projectField.options}
          selected={selectedProjectIds}
          onChange={(ids) => {
            setSelectedProjectIds(ids);
            saveFields({
              project_ids: ids,
              project_labels: labelsFor(ids, projectField.nameFor),
            });
          }}
          placeholder="Select projects..."
        />
        <ReadonlyLinkChips items={projectField.readonly} />
      </div>

      {/* Contracts */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Linked Contracts
        </Label>
        <MultiSelectCombobox
          options={documentField.options}
          selected={selectedContractIds}
          onChange={(ids) => {
            setSelectedContractIds(ids);
            saveFields({
              contract_ids: ids,
              contract_labels: labelsFor(ids, documentField.nameFor),
            });
          }}
          placeholder="Select contracts..."
        />
        <ReadonlyLinkChips items={documentField.readonly} />
      </div>
    </>
  );
}

// Display-only chips for linked entities the viewer can't access. No remove
// affordance and no click-through — the backend preserves these links on write.
function ReadonlyLinkChips({
  items,
}: {
  items: { id: string; name: string }[];
}) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      {items.map((item) => (
        <Badge
          key={item.id}
          variant="secondary"
          className="text-xs font-normal opacity-70"
          title="Shared by a teammate — you don't have access"
        >
          {item.name}
        </Badge>
      ))}
    </div>
  );
}
