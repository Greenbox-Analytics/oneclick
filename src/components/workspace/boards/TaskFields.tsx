import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
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
  assigneeName: string;
  setAssigneeName: (v: string) => void;
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
  assigneeName,
  setAssigneeName,
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
  onNavigateToTask,
}: TaskFieldsProps) {
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

      {/* Assignee */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Assignee
        </Label>
        <Input
          value={assigneeName}
          onChange={(e) => setAssigneeName(e.target.value)}
          onBlur={() => saveField("assignee_name", assigneeName)}
          placeholder="Assignee name"
        />
      </div>

      <Separator />

      {/* Artists */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Involved Artists
        </Label>
        <MultiSelectCombobox
          options={artists.map((a) => ({ id: a.id, label: a.name }))}
          selected={selectedArtistIds}
          onChange={(ids) => {
            setSelectedArtistIds(ids);
            saveField("artist_ids", ids);
          }}
          placeholder="Select artists..."
        />
      </div>

      {/* Projects */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Linked Projects
        </Label>
        <MultiSelectCombobox
          options={projects.map((p) => ({ id: p.id, label: p.name }))}
          selected={selectedProjectIds}
          onChange={(ids) => {
            setSelectedProjectIds(ids);
            saveField("project_ids", ids);
          }}
          placeholder="Select projects..."
        />
      </div>

      {/* Contracts */}
      <div className="space-y-2">
        <Label className="text-xs text-muted-foreground uppercase tracking-wider">
          Linked Contracts
        </Label>
        <MultiSelectCombobox
          options={contracts.map((c) => ({
            id: c.id,
            label: c.file_name,
          }))}
          selected={selectedContractIds}
          onChange={(ids) => {
            setSelectedContractIds(ids);
            saveField("contract_ids", ids);
          }}
          placeholder="Select contracts..."
        />
      </div>
    </>
  );
}
