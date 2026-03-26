import { useState, useEffect, useCallback } from "react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
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
import { CalendarIcon, Trash2, Plus, X, Loader2, CornerDownRight, ArrowUpRight, Search, Link } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { format } from "date-fns";
import { cn } from "@/lib/utils";

import { useTaskDetail } from "@/hooks/useTaskDetail";
import { useBoards } from "@/hooks/useBoards";
import { useParentTasks } from "@/hooks/useParentTasks";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { ColorPicker } from "./ColorPicker";
import { MultiSelectCombobox } from "./MultiSelectCombobox";
import { CommentSection } from "./CommentSection";
import { getLabelColor } from "./labelColors";

interface TaskDetailPanelProps {
  taskId: string | null;
  onClose: () => void;
  onNavigateToTask?: (taskId: string) => void;
}

export function TaskDetailPanel({ taskId, onClose, onNavigateToTask }: TaskDetailPanelProps) {
  const { task, isLoading, addComment, deleteComment } = useTaskDetail(taskId);
  const { columns, tasks: allBoardTasks, updateTask, deleteTask, createTask } = useBoards();
  const { parents } = useParentTasks();
  const { artists } = useArtistsList();
  const [newSubtaskTitle, setNewSubtaskTitle] = useState("");
  const [showAllSubtasks, setShowAllSubtasks] = useState(false);
  const [linkSearchOpen, setLinkSearchOpen] = useState(false);
  const [linkSearch, setLinkSearch] = useState("");

  // Local state for editable fields
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [priority, setPriority] = useState<string>("");
  const [startDate, setStartDate] = useState<Date | undefined>();
  const [dueDate, setDueDate] = useState<Date | undefined>();
  const [color, setColor] = useState<string>("");
  const [assigneeName, setAssigneeName] = useState("");
  const [selectedArtistIds, setSelectedArtistIds] = useState<string[]>([]);
  const [selectedProjectIds, setSelectedProjectIds] = useState<string[]>([]);
  const [selectedContractIds, setSelectedContractIds] = useState<string[]>([]);

  // Projects filtered by selected artists, contracts filtered by selected projects
  const { projects, contracts } = useProjectsList(
    selectedArtistIds.length > 0 ? selectedArtistIds : undefined,
    selectedProjectIds.length > 0 ? selectedProjectIds : undefined
  );
  const [labels, setLabels] = useState<string[]>([]);
  const [newLabel, setNewLabel] = useState("");
  const [statusColumnId, setStatusColumnId] = useState<string>("");

  // Sync local state when task data loads
  useEffect(() => {
    if (task) {
      setTitle(task.title || "");
      setDescription(task.description || "");
      setPriority(task.priority || "");
      setStartDate(task.start_date ? new Date(task.start_date) : undefined);
      setDueDate(task.due_date ? new Date(task.due_date) : undefined);
      setColor(task.color || "");
      setAssigneeName(task.assignee_name || "");
      setSelectedArtistIds(task.artist_ids || []);
      setSelectedProjectIds(task.project_ids || []);
      setSelectedContractIds(task.contract_ids || []);
      setLabels(task.labels || []);
      setStatusColumnId(task.column_id || "");
    }
  }, [task]);

  const saveField = useCallback(
    (field: string, value: unknown) => {
      if (!taskId) return;
      updateTask({ id: taskId, [field]: value });
    },
    [taskId, updateTask]
  );

  const handleDelete = () => {
    if (!taskId) return;
    deleteTask(taskId);
    onClose();
  };

  const addLabel = () => {
    if (!newLabel.trim() || labels.includes(newLabel.trim())) return;
    const updated = [...labels, newLabel.trim()];
    setLabels(updated);
    setNewLabel("");
    saveField("labels", updated);
  };

  const removeLabel = (label: string) => {
    const updated = labels.filter((l) => l !== label);
    setLabels(updated);
    saveField("labels", updated);
  };

  return (
    <Sheet open={!!taskId} onOpenChange={(open) => !open && onClose()} modal={false}>
      <SheetContent
        side="right"
        className="sm:max-w-lg w-[500px] overflow-y-auto"
      >
        {isLoading || !task ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : (
          <>
            <SheetHeader>
              <div className="flex items-center gap-3">
                {color && (
                  <div
                    className="w-4 h-4 rounded-full shrink-0"
                    style={{ backgroundColor: color }}
                  />
                )}
                <SheetTitle className="sr-only">Task Details</SheetTitle>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onBlur={() => saveField("title", title)}
                  className="text-lg font-semibold border-none px-0 h-auto focus-visible:ring-0"
                  placeholder="Task title"
                />
              </div>
              <SheetDescription className="sr-only">
                Edit task details
              </SheetDescription>
            </SheetHeader>

            <div className="space-y-6 mt-6">
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

              {/* Labels */}
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Labels
                </Label>
                <div className="flex flex-wrap gap-1.5 mb-2">
                  {labels.map((label) => {
                    const lc = getLabelColor(label);
                    return (
                    <Badge
                      key={label}
                      variant="secondary"
                      className={`gap-1 pr-1 ${lc.bg} ${lc.text}`}
                    >
                      {label}
                      <button
                        type="button"
                        onClick={() => removeLabel(label)}
                        className="hover:text-destructive"
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                    );
                  })}
                </div>
                <div className="flex gap-2">
                  <Input
                    value={newLabel}
                    onChange={(e) => setNewLabel(e.target.value)}
                    placeholder="Add label..."
                    className="h-8"
                    onKeyDown={(e) => e.key === "Enter" && addLabel()}
                  />
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={addLabel}
                    disabled={!newLabel.trim()}
                    className="h-8"
                  >
                    <Plus className="h-3.5 w-3.5" />
                  </Button>
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

              <Separator />

              {/* Subtasks (shown for parent tasks) */}
              {task.is_parent && (
                <div className="space-y-2">
                  <Label className="text-xs text-muted-foreground uppercase tracking-wider">
                    Subtasks ({task.children?.length || 0})
                  </Label>
                  {task.children && task.children.length > 0 && (
                    <div className="rounded-lg border divide-y">
                      {(showAllSubtasks ? task.children : task.children.slice(0, 5)).map((child) => (
                        <div
                          key={child.id}
                          className="flex items-center gap-2 px-3 py-2 hover:bg-muted/50 transition-colors text-sm group"
                        >
                          <button
                            className="flex items-center gap-2 flex-1 min-w-0 text-left"
                            onClick={() => {
                              if (onNavigateToTask) onNavigateToTask(child.id);
                            }}
                          >
                            <CornerDownRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                            {child.color && (
                              <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: child.color }} />
                            )}
                            <span className="flex-1 truncate">{child.title}</span>
                            {child.priority && (
                              <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                                {child.priority}
                              </Badge>
                            )}
                          </button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground hover:text-destructive"
                            onClick={(e) => {
                              e.stopPropagation();
                              updateTask({ id: child.id, parent_task_id: null });
                            }}
                            title="Unlink subtask"
                          >
                            <X className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      ))}
                      {task.children.length > 5 && (
                        <button
                          className="w-full text-xs text-muted-foreground hover:text-foreground py-2 transition-colors"
                          onClick={() => setShowAllSubtasks(!showAllSubtasks)}
                        >
                          {showAllSubtasks ? "Show less" : `Show ${task.children.length - 5} more`}
                        </button>
                      )}
                    </div>
                  )}
                  <div className="flex gap-2">
                    <Input
                      value={newSubtaskTitle}
                      onChange={(e) => setNewSubtaskTitle(e.target.value)}
                      placeholder="Add subtask..."
                      className="h-8"
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && newSubtaskTitle.trim()) {
                          createTask({ title: newSubtaskTitle.trim(), parent_task_id: task.id });
                          setNewSubtaskTitle("");
                        }
                      }}
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-8"
                      disabled={!newSubtaskTitle.trim()}
                      onClick={() => {
                        if (newSubtaskTitle.trim()) {
                          createTask({ title: newSubtaskTitle.trim(), parent_task_id: task.id });
                          setNewSubtaskTitle("");
                        }
                      }}
                    >
                      <Plus className="h-3.5 w-3.5" />
                    </Button>
                  </div>

                  {/* Link existing task as subtask */}
                  <Popover open={linkSearchOpen} onOpenChange={setLinkSearchOpen}>
                    <PopoverTrigger asChild>
                      <Button variant="outline" size="sm" className="w-full justify-start text-muted-foreground h-8">
                        <Link className="h-3.5 w-3.5 mr-2" />
                        Link existing task...
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[320px] p-2" align="start">
                      <div className="relative mb-2">
                        <Search className="absolute left-2 top-2 h-3.5 w-3.5 text-muted-foreground" />
                        <Input
                          placeholder="Search tasks..."
                          value={linkSearch}
                          onChange={(e) => setLinkSearch(e.target.value)}
                          className="pl-7 h-8 text-sm"
                          autoFocus
                        />
                      </div>
                      <ScrollArea className="max-h-[200px]">
                        {(() => {
                          const childIds = new Set(task.children?.map((c) => c.id) || []);
                          const orphanTasks = allBoardTasks.filter(
                            (t) => !t.parent_task_id && !t.is_parent && t.id !== task.id && !childIds.has(t.id)
                          );
                          const filtered = linkSearch
                            ? orphanTasks.filter((t) =>
                                t.title.toLowerCase().includes(linkSearch.toLowerCase())
                              )
                            : orphanTasks;

                          if (filtered.length === 0) {
                            return (
                              <p className="text-sm text-muted-foreground text-center py-3">
                                No tasks available to link
                              </p>
                            );
                          }

                          return filtered.map((t) => (
                            <button
                              key={t.id}
                              className="w-full flex items-center gap-2 px-2 py-1.5 rounded-sm hover:bg-muted text-left text-sm"
                              onClick={() => {
                                updateTask({ id: t.id, parent_task_id: task.id });
                                setLinkSearch("");
                                setLinkSearchOpen(false);
                              }}
                            >
                              {t.color && (
                                <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: t.color }} />
                              )}
                              <span className="flex-1 truncate">{t.title}</span>
                            </button>
                          ));
                        })()}
                      </ScrollArea>
                    </PopoverContent>
                  </Popover>
                </div>
              )}

              <Separator />

              {/* Comments */}
              <div className="space-y-2">
                <Label className="text-xs text-muted-foreground uppercase tracking-wider">
                  Comments
                </Label>
                <CommentSection
                  taskId={task.id}
                  comments={task.comments || []}
                  onAdd={addComment}
                  onDelete={deleteComment}
                />
              </div>

              <Separator />

              {/* Delete */}
              <Button
                variant="destructive"
                size="sm"
                onClick={handleDelete}
                className="w-full"
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete Task
              </Button>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
