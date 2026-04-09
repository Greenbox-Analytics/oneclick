import { useState, useEffect, useCallback } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Trash2, Loader2 } from "lucide-react";
import { format } from "date-fns";
import { parseDateString, getTodayString } from "@/lib/dateUtils";
import { useTaskDetail } from "@/hooks/useTaskDetail";
import { useBoards } from "@/hooks/useBoards";
import { useParentTasks } from "@/hooks/useParentTasks";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { CommentSection } from "./CommentSection";
import { TaskFields } from "./TaskFields";
import { TaskLabels } from "./TaskLabels";
import { TaskSubtasks } from "./TaskSubtasks";
import type { BoardTaskDetail } from "@/types/integrations";

interface TaskDetailPanelProps {
  taskId: string | null;
  onClose: () => void;
  onNavigateToTask?: (taskId: string) => void;
  mode?: "create" | "edit";
  createColumnId?: string;
  timezone?: string;
}

export function TaskDetailPanel({
  taskId,
  onClose,
  onNavigateToTask,
  mode = "edit",
  createColumnId,
  timezone,
}: TaskDetailPanelProps) {
  const isCreateMode = mode === "create";
  const { task, isLoading, addComment, deleteComment } = useTaskDetail(
    isCreateMode ? null : taskId
  );
  const { columns, tasks: allBoardTasks, updateTask, deleteTask, createTask } = useBoards();
  const { parents } = useParentTasks();
  const { artists } = useArtistsList();
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
  const { projects, contracts } = useProjectsList(
    selectedArtistIds.length > 0 ? selectedArtistIds : undefined,
    selectedProjectIds.length > 0 ? selectedProjectIds : undefined
  );
  const [labels, setLabels] = useState<string[]>([]);
  const [statusColumnId, setStatusColumnId] = useState<string>("");

  // Hydrate fields from existing task in edit mode
  useEffect(() => {
    if (task && !isCreateMode) {
      setTitle(task.title || "");
      setDescription(task.description || "");
      setPriority(task.priority || "");
      setStartDate(task.start_date ? parseDateString(task.start_date) : undefined);
      setDueDate(task.due_date ? parseDateString(task.due_date) : undefined);
      setColor(task.color || "");
      setAssigneeName(task.assignee_name || "");
      setSelectedArtistIds(task.artist_ids || []);
      setSelectedProjectIds(task.project_ids || []);
      setSelectedContractIds(task.contract_ids || []);
      setLabels(task.labels || []);
      setStatusColumnId(task.column_id || "");
    }
  }, [task, isCreateMode]);

  // Initialize create mode defaults
  useEffect(() => {
    if (isCreateMode) {
      const todayStr = getTodayString(timezone);
      setTitle("");
      setDescription("");
      setPriority("");
      setStartDate(parseDateString(todayStr));
      setDueDate(undefined);
      setColor("");
      setAssigneeName("");
      setSelectedArtistIds([]);
      setSelectedProjectIds([]);
      setSelectedContractIds([]);
      setLabels([]);
      setStatusColumnId(createColumnId || "");
    }
  }, [isCreateMode, createColumnId, timezone]);

  // In edit mode, save field immediately. In create mode, no-op.
  const saveField = useCallback(
    (field: string, value: unknown) => {
      if (isCreateMode) return;
      if (!taskId) return;
      updateTask({ id: taskId, [field]: value });
    },
    [taskId, updateTask, isCreateMode]
  );

  const handleSaveNewTask = () => {
    if (!title.trim()) return;
    createTask({
      column_id: createColumnId,
      title: title.trim(),
      description: description || undefined,
      priority: priority || undefined,
      start_date: startDate ? format(startDate, "yyyy-MM-dd") : undefined,
      due_date: dueDate ? format(dueDate, "yyyy-MM-dd") : undefined,
      color: color || undefined,
      artist_ids: selectedArtistIds.length > 0 ? selectedArtistIds : undefined,
      project_ids: selectedProjectIds.length > 0 ? selectedProjectIds : undefined,
      contract_ids: selectedContractIds.length > 0 ? selectedContractIds : undefined,
      labels: labels.length > 0 ? labels : undefined,
    });
    onClose();
  };

  const handleDelete = () => {
    if (!taskId) return;
    deleteTask(taskId);
    onClose();
  };

  const isOpen = isCreateMode ? !!createColumnId : !!taskId;

  const taskForFields = isCreateMode
    ? ({
        id: "",
        user_id: "",
        title: "",
        position: 0,
        is_parent: false,
        parent: null,
        artists: [],
        projects: [],
        contracts: [],
        comments: [],
        created_at: "",
        updated_at: "",
      } as unknown as BoardTaskDetail)
    : task;

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()} modal={false}>
      <SheetContent side="right" className="sm:max-w-lg w-[500px] overflow-y-auto">
        {!isCreateMode && (isLoading || !task) ? (
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
                <SheetTitle className="sr-only">
                  {isCreateMode ? "New Task" : "Task Details"}
                </SheetTitle>
                <Input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  onBlur={() => {
                    if (!isCreateMode) saveField("title", title);
                  }}
                  className="text-lg font-semibold border-none px-0 h-auto focus-visible:ring-0"
                  placeholder={isCreateMode ? "New task title..." : "Task title"}
                  autoFocus={isCreateMode}
                />
              </div>
              <SheetDescription className="sr-only">
                {isCreateMode ? "Create a new task" : "Edit task details"}
              </SheetDescription>
            </SheetHeader>

            <div className="space-y-6 mt-6">
              {taskForFields && (
                <TaskFields
                  task={taskForFields}
                  description={description}
                  setDescription={setDescription}
                  priority={priority}
                  setPriority={setPriority}
                  startDate={startDate}
                  setStartDate={setStartDate}
                  dueDate={dueDate}
                  setDueDate={setDueDate}
                  color={color}
                  setColor={setColor}
                  assigneeName={assigneeName}
                  setAssigneeName={setAssigneeName}
                  statusColumnId={statusColumnId}
                  setStatusColumnId={setStatusColumnId}
                  columns={columns}
                  parents={parents}
                  artists={artists.map((a) => ({ id: a.id, name: a.name }))}
                  projects={projects.map((p) => ({ id: p.id, name: p.name }))}
                  contracts={contracts.map((c) => ({
                    id: c.id,
                    file_name: c.file_name,
                  }))}
                  selectedArtistIds={selectedArtistIds}
                  setSelectedArtistIds={setSelectedArtistIds}
                  selectedProjectIds={selectedProjectIds}
                  setSelectedProjectIds={setSelectedProjectIds}
                  selectedContractIds={selectedContractIds}
                  setSelectedContractIds={setSelectedContractIds}
                  saveField={saveField}
                  onNavigateToTask={onNavigateToTask}
                />
              )}

              <TaskLabels
                labels={labels}
                setLabels={setLabels}
                saveField={saveField}
              />

              {!isCreateMode && (
                <>
                  <Separator />
                  <TaskSubtasks
                    task={task!}
                    allBoardTasks={allBoardTasks}
                    onNavigateToTask={onNavigateToTask}
                    updateTask={updateTask}
                    createTask={createTask}
                  />

                  <Separator />
                  <div className="space-y-2">
                    <Label className="text-xs text-muted-foreground uppercase tracking-wider">
                      Comments
                    </Label>
                    <CommentSection
                      taskId={task!.id}
                      comments={task!.comments || []}
                      onAdd={addComment}
                      onDelete={deleteComment}
                    />
                  </div>
                </>
              )}

              <Separator />

              {isCreateMode ? (
                <div className="flex gap-2">
                  <Button onClick={handleSaveNewTask} disabled={!title.trim()} className="flex-1">
                    Save
                  </Button>
                  <Button variant="outline" onClick={onClose}>
                    Cancel
                  </Button>
                </div>
              ) : (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={handleDelete}
                  className="w-full"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete Task
                </Button>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
