import { useState, useEffect, useCallback } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Trash2, Loader2 } from "lucide-react";
import { format } from "date-fns";
import { parseDateString, getTodayString } from "@/lib/dateUtils";
import { toast } from "sonner";
import { useTaskDetail } from "@/hooks/useTaskDetail";
import { useBoards, type OptimisticTaskContext } from "@/hooks/useBoards";
import { useParentTasks } from "@/hooks/useParentTasks";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { CommentSection } from "./CommentSection";
import { TaskFields } from "./TaskFields";
import { TaskLabels } from "./TaskLabels";
import { TaskSubtasks } from "./TaskSubtasks";
import type { BoardTaskDetail } from "@/types/integrations";
import { useGatedAction } from "@/hooks/useGatedAction";
import { useEntitlements } from "@/hooks/useEntitlements";
import { apiFetch, API_URL } from "@/lib/apiFetch";

interface TaskDetailPanelProps {
  taskId: string | null;
  onClose: () => void;
  onNavigateToTask?: (taskId: string) => void;
  mode?: "create" | "edit";
  createColumnId?: string;
  timezone?: string;
  boardId?: string;
  // Threaded from the board switcher so TaskFields can pick the assignee UI
  // (get_task_detail returns board_id only, not team_id).
  teamId?: string | null;
}

export function TaskDetailPanel({
  taskId,
  onClose,
  onNavigateToTask,
  mode = "edit",
  createColumnId,
  timezone,
  boardId,
  teamId,
}: TaskDetailPanelProps) {
  const isCreateMode = mode === "create";
  const { task, isLoading, addComment, deleteComment } = useTaskDetail(
    isCreateMode ? null : taskId
  );
  const {
    columns,
    tasks: allBoardTasks,
    updateTask,
    deleteTask,
    createTask,
    applyOptimisticTaskCreate,
    rollbackTaskCaches,
    reconcileTaskCaches,
  } = useBoards({ boardId });
  const { parents } = useParentTasks(undefined, undefined, boardId);

  const { data: ent } = useEntitlements();
  const taskCap = ent?.caps.maxTasks ?? 0;
  const currentTaskCount = allBoardTasks.length;
  const { artists } = useArtistsList();
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [priority, setPriority] = useState<string>("");
  const [startDate, setStartDate] = useState<Date | undefined>();
  const [dueDate, setDueDate] = useState<Date | undefined>();
  const [color, setColor] = useState<string>("");
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
      // Hydrate the pickers from the can_open subset only — the full accessible
      // linked set the viewer may edit. Non-accessible links render as read-only
      // chips inside TaskFields and are preserved server-side by the merge.
      setSelectedArtistIds(task.artists?.filter((a) => a.can_open).map((a) => a.id) ?? []);
      setSelectedProjectIds(task.projects?.filter((p) => p.can_open).map((p) => p.id) ?? []);
      setSelectedContractIds(task.documents?.filter((d) => d.can_open).map((d) => d.id) ?? []);
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

  // Batch several fields into one updateTask call. Used for linked entities so
  // the backend merge sees <kind>_ids and <kind>_labels in the same write.
  const saveFields = useCallback(
    (fields: Record<string, unknown>) => {
      if (isCreateMode) return;
      if (!taskId) return;
      updateTask({ id: taskId, ...fields });
    },
    [taskId, updateTask, isCreateMode]
  );

  type CreateTaskVars = {
    column_id?: string;
    title: string;
    description?: string;
    priority?: string;
    start_date?: string;
    due_date?: string;
    color?: string;
    board_id?: string;
    artist_ids?: string[];
    project_ids?: string[];
    contract_ids?: string[];
    artist_labels?: Record<string, string>;
    project_labels?: Record<string, string>;
    contract_labels?: Record<string, string>;
    labels?: string[];
  };

  const { mutate: gatedCreateTask, isPending: isCreatingTask, paywallElement: taskPaywallElement } = useGatedAction<
    unknown,
    CreateTaskVars,
    OptimisticTaskContext
  >({
    mutationFn: async (vars) => {
      return apiFetch(`${API_URL}/boards/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(vars),
      });
    },
    // Optimistic: drop the card onto the board and close the panel the instant Save
    // is clicked. The POST runs in the background; onSettled reconciles the temp card
    // with the real server row (or rolls it back on failure).
    onMutate: async (vars) => {
      const shapeLinks = (ids?: string[], labels?: Record<string, string>) =>
        (ids || []).map((id) => ({ id, name: labels?.[id] || "…", can_open: true }));
      const context = await applyOptimisticTaskCreate({
        column_id: vars.column_id,
        title: vars.title,
        description: vars.description,
        priority: vars.priority,
        start_date: vars.start_date,
        due_date: vars.due_date,
        color: vars.color,
        board_id: vars.board_id,
        artist_ids: vars.artist_ids,
        project_ids: vars.project_ids,
        contract_ids: vars.contract_ids,
        labels: vars.labels,
        artists: shapeLinks(vars.artist_ids, vars.artist_labels),
        projects: shapeLinks(vars.project_ids, vars.project_labels),
        documents: shapeLinks(vars.contract_ids, vars.contract_labels),
      });
      onClose();
      return context;
    },
    onError: () => {
      // 402/storage-cap are swallowed into the paywall modal by useGatedAction; this
      // only fires for real failures (network, 500). Rollback happens in onSettled.
      toast.error("Couldn't create the task — please try again.");
    },
    onSettled: (_data, err, _vars, context) => {
      // Roll back the temp card on any failure (incl. the swallowed-402 path, where
      // onError never runs), then always reconcile temp → real server row.
      if (err) rollbackTaskCaches(context);
      reconcileTaskCaches();
    },
    resource: "task",
    currentCount: currentTaskCount,
    cap: taskCap,
  });

  const handleSaveNewTask = () => {
    if (!title.trim()) return;
    // Snapshot the picked entities' names into *_labels so newly-created links
    // carry a non-NULL label (visible to teammates who can't access them).
    const labelsFrom = (ids: string[], nameFor: (id: string) => string | undefined) =>
      Object.fromEntries(
        ids.map((id) => [id, nameFor(id)]).filter(([, name]) => !!name)
      ) as Record<string, string>;
    gatedCreateTask({
      column_id: createColumnId,
      title: title.trim(),
      description: description || undefined,
      priority: priority || undefined,
      start_date: startDate ? format(startDate, "yyyy-MM-dd") : undefined,
      due_date: dueDate ? format(dueDate, "yyyy-MM-dd") : undefined,
      color: color || undefined,
      board_id: boardId,
      artist_ids: selectedArtistIds.length > 0 ? selectedArtistIds : undefined,
      project_ids: selectedProjectIds.length > 0 ? selectedProjectIds : undefined,
      contract_ids: selectedContractIds.length > 0 ? selectedContractIds : undefined,
      artist_labels:
        selectedArtistIds.length > 0
          ? labelsFrom(selectedArtistIds, (id) => artists.find((a) => a.id === id)?.name)
          : undefined,
      project_labels:
        selectedProjectIds.length > 0
          ? labelsFrom(selectedProjectIds, (id) => projects.find((p) => p.id === id)?.name)
          : undefined,
      contract_labels:
        selectedContractIds.length > 0
          ? labelsFrom(selectedContractIds, (id) => contracts.find((c) => c.id === id)?.file_name)
          : undefined,
      labels: labels.length > 0 ? labels : undefined,
    });
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
        documents: [],
        comments: [],
        created_at: "",
        updated_at: "",
      } as unknown as BoardTaskDetail)
    : task;

  return (
    <>
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()} modal={false}>
      <SheetContent side="right" className="w-full sm:w-[500px] sm:max-w-[500px] overflow-y-auto">
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
                  teamId={teamId}
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
                  saveFields={saveFields}
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
                  <Button onClick={handleSaveNewTask} disabled={!title.trim() || isCreatingTask} className="flex-1">
                    {isCreatingTask ? <Loader2 className="w-4 h-4 animate-spin mr-1" /> : null}
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
    {taskPaywallElement}
    </>
  );
}
