import { useState, useEffect, useCallback } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Trash2, Loader2 } from "lucide-react";
import { useTaskDetail } from "@/hooks/useTaskDetail";
import { useBoards } from "@/hooks/useBoards";
import { useParentTasks } from "@/hooks/useParentTasks";
import { useArtistsList } from "@/hooks/useArtistsList";
import { useProjectsList } from "@/hooks/useProjectsList";
import { CommentSection } from "./CommentSection";
import { TaskFields } from "./TaskFields";
import { TaskLabels } from "./TaskLabels";
import { TaskSubtasks } from "./TaskSubtasks";

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

  return (
    <Sheet open={!!taskId} onOpenChange={(open) => !open && onClose()} modal={false}>
      <SheetContent side="right" className="sm:max-w-lg w-[500px] overflow-y-auto">
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
              <SheetDescription className="sr-only">Edit task details</SheetDescription>
            </SheetHeader>

            <div className="space-y-6 mt-6">
              <TaskFields
                task={task}
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
                contracts={contracts.map((c) => ({ id: c.id, file_name: c.file_name }))}
                selectedArtistIds={selectedArtistIds}
                setSelectedArtistIds={setSelectedArtistIds}
                selectedProjectIds={selectedProjectIds}
                setSelectedProjectIds={setSelectedProjectIds}
                selectedContractIds={selectedContractIds}
                setSelectedContractIds={setSelectedContractIds}
                saveField={saveField}
                onNavigateToTask={onNavigateToTask}
              />

              <TaskLabels
                labels={labels}
                setLabels={setLabels}
                saveField={saveField}
              />

              <Separator />

              <TaskSubtasks
                task={task}
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
                  taskId={task.id}
                  comments={task.comments || []}
                  onAdd={addComment}
                  onDelete={deleteComment}
                />
              </div>

              <Separator />

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
