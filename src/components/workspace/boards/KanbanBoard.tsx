import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Loader2, ChevronLeft, ChevronRight } from "lucide-react";
import {
  DndContext,
  DragOverlay,
  closestCorners,
  DragStartEvent,
  DragEndEvent,
  PointerSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { SortableContext, rectSortingStrategy } from "@dnd-kit/sortable";
import { Separator } from "@/components/ui/separator";
import { KanbanColumn } from "./KanbanColumn";
import { KanbanCard } from "./KanbanCard";
import { TaskDetailPanel } from "./TaskDetailPanel";
import { TasksOverview } from "./TasksOverview";
import { useBoards } from "@/hooks/useBoards";
import { useParentTasks } from "@/hooks/useParentTasks";
import { useBoardPeriod } from "@/hooks/useBoardPeriod";
import { useWorkspaceSettings } from "@/hooks/useWorkspaceSettings";

interface KanbanBoardProps {
  artistId?: string;
}

export function KanbanBoard({ artistId }: KanbanBoardProps) {
  const { settings } = useWorkspaceSettings();
  const {
    periodStart,
    periodEnd,
    periodLabel,
    isCurrentPeriod,
    goToPrevPeriod,
    goToNextPeriod,
    goToCurrentPeriod,
  } = useBoardPeriod({
    boardPeriod: settings?.board_period || "monthly",
    customPeriodDays: settings?.custom_period_days ?? 14,
  });

  const {
    columns,
    tasks,
    isLoading,
    createColumn,
    updateColumn,
    deleteColumn,
    createTask,
    deleteTask,
    reorderTasks,
    createDefaults,
  } = useBoards({
    artistId,
    periodStart,
    periodEnd,
    isCurrentPeriod,
  });

  const { parents, createParent } = useParentTasks();

  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [isAddingColumn, setIsAddingColumn] = useState(false);
  const [newColumnTitle, setNewColumnTitle] = useState("");

  // DnD sensors — add activation distance to avoid triggering on clicks
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: { distance: 8 },
    })
  );

  const handleDragStart = useCallback((event: DragStartEvent) => {
    setActiveTaskId(event.active.id as string);
  }, []);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      setActiveTaskId(null);
      const { active, over } = event;
      if (!over) return;

      const taskId = active.id as string;
      const activeTask = tasks.find((t) => t.id === taskId);
      if (!activeTask) return;

      // Determine target column
      let targetColumnId: string;
      if (over.data.current?.type === "column") {
        targetColumnId = over.data.current.columnId;
      } else if (over.data.current?.type === "task") {
        // Dropped on another task — use that task's column
        const overTask = tasks.find((t) => t.id === over.id);
        targetColumnId = overTask?.column_id || activeTask.column_id || "";
      } else {
        // Fallback: extract column id from droppable id
        const overId = over.id as string;
        targetColumnId = overId.startsWith("column-") ? overId.replace("column-", "") : activeTask.column_id || "";
      }

      if (!targetColumnId) return;

      // Calculate position: place at end of target column
      const targetColumnTasks = tasks
        .filter((t) => t.column_id === targetColumnId && t.id !== taskId)
        .sort((a, b) => a.position - b.position);

      let newPosition: number;
      if (over.data.current?.type === "task" && over.id !== active.id) {
        // Insert at the position of the task we dropped on
        const overIndex = targetColumnTasks.findIndex((t) => t.id === over.id);
        newPosition = overIndex >= 0 ? overIndex : targetColumnTasks.length;
      } else {
        newPosition = targetColumnTasks.length;
      }

      // Only update if something changed
      if (activeTask.column_id === targetColumnId && activeTask.position === newPosition) {
        return;
      }

      reorderTasks([
        {
          task_id: taskId,
          target_column_id: targetColumnId,
          position: newPosition,
        },
      ]);
    },
    [tasks, reorderTasks]
  );

  const handleAddColumn = () => {
    if (!newColumnTitle.trim()) return;
    createColumn({
      title: newColumnTitle.trim(),
      artist_id: artistId,
    });
    setNewColumnTitle("");
    setIsAddingColumn(false);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (columns.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-center">
        <h3 className="text-lg font-semibold mb-2">No board yet</h3>
        <p className="text-muted-foreground mb-4">
          Create your first project board to start managing tasks
        </p>
        <Button onClick={() => createDefaults()}>
          <Plus className="w-4 h-4 mr-2" />
          Create Default Board
        </Button>
      </div>
    );
  }

  return (
    <>
      {/* Period navigation */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={goToPrevPeriod}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={goToNextPeriod}>
            <ChevronRight className="h-4 w-4" />
          </Button>
          <h3 className="text-lg font-semibold ml-2">{periodLabel}</h3>
        </div>
        {!isCurrentPeriod && (
          <Button variant="outline" size="sm" onClick={goToCurrentPeriod}>
            Current
          </Button>
        )}
      </div>

      <DndContext
        sensors={sensors}
        collisionDetection={closestCorners}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <SortableContext items={tasks.map((t) => t.id)} strategy={rectSortingStrategy}>
        <div className="flex gap-4 overflow-x-auto pb-4 min-h-[400px]">
          {columns
            .sort((a, b) => a.position - b.position)
            .map((column) => (
              <KanbanColumn
                key={column.id}
                column={column}
                tasks={tasks}
                parentTasks={parents}
                onCreateTask={createTask}
                onCreateParent={createParent}
                onDeleteTask={deleteTask}
                onDeleteColumn={deleteColumn}
                onUpdateColumn={updateColumn}
                onTaskClick={setSelectedTaskId}
              />
            ))}

          {/* Add column */}
          <div className="w-72 shrink-0">
            {isAddingColumn ? (
              <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                <Input
                  placeholder="Column title..."
                  value={newColumnTitle}
                  onChange={(e) => setNewColumnTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAddColumn();
                    if (e.key === "Escape") setIsAddingColumn(false);
                  }}
                  autoFocus
                />
                <div className="flex gap-2">
                  <Button size="sm" onClick={handleAddColumn} className="flex-1">
                    Add
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setIsAddingColumn(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            ) : (
              <Button
                variant="outline"
                className="w-full justify-start border-dashed h-12"
                onClick={() => setIsAddingColumn(true)}
              >
                <Plus className="w-4 h-4 mr-2" />
                Add column
              </Button>
            )}
          </div>
        </div>
        </SortableContext>
        <DragOverlay dropAnimation={null}>
          {activeTaskId ? (() => {
            const t = tasks.find((t) => t.id === activeTaskId);
            return t ? (
              <div className="w-72 shadow-xl">
                <KanbanCard task={t} onDelete={() => {}} onClick={() => {}} />
              </div>
            ) : null;
          })() : null}
        </DragOverlay>
      </DndContext>

      {/* Parent tasks section */}
      <Separator className="my-8" />
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-1">Epics</h3>
        <p className="text-sm text-muted-foreground">
          Group and organize subtasks under epics
        </p>
      </div>
      <TasksOverview />

      {/* Task detail side panel */}
      <TaskDetailPanel
        taskId={selectedTaskId}
        onClose={() => setSelectedTaskId(null)}
      />
    </>
  );
}
