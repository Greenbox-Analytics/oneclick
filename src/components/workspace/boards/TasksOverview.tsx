import { useState, useCallback, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Plus, Search, Loader2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DndContext,
  DragOverlay,
  closestCorners,
  DragStartEvent,
  DragEndEvent,
  PointerSensor,
  useSensor,
  useSensors,
  useDroppable,
} from "@dnd-kit/core";
import { SortableContext, rectSortingStrategy } from "@dnd-kit/sortable";
import { useParentTasks } from "@/hooks/useParentTasks";
import { useBoards } from "@/hooks/useBoards";
import { useArtistsList } from "@/hooks/useArtistsList";
import { ParentTaskRow } from "./ParentTaskRow";
import { ParentKanbanCard } from "./ParentKanbanCard";
import { TaskDetailPanel } from "./TaskDetailPanel";
import type { BoardColumn, ParentTaskWithChildren } from "@/types/integrations";

export function TasksOverview() {
  const [searchQuery, setSearchQuery] = useState("");
  const [artistFilter, setArtistFilter] = useState<string>("");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [isCreatingParent, setIsCreatingParent] = useState(false);
  const [newParentTitle, setNewParentTitle] = useState("");
  const [activeParentId, setActiveParentId] = useState<string | null>(null);

  const [debouncedSearch, setDebouncedSearch] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(searchQuery), 1000);
    return () => clearTimeout(timer);
  }, [searchQuery]);

  const { parents, ungrouped, isLoading, createParent, deleteParent } = useParentTasks(
    debouncedSearch || undefined,
    artistFilter || undefined
  );
  const { columns, createTask, updateTask } = useBoards();
  const { artists } = useArtistsList();

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } })
  );

  const handleCreateParent = () => {
    if (!newParentTitle.trim()) return;
    createParent({ title: newParentTitle.trim(), start_date: new Date().toISOString().split("T")[0] });
    setNewParentTitle("");
    setIsCreatingParent(false);
  };

  const handleAddSubtask = (parentId: string, title: string) => {
    createTask({ title, parent_task_id: parentId, start_date: new Date().toISOString().split("T")[0] });
  };

  const handleParentDragStart = useCallback((event: DragStartEvent) => {
    setActiveParentId(event.active.id as string);
  }, []);

  const handleParentDragEnd = useCallback(
    (event: DragEndEvent) => {
      setActiveParentId(null);
      const { active, over } = event;
      if (!over) return;

      const parentId = active.id as string;

      // Determine target column (null = backlog)
      let targetColumnId: string | null = null;
      let found = false;
      if (over.data.current?.type === "parent-column") {
        targetColumnId = over.data.current.columnId;
        found = true;
      } else if (over.data.current?.type === "parent-task") {
        const overParent = parents.find((p) => p.id === over.id);
        targetColumnId = overParent?.column_id || null;
        found = true;
      }

      if (!found) return;

      const activeParent = parents.find((p) => p.id === parentId);
      const currentColumnId = activeParent?.column_id || null;
      if (currentColumnId === targetColumnId) return;

      // Pass column_id (null clears it for backlog)
      updateTask({ id: parentId, column_id: targetColumnId ?? "" });
    },
    [parents, updateTask]
  );

  // Filter out "Review" for parent board
  const parentColumns = columns.filter(
    (c) => c.title.toLowerCase() !== "review"
  );

  // Group parents by column for the Kanban board
  const parentsByColumn = new Map<string, ParentTaskWithChildren[]>();
  const backlogParents: ParentTaskWithChildren[] = [];
  for (const p of parents) {
    if (p.column_id) {
      if (!parentsByColumn.has(p.column_id)) parentsByColumn.set(p.column_id, []);
      parentsByColumn.get(p.column_id)!.push(p);
    } else {
      backlogParents.push(p);
    }
  }

  const allParentIds = parents.map((p) => p.id);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <>
      <div className="space-y-6">
        {/* Header + filters */}
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2 flex-1 min-w-[300px]">
            <div className="relative flex-1 max-w-[300px]">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search tasks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 h-9"
              />
            </div>
            <Select
              value={artistFilter}
              onValueChange={(val) => setArtistFilter(val === "all" ? "" : val)}
            >
              <SelectTrigger className="w-[180px] h-9">
                <SelectValue placeholder="All artists" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All artists</SelectItem>
                {artists.map((a) => (
                  <SelectItem key={a.id} value={a.id}>
                    {a.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {isCreatingParent ? (
            <div className="flex gap-2 items-center">
              <Input
                placeholder="Parent task title..."
                value={newParentTitle}
                onChange={(e) => setNewParentTitle(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleCreateParent();
                  if (e.key === "Escape") setIsCreatingParent(false);
                }}
                autoFocus
                className="w-[250px] h-9"
              />
              <Button size="sm" onClick={handleCreateParent}>
                Create
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setIsCreatingParent(false)}
              >
                Cancel
              </Button>
            </div>
          ) : (
            <Button size="sm" onClick={() => setIsCreatingParent(true)}>
              <Plus className="h-4 w-4 mr-1" />
              New Campaign
            </Button>
          )}
        </div>

        {parents.length === 0 && ungrouped.length === 0 && !artistFilter && (
          <div className="text-center py-12 text-muted-foreground">
            <p className="text-lg font-semibold mb-2">No campaigns yet</p>
            <p>Create a campaign to start organizing your work</p>
          </div>
        )}

        {parents.length === 0 && artistFilter && (
          <div className="flex items-center gap-2 rounded-lg border border-yellow-200 bg-yellow-50 dark:border-yellow-900 dark:bg-yellow-950/30 px-4 py-3 text-sm text-yellow-800 dark:text-yellow-200">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <p>No campaigns found for the selected artist. Try selecting a different artist or create a new campaign.</p>
          </div>
        )}

        {/* Parent task Kanban board */}
        {parents.length > 0 && parentColumns.length > 0 && (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCorners}
            onDragStart={handleParentDragStart}
            onDragEnd={handleParentDragEnd}
          >
            <SortableContext items={allParentIds} strategy={rectSortingStrategy}>
              <div className="flex gap-3 overflow-x-auto pb-3">
                {/* Unassigned column */}
                <ParentColumn
                  id="backlog"
                  title="Backlog"
                  color="#9ca3af"
                  parents={backlogParents}
                  onCardClick={setSelectedTaskId}
                />
                {parentColumns
                  .sort((a, b) => a.position - b.position)
                  .map((col) => (
                    <ParentColumn
                      key={col.id}
                      id={col.id}
                      title={col.title}
                      color={col.color}
                      parents={parentsByColumn.get(col.id) || []}
                      onCardClick={setSelectedTaskId}
                    />
                  ))}
              </div>
            </SortableContext>
            <DragOverlay dropAnimation={null}>
              {activeParentId ? (() => {
                const p = parents.find((p) => p.id === activeParentId);
                return p ? (
                  <div className="w-56 shadow-xl">
                    <ParentKanbanCard parent={p} onClick={() => {}} />
                  </div>
                ) : null;
              })() : null}
            </DragOverlay>
          </DndContext>
        )}

        {/* Active campaign tasks list (excludes done campaigns) */}
        {(() => {
          const activeParents = parents.filter(
            (p) => p.column_title?.toLowerCase() !== "done"
          );
          return activeParents.length > 0 ? (
            <>
              <Separator />
              <h4 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Active Campaign Tasks
              </h4>
              <div className="space-y-3">
                {activeParents.map((parent) => (
                  <ParentTaskRow
                    key={parent.id}
                    parent={parent}
                    onTaskClick={setSelectedTaskId}
                    onParentClick={setSelectedTaskId}
                    onAddSubtask={handleAddSubtask}
                    onDelete={deleteParent}
                  />
                ))}
              </div>
            </>
          ) : null;
        })()}

      </div>

      <TaskDetailPanel
        taskId={selectedTaskId}
        onClose={() => setSelectedTaskId(null)}
      />
    </>
  );
}

// --- Mini Kanban column for parent tasks ---

function ParentColumn({
  id,
  title,
  color,
  parents,
  onCardClick,
}: {
  id: string;
  title: string;
  color?: string | null;
  parents: ParentTaskWithChildren[];
  onCardClick: (taskId: string) => void;
}) {
  const { setNodeRef, isOver } = useDroppable({
    id: `parent-col-${id}`,
    data: { type: "parent-column", columnId: id === "backlog" ? null : id },
  });

  return (
    <div
      className={cn(
        "flex flex-col w-56 shrink-0 bg-muted/50 rounded-lg transition-all duration-200",
        isOver && "ring-2 shadow-lg"
      )}
      style={isOver && color ? {
        boxShadow: `0 0 20px ${color}30`,
        outline: `2px solid ${color}`,
        outlineOffset: '2px',
      } : undefined}
    >
      <div className="flex items-center gap-2 px-3 py-2">
        {color && (
          <div
            className="w-2 h-2 rounded-full shrink-0"
            style={{ backgroundColor: color }}
          />
        )}
        <h5 className="text-xs font-semibold">{title}</h5>
        <span className="text-[10px] text-muted-foreground bg-muted rounded-full px-1.5">
          {parents.length}
        </span>
      </div>
      <div
        ref={setNodeRef}
        className="flex-1 overflow-y-auto px-2 pb-2 space-y-2 min-h-[60px]"
      >
        {parents.map((parent) => (
          <ParentKanbanCard
            key={parent.id}
            parent={parent}
            onClick={onCardClick}
          />
        ))}
      </div>
    </div>
  );
}

