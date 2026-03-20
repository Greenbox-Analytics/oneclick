import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Plus, MoreHorizontal, Trash2, Pencil } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useDroppable } from "@dnd-kit/core";
import { KanbanCard } from "./KanbanCard";
import type { BoardColumn, BoardTask, ParentTaskWithChildren } from "@/types/integrations";

interface KanbanColumnProps {
  column: BoardColumn;
  tasks: BoardTask[];
  parentTasks: ParentTaskWithChildren[];
  onCreateTask: (data: { column_id?: string; title: string; parent_task_id?: string; is_parent?: boolean }) => void;
  onCreateParent: (data: { title: string }) => void;
  onDeleteTask: (taskId: string) => void;
  onDeleteColumn: (columnId: string) => void;
  onUpdateColumn: (data: { id: string; title?: string }) => void;
  onTaskClick: (taskId: string) => void;
}

export function KanbanColumn({
  column,
  tasks,
  parentTasks,
  onCreateTask,
  onCreateParent,
  onDeleteTask,
  onDeleteColumn,
  onUpdateColumn,
  onTaskClick,
}: KanbanColumnProps) {
  const [isAdding, setIsAdding] = useState(false);
  const [newTaskTitle, setNewTaskTitle] = useState("");
  const [isParent, setIsParent] = useState(false);
  const [selectedParentId, setSelectedParentId] = useState("");
  const [isRenaming, setIsRenaming] = useState(false);
  const [renameTitle, setRenameTitle] = useState(column.title);

  const handleRename = () => {
    const trimmed = renameTitle.trim();
    if (trimmed && trimmed !== column.title) {
      onUpdateColumn({ id: column.id, title: trimmed });
    }
    setIsRenaming(false);
  };

  const { setNodeRef, isOver } = useDroppable({
    id: `column-${column.id}`,
    data: { type: "column", columnId: column.id },
  });

  const handleAddTask = () => {
    if (!newTaskTitle.trim()) return;

    if (isParent) {
      onCreateParent({ title: newTaskTitle.trim() });
    } else {
      onCreateTask({
        column_id: column.id,
        title: newTaskTitle.trim(),
        parent_task_id: selectedParentId || undefined,
      });
    }

    setNewTaskTitle("");
    setIsParent(false);
    setSelectedParentId("");
    setIsAdding(false);
  };

  const columnTasks = tasks
    .filter((t) => t.column_id === column.id)
    .sort((a, b) => a.position - b.position);

  return (
    <div
      className={`flex flex-col w-72 shrink-0 bg-muted/50 rounded-lg transition-all duration-200 ${
        isOver ? "ring-2 shadow-lg" : ""
      }`}
      style={isOver && column.color ? {
        boxShadow: `0 0 20px ${column.color}30`,
        borderColor: column.color,
        ringColor: column.color,
        outline: `2px solid ${column.color}`,
        outlineOffset: '2px',
      } : undefined}
    >
      {/* Column header */}
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex items-center gap-2 min-w-0">
          {column.color && (
            <div
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ backgroundColor: column.color }}
            />
          )}
          {isRenaming ? (
            <Input
              value={renameTitle}
              onChange={(e) => setRenameTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleRename();
                if (e.key === "Escape") {
                  setRenameTitle(column.title);
                  setIsRenaming(false);
                }
              }}
              onBlur={handleRename}
              autoFocus
              className="h-6 text-sm font-semibold px-1 py-0"
            />
          ) : (
            <h4 className="text-sm font-semibold truncate">{column.title}</h4>
          )}
          <span className="text-xs text-muted-foreground bg-muted rounded-full px-1.5 shrink-0">
            {columnTasks.length}
          </span>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={() => {
                setRenameTitle(column.title);
                setIsRenaming(true);
              }}
            >
              <Pencil className="h-4 w-4 mr-2" />
              Rename column
            </DropdownMenuItem>
            <DropdownMenuItem
              className="text-destructive"
              onClick={() => onDeleteColumn(column.id)}
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete column
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Task list (droppable zone) */}
      <div
        ref={setNodeRef}
        className="flex-1 overflow-y-auto px-2 pb-2 space-y-2 min-h-[100px]"
      >
        {columnTasks.map((task) => (
          <KanbanCard
            key={task.id}
            task={task}
            onDelete={onDeleteTask}
            onClick={onTaskClick}
          />
        ))}
      </div>

      {/* Add task */}
      <div className="px-2 pb-2">
        {isAdding ? (
          <div className="space-y-2">
            <Input
              placeholder="Task title..."
              value={newTaskTitle}
              onChange={(e) => setNewTaskTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAddTask();
                if (e.key === "Escape") {
                  setIsAdding(false);
                  setIsParent(false);
                  setSelectedParentId("");
                }
              }}
              autoFocus
            />

            {/* Parent task checkbox */}
            <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
              <Checkbox
                checked={isParent}
                onCheckedChange={(checked) => {
                  setIsParent(!!checked);
                  if (checked) setSelectedParentId("");
                }}
              />
              Make this an epic
            </label>

            {/* Link to parent dropdown (only if not making a parent) */}
            {!isParent && parentTasks.length > 0 && (
              <Select value={selectedParentId} onValueChange={setSelectedParentId}>
                <SelectTrigger className="h-8 text-xs">
                  <SelectValue placeholder="Link to parent (optional)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">No parent</SelectItem>
                  {parentTasks.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.title}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}

            <div className="flex gap-2">
              <Button size="sm" onClick={handleAddTask} className="flex-1">
                Add
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  setIsAdding(false);
                  setIsParent(false);
                  setSelectedParentId("");
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-start text-muted-foreground"
            onClick={() => setIsAdding(true)}
          >
            <Plus className="h-4 w-4 mr-1" />
            Add task
          </Button>
        )}
      </div>
    </div>
  );
}
