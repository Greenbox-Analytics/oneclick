import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Plus, X, CornerDownRight, Search, Link } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { BoardTask, BoardTaskDetail } from "@/types/integrations";

interface TaskSubtasksProps {
  task: BoardTaskDetail;
  allBoardTasks: BoardTask[];
  onNavigateToTask?: (taskId: string) => void;
  updateTask: (data: { id: string; parent_task_id?: string | null }) => void;
  createTask: (data: { title: string; parent_task_id: string }) => void;
}

export function TaskSubtasks({
  task,
  allBoardTasks,
  onNavigateToTask,
  updateTask,
  createTask,
}: TaskSubtasksProps) {
  const [newSubtaskTitle, setNewSubtaskTitle] = useState("");
  const [showAllSubtasks, setShowAllSubtasks] = useState(false);
  const [linkSearchOpen, setLinkSearchOpen] = useState(false);
  const [linkSearch, setLinkSearch] = useState("");

  if (!task.is_parent) return null;

  return (
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
  );
}
