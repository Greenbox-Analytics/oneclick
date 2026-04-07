import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  ChevronDown,
  ChevronRight,
  Plus,
  Calendar,
  CornerDownRight,
  Trash2,
  Pencil,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { getLabelColor } from "./labelColors";
import type { ParentTaskWithChildren, BoardTask } from "@/types/integrations";

const PRIORITY_COLORS: Record<string, string> = {
  low: "bg-slate-100 text-slate-700",
  medium: "bg-blue-100 text-blue-700",
  high: "bg-orange-100 text-orange-700",
  urgent: "bg-red-100 text-red-700",
};

interface ParentTaskRowProps {
  parent: ParentTaskWithChildren;
  onTaskClick: (taskId: string) => void;
  onParentClick: (taskId: string) => void;
  onAddSubtask: (parentId: string, title: string) => void;
  onDelete: (taskId: string) => void;
}

export const ParentTaskRow = React.memo(function ParentTaskRow({
  parent,
  onTaskClick,
  onParentClick,
  onAddSubtask,
  onDelete,
}: ParentTaskRowProps) {
  const [expanded, setExpanded] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [newTitle, setNewTitle] = useState("");

  const children = parent.children || [];
  const doneCount = children.filter(
    (c) => c.column_title?.toLowerCase() === "done"
  ).length;
  const totalCount = children.length;
  const progressPercent = totalCount > 0 ? (doneCount / totalCount) * 100 : 0;

  const handleAddSubtask = () => {
    if (!newTitle.trim()) return;
    onAddSubtask(parent.id, newTitle.trim());
    setNewTitle("");
    setIsAdding(false);
  };

  const formatDate = (d?: string) =>
    d ? new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : null;

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* Parent header */}
      <div
        className="flex items-center gap-3 px-4 py-3 hover:bg-muted/50 transition-colors cursor-pointer"
        onClick={() => setExpanded(!expanded)}
        style={{
          borderLeftColor: parent.color || "transparent",
          borderLeftWidth: parent.color ? 4 : 0,
        }}
      >
        {expanded ? (
          <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
        )}

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <button
              className="font-medium text-sm hover:underline text-left"
              onClick={(e) => {
                e.stopPropagation();
                onParentClick(parent.id);
              }}
            >
              {parent.title}
            </button>
            {parent.priority && (
              <Badge
                variant="secondary"
                className={`text-[10px] px-1.5 py-0 ${PRIORITY_COLORS[parent.priority] || ""}`}
              >
                {parent.priority}
              </Badge>
            )}
            {parent.labels?.map((label) => {
              const lc = getLabelColor(label);
              return (
                <Badge key={label} variant="secondary" className={`text-[10px] px-1.5 py-0 ${lc.bg} ${lc.text}`}>
                  {label}
                </Badge>
              );
            })}
            {parent.artists?.map((a) => (
              <Badge key={a.id} variant="secondary" className="text-[10px] px-1.5 py-0">
                {a.name}
              </Badge>
            ))}
          </div>

          {/* Progress + dates */}
          <div className="flex items-center gap-3 mt-1.5">
            {totalCount > 0 && (
              <div className="flex items-center gap-2 min-w-[120px]">
                <Progress value={progressPercent} className="h-1.5 flex-1" />
                <span className="text-[10px] text-muted-foreground whitespace-nowrap">
                  {doneCount}/{totalCount}
                </span>
              </div>
            )}
            {(parent.start_date || parent.due_date) && (
              <span className="text-[10px] text-muted-foreground flex items-center gap-0.5">
                <Calendar className="h-3 w-3" />
                {formatDate(parent.start_date)}
                {parent.start_date && parent.due_date && " – "}
                {formatDate(parent.due_date)}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {parent.column_title && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
              {parent.column_title}
            </Badge>
          )}
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {totalCount} {totalCount === 1 ? "task" : "tasks"}
          </span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-primary"
            onClick={(e) => {
              e.stopPropagation();
              onParentClick(parent.id);
            }}
          >
            <Pencil className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-muted-foreground hover:text-destructive"
            onClick={(e) => {
              e.stopPropagation();
              onDelete(parent.id);
            }}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Children */}
      {expanded && (
        <div className="border-t bg-muted/20">
          {children.length === 0 && !isAdding && (
            <p className="text-sm text-muted-foreground text-center py-4">
              No subtasks yet
            </p>
          )}
          {(showAll ? children : children.slice(0, 5)).map((child) => (
            <ChildRow key={child.id} child={child} onClick={onTaskClick} />
          ))}
          {children.length > 5 && (
            <button
              className="w-full text-xs text-muted-foreground hover:text-foreground py-2 border-t transition-colors"
              onClick={(e) => {
                e.stopPropagation();
                setShowAll(!showAll);
              }}
            >
              {showAll ? "Show less" : `Show ${children.length - 5} more`}
            </button>
          )}

          {/* Add subtask */}
          <div className="px-4 py-2 border-t">
            {isAdding ? (
              <div className="flex gap-2 items-center pl-6">
                <Input
                  placeholder="Subtask title..."
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAddSubtask();
                    if (e.key === "Escape") setIsAdding(false);
                  }}
                  autoFocus
                  className="h-8"
                />
                <Button size="sm" onClick={handleAddSubtask} className="h-8">
                  Add
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => setIsAdding(false)}
                  className="h-8"
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground pl-6"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsAdding(true);
                }}
              >
                <Plus className="h-3.5 w-3.5 mr-1" />
                Add subtask
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

function ChildRow({
  child,
  onClick,
}: {
  child: BoardTask;
  onClick: (taskId: string) => void;
}) {
  const formatDate = (d?: string) =>
    d ? new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : null;

  return (
    <button
      className="w-full flex items-center gap-3 px-4 py-2.5 hover:bg-muted/50 transition-colors text-left border-t"
      onClick={() => onClick(child.id)}
    >
      <CornerDownRight className="h-3.5 w-3.5 text-muted-foreground shrink-0 ml-2" />
      {child.color && (
        <div
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: child.color }}
        />
      )}
      <span className="text-sm flex-1 truncate">{child.title}</span>
      {child.priority && (
        <Badge
          variant="secondary"
          className={cn("text-[10px] px-1.5 py-0", PRIORITY_COLORS[child.priority] || "")}
        >
          {child.priority}
        </Badge>
      )}
      {child.column_title && (
        <span className="text-[10px] text-muted-foreground bg-muted rounded px-1.5 py-0.5">
          {child.column_title}
        </span>
      )}
      {child.due_date && (
        <span className="text-[10px] text-muted-foreground flex items-center gap-0.5">
          <Calendar className="h-3 w-3" />
          {formatDate(child.due_date)}
        </span>
      )}
    </button>
  );
}
