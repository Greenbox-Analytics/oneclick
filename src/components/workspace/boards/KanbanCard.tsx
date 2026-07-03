import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Calendar, MoreHorizontal, Trash2, ExternalLink, Users, CornerDownRight, AlertCircle, ArrowRightLeft } from "lucide-react";
import { parseDateString, getTodayString } from "@/lib/dateUtils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { getLabelColor } from "./labelColors";
import type { BoardTask } from "@/types/integrations";

interface MoveOption {
  id: string;
  title: string;
}

interface KanbanCardProps {
  task: BoardTask;
  onDelete: (taskId: string) => void;
  onClick: (taskId: string) => void;
  timezone?: string;
  disableDrag?: boolean;
  moveOptions?: MoveOption[];
  onMove?: (taskId: string, targetColumnId: string) => void;
}

function initials(name?: string | null): string {
  if (!name) return "?";
  return (
    name
      .split(" ")
      .map((w) => w[0])
      .filter(Boolean)
      .slice(0, 2)
      .join("")
      .toUpperCase() || "?"
  );
}

const PRIORITY_COLORS: Record<string, string> = {
  low: "bg-slate-100 text-slate-700",
  medium: "bg-blue-100 text-blue-700",
  high: "bg-orange-100 text-orange-700",
  urgent: "bg-red-100 text-red-700",
};

export const KanbanCard = React.memo(function KanbanCard({ task, onDelete, onClick, timezone, disableDrag, moveOptions, onMove }: KanbanCardProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: task.id, data: { type: "task", task }, disabled: !!disableDrag });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0 : 1,
    borderLeftColor: task.color || "transparent",
    borderLeftWidth: task.color ? 3 : 1,
  };

  const dragProps = disableDrag ? {} : { ...attributes, ...listeners };
  const cursorClass = disableDrag ? "cursor-pointer" : "cursor-grab active:cursor-grabbing";

  const today = getTodayString(timezone);
  const isDueToday = task.due_date === today;
  const isOverdue = !!(task.due_date && task.due_date < today && !task.completed_at);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return null;
    return parseDateString(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const dateDisplay = (() => {
    const start = formatDate(task.start_date);
    const due = formatDate(task.due_date);
    if (start && due) return `${start} – ${due}`;
    if (due) return due;
    if (start) return `From ${start}`;
    return null;
  })();

  return (
    <Card
      ref={setNodeRef}
      style={style}
      className={`${cursorClass} hover:shadow-sm transition-shadow`}
      onClick={() => onClick(task.id)}
      {...dragProps}
    >
      <CardContent className="p-3 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            {task.parent_title && (
              <p className="text-[10px] text-muted-foreground flex items-center gap-0.5 mb-0.5">
                <CornerDownRight className="h-2.5 w-2.5" />
                {task.parent_title}
              </p>
            )}
            <p className="text-sm font-medium leading-snug">{task.title}</p>
          </div>
          {isDueToday && (
            <Badge className="bg-red-500 text-white text-[10px] px-1.5 py-0 shrink-0 hover:bg-red-600">
              <AlertCircle className="h-3 w-3 mr-0.5" />
              Due Today
            </Badge>
          )}
          {isOverdue && (
            <Badge className="bg-orange-500 text-white text-[10px] px-1.5 py-0 shrink-0 hover:bg-orange-600">
              <AlertCircle className="h-3 w-3 mr-0.5" />
              Overdue
            </Badge>
          )}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 shrink-0"
                onClick={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
              >
                <MoreHorizontal className="h-3.5 w-3.5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {moveOptions && moveOptions.length > 0 && onMove && (
                <>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger>
                      <ArrowRightLeft className="h-4 w-4 mr-2" />
                      Move to...
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent>
                      <DropdownMenuLabel className="text-xs text-muted-foreground">
                        Column
                      </DropdownMenuLabel>
                      {moveOptions.map((opt) => (
                        <DropdownMenuItem
                          key={opt.id}
                          onClick={(e) => {
                            e.stopPropagation();
                            onMove(task.id, opt.id);
                          }}
                        >
                          {opt.title}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                  <DropdownMenuSeparator />
                </>
              )}
              {task.external_url && (
                <DropdownMenuItem
                  onClick={(e) => {
                    e.stopPropagation();
                    window.open(task.external_url, "_blank");
                  }}
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Open in {task.external_provider || "external"}
                </DropdownMenuItem>
              )}
              <DropdownMenuItem
                className="text-destructive"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(task.id);
                }}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {task.description && (
          <p className="text-xs text-muted-foreground line-clamp-2">
            {task.description}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-1.5">
          {task.priority && (
            <Badge variant="secondary" className={`text-[10px] px-1.5 py-0 ${PRIORITY_COLORS[task.priority] || ""}`}>
              {task.priority}
            </Badge>
          )}
          {task.labels?.map((label) => {
            const lc = getLabelColor(label);
            return (
              <Badge key={label} variant="secondary" className={`text-[10px] px-1.5 py-0 ${lc.bg} ${lc.text}`}>
                {label}
              </Badge>
            );
          })}
        </div>

        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          {dateDisplay && (
            <span className="flex items-center gap-0.5">
              <Calendar className="h-3 w-3" />
              {dateDisplay}
            </span>
          )}
          {task.assignees && task.assignees.length > 0 && (
            <div className="flex items-center -space-x-1.5">
              {task.assignees.slice(0, 3).map((a) => (
                <Avatar key={a.user_id} className="h-5 w-5 border border-background">
                  <AvatarImage src={a.avatar_url ?? undefined} />
                  <AvatarFallback className="text-[9px]">
                    {initials(a.full_name)}
                  </AvatarFallback>
                </Avatar>
              ))}
              {task.assignees.length > 3 && (
                <span className="text-[10px] text-muted-foreground pl-2">
                  +{task.assignees.length - 3}
                </span>
              )}
            </div>
          )}
          {(task.artist_ids?.length || 0) > 0 && (
            <span className="flex items-center gap-0.5 ml-auto">
              <Users className="h-3 w-3" />
              {task.artist_ids!.length}
            </span>
          )}
          {task.external_provider && (
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0 ml-auto">
              Synced
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
});
