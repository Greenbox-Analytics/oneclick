import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Calendar, MoreHorizontal, Trash2, ExternalLink, Users, CornerDownRight, AlertCircle } from "lucide-react";
import { format } from "date-fns";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { getLabelColor } from "./labelColors";
import type { BoardTask } from "@/types/integrations";

interface KanbanCardProps {
  task: BoardTask;
  onDelete: (taskId: string) => void;
  onClick: (taskId: string) => void;
}

const PRIORITY_COLORS: Record<string, string> = {
  low: "bg-slate-100 text-slate-700",
  medium: "bg-blue-100 text-blue-700",
  high: "bg-orange-100 text-orange-700",
  urgent: "bg-red-100 text-red-700",
};

export function KanbanCard({ task, onDelete, onClick }: KanbanCardProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: task.id, data: { type: "task", task } });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0 : 1,
    borderLeftColor: task.color || "transparent",
    borderLeftWidth: task.color ? 3 : 1,
  };

  const isDueToday = task.due_date === format(new Date(), "yyyy-MM-dd");

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return null;
    return new Date(dateStr).toLocaleDateString("en-US", { month: "short", day: "numeric" });
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
      className="cursor-grab active:cursor-grabbing hover:shadow-sm transition-shadow"
      onClick={() => onClick(task.id)}
      {...attributes}
      {...listeners}
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
}
