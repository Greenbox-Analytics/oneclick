import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Pencil, ListTree } from "lucide-react";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { getLabelColor } from "./labelColors";
import type { ParentTaskWithChildren } from "@/types/integrations";

const PRIORITY_COLORS: Record<string, string> = {
  low: "bg-slate-100 text-slate-700",
  medium: "bg-blue-100 text-blue-700",
  high: "bg-orange-100 text-orange-700",
  urgent: "bg-red-100 text-red-700",
};

interface ParentKanbanCardProps {
  parent: ParentTaskWithChildren;
  onClick: (taskId: string) => void;
}

export function ParentKanbanCard({ parent, onClick }: ParentKanbanCardProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: parent.id, data: { type: "parent-task", parent } });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0 : 1,
    borderLeftColor: parent.color || "transparent",
    borderLeftWidth: parent.color ? 3 : 1,
  };

  const childCount = parent.children?.length || parent.child_count || 0;

  return (
    <Card
      ref={setNodeRef}
      style={style}
      className="cursor-grab active:cursor-grabbing hover:shadow-sm transition-shadow"
      {...attributes}
      {...listeners}
    >
      <CardContent className="p-3 space-y-2">
        <div className="flex items-start justify-between gap-2">
          <p className="text-sm font-medium leading-snug">{parent.title}</p>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 shrink-0"
            onClick={(e) => {
              e.stopPropagation();
              onClick(parent.id);
            }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <Pencil className="h-3.5 w-3.5" />
          </Button>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
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
              <Badge
                key={label}
                variant="secondary"
                className={`text-[10px] px-1.5 py-0 ${lc.bg} ${lc.text}`}
              >
                {label}
              </Badge>
            );
          })}
        </div>

        {childCount > 0 && (
          <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <ListTree className="h-3 w-3" />
            {childCount} {childCount === 1 ? "subtask" : "subtasks"}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
