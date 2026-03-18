import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronLeft, ChevronRight, Loader2, Search } from "lucide-react";
import {
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  eachDayOfInterval,
  format,
  isSameMonth,
  isToday,
  addMonths,
  subMonths,
} from "date-fns";
import { cn } from "@/lib/utils";
import { useCalendarTasks } from "@/hooks/useCalendarTasks";
import { TaskDetailPanel } from "./TaskDetailPanel";
import type { BoardTask } from "@/types/integrations";

const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MAX_VISIBLE_TASKS = 3;

export function CalendarView() {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();
  const { tasks, isLoading } = useCalendarTasks(year, month);

  // Build the calendar grid days
  const calendarDays = useMemo(() => {
    const monthStart = startOfMonth(currentDate);
    const monthEnd = endOfMonth(currentDate);
    const gridStart = startOfWeek(monthStart);
    const gridEnd = endOfWeek(monthEnd);
    return eachDayOfInterval({ start: gridStart, end: gridEnd });
  }, [currentDate]);

  // Group tasks by date
  const tasksByDate = useMemo(() => {
    const map = new Map<string, BoardTask[]>();
    const filtered = searchQuery
      ? tasks.filter((t) =>
          t.title.toLowerCase().includes(searchQuery.toLowerCase())
        )
      : tasks;

    for (const task of filtered) {
      if (task.due_date) {
        const key = task.due_date;
        if (!map.has(key)) map.set(key, []);
        map.get(key)!.push(task);
      }
      // Also show on start_date if different from due_date
      if (task.start_date && task.start_date !== task.due_date) {
        const key = task.start_date;
        if (!map.has(key)) map.set(key, []);
        map.get(key)!.push(task);
      }
    }
    return map;
  }, [tasks, searchQuery]);

  const goToPrev = () => setCurrentDate(subMonths(currentDate, 1));
  const goToNext = () => setCurrentDate(addMonths(currentDate, 1));
  const goToToday = () => setCurrentDate(new Date());

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={goToPrev}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" onClick={goToNext}>
              <ChevronRight className="h-4 w-4" />
            </Button>
            <h3 className="text-xl font-semibold ml-2">
              {format(currentDate, "MMMM yyyy")}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search tasks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 w-[200px] h-9"
              />
            </div>
            <Button variant="outline" size="sm" onClick={goToToday}>
              Today
            </Button>
          </div>
        </div>

        {/* Day headers */}
        <div className="grid grid-cols-7 gap-px bg-border rounded-t-lg overflow-hidden">
          {DAY_NAMES.map((day) => (
            <div
              key={day}
              className="bg-muted px-2 py-2 text-center text-xs font-medium text-muted-foreground"
            >
              {day}
            </div>
          ))}
        </div>

        {/* Calendar grid */}
        <div className="grid grid-cols-7 gap-px bg-border rounded-b-lg overflow-hidden -mt-4">
          {calendarDays.map((day) => {
            const dateKey = format(day, "yyyy-MM-dd");
            const dayTasks = tasksByDate.get(dateKey) || [];
            const isCurrentMonth = isSameMonth(day, currentDate);
            const today = isToday(day);
            const visibleTasks = dayTasks.slice(0, MAX_VISIBLE_TASKS);
            const overflow = dayTasks.length - MAX_VISIBLE_TASKS;

            return (
              <div
                key={dateKey}
                className={cn(
                  "bg-background min-h-[100px] p-1.5 transition-colors",
                  !isCurrentMonth && "bg-muted/30"
                )}
              >
                {/* Day number */}
                <div className="flex items-center justify-between mb-1">
                  <span
                    className={cn(
                      "text-xs font-medium w-6 h-6 flex items-center justify-center rounded-full",
                      !isCurrentMonth && "text-muted-foreground/50",
                      today && "bg-primary text-primary-foreground"
                    )}
                  >
                    {format(day, "d")}
                  </span>
                </div>

                {/* Task pills */}
                <div className="space-y-0.5">
                  {visibleTasks.map((task) => (
                    <button
                      key={task.id}
                      onClick={() => setSelectedTaskId(task.id)}
                      className={cn(
                        "w-full text-left px-1.5 py-0.5 rounded text-[10px] leading-tight truncate",
                        "hover:opacity-80 transition-opacity cursor-pointer",
                        task.color
                          ? "text-white"
                          : "bg-primary/10 text-primary"
                      )}
                      style={
                        task.color
                          ? { backgroundColor: task.color }
                          : undefined
                      }
                      title={task.title}
                    >
                      {task.title}
                    </button>
                  ))}
                  {overflow > 0 && (
                    <p className="text-[10px] text-muted-foreground px-1.5">
                      +{overflow} more
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Task detail side panel */}
      <TaskDetailPanel
        taskId={selectedTaskId}
        onClose={() => setSelectedTaskId(null)}
      />
    </>
  );
}
