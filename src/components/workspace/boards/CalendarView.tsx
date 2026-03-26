import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronLeft, ChevronRight, Loader2, Search, Plus } from "lucide-react";
import {
  startOfMonth,
  endOfMonth,
  startOfWeek,
  endOfWeek,
  startOfYear,
  endOfYear,
  startOfDay,
  endOfDay,
  eachDayOfInterval,
  eachMonthOfInterval,
  format,
  isSameMonth,
  isSameDay,
  isToday,
  addMonths,
  subMonths,
  addWeeks,
  subWeeks,
  addDays,
  subDays,
  addYears,
  subYears,
  getDay,
} from "date-fns";
import { cn } from "@/lib/utils";
import { useCalendarTasks } from "@/hooks/useCalendarTasks";
import { useBoards } from "@/hooks/useBoards";
import { useWorkspaceSettings } from "@/hooks/useWorkspaceSettings";
import { TaskDetailPanel } from "./TaskDetailPanel";
import type { BoardTask } from "@/types/integrations";

type CalendarViewMode = "day" | "week" | "month" | "year";

const DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MAX_VISIBLE_TASKS = 3;

export function CalendarView() {
  const { settings } = useWorkspaceSettings();
  const [currentDate, setCurrentDate] = useState(new Date());
  const [viewMode, setViewMode] = useState<CalendarViewMode>(settings?.calendar_view || "month");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);

  // Task creation dialog state
  const [createDate, setCreateDate] = useState<string | null>(null);
  const [newTaskTitle, setNewTaskTitle] = useState("");
  const [newTaskDescription, setNewTaskDescription] = useState("");
  const [newTaskPriority, setNewTaskPriority] = useState<string>("");
  const [newTaskColumnId, setNewTaskColumnId] = useState<string>("");
  const [newTaskLabels, setNewTaskLabels] = useState("");

  // Compute date range based on view mode
  const { rangeStart, rangeEnd } = useMemo(() => {
    switch (viewMode) {
      case "day":
        return {
          rangeStart: format(startOfDay(currentDate), "yyyy-MM-dd"),
          rangeEnd: format(endOfDay(currentDate), "yyyy-MM-dd"),
        };
      case "week": {
        const ws = startOfWeek(currentDate);
        const we = endOfWeek(currentDate);
        return {
          rangeStart: format(ws, "yyyy-MM-dd"),
          rangeEnd: format(we, "yyyy-MM-dd"),
        };
      }
      case "month": {
        const ms = startOfWeek(startOfMonth(currentDate));
        const me = endOfWeek(endOfMonth(currentDate));
        return {
          rangeStart: format(ms, "yyyy-MM-dd"),
          rangeEnd: format(me, "yyyy-MM-dd"),
        };
      }
      case "year": {
        const ys = startOfYear(currentDate);
        const ye = endOfYear(currentDate);
        return {
          rangeStart: format(ys, "yyyy-MM-dd"),
          rangeEnd: format(ye, "yyyy-MM-dd"),
        };
      }
    }
  }, [currentDate, viewMode]);

  const { tasks, isLoading } = useCalendarTasks(rangeStart, rangeEnd);
  const { columns, createTask } = useBoards();

  const handleCreateTask = () => {
    if (!newTaskTitle.trim() || !createDate) return;
    const columnId = newTaskColumnId || columns.find((c) => c.title === "To Do")?.id || columns[0]?.id;
    if (!columnId) return;
    createTask({
      title: newTaskTitle.trim(),
      description: newTaskDescription.trim() || undefined,
      priority: newTaskPriority || undefined,
      column_id: columnId,
      due_date: createDate,
      labels: newTaskLabels.trim() ? newTaskLabels.split(",").map((l) => l.trim()).filter(Boolean) : undefined,
    });
    setCreateDate(null);
    setNewTaskTitle("");
    setNewTaskDescription("");
    setNewTaskPriority("");
    setNewTaskColumnId("");
    setNewTaskLabels("");
  };

  // Group tasks by date
  const tasksByDate = useMemo(() => {
    const map = new Map<string, BoardTask[]>();
    const filtered = searchQuery
      ? tasks.filter((t) => t.title.toLowerCase().includes(searchQuery.toLowerCase()))
      : tasks;
    for (const task of filtered) {
      if (task.due_date) {
        if (!map.has(task.due_date)) map.set(task.due_date, []);
        map.get(task.due_date)!.push(task);
      }
      if (task.start_date && task.start_date !== task.due_date) {
        if (!map.has(task.start_date)) map.set(task.start_date, []);
        map.get(task.start_date)!.push(task);
      }
    }
    return map;
  }, [tasks, searchQuery]);

  // Navigation
  const goToPrev = () => {
    switch (viewMode) {
      case "day": setCurrentDate(subDays(currentDate, 1)); break;
      case "week": setCurrentDate(subWeeks(currentDate, 1)); break;
      case "month": setCurrentDate(subMonths(currentDate, 1)); break;
      case "year": setCurrentDate(subYears(currentDate, 1)); break;
    }
  };
  const goToNext = () => {
    switch (viewMode) {
      case "day": setCurrentDate(addDays(currentDate, 1)); break;
      case "week": setCurrentDate(addWeeks(currentDate, 1)); break;
      case "month": setCurrentDate(addMonths(currentDate, 1)); break;
      case "year": setCurrentDate(addYears(currentDate, 1)); break;
    }
  };
  const goToToday = () => setCurrentDate(new Date());

  const headerLabel = useMemo(() => {
    switch (viewMode) {
      case "day": return format(currentDate, "EEEE, MMMM d, yyyy");
      case "week": {
        const ws = startOfWeek(currentDate);
        const we = endOfWeek(currentDate);
        return `${format(ws, "MMM d")} – ${format(we, "MMM d, yyyy")}`;
      }
      case "month": return format(currentDate, "MMMM yyyy");
      case "year": return format(currentDate, "yyyy");
    }
  }, [currentDate, viewMode]);

  if (isLoading && tasks.length === 0) {
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
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="icon" onClick={goToPrev}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" onClick={goToNext}>
              <ChevronRight className="h-4 w-4" />
            </Button>
            <h3 className="text-xl font-semibold ml-2">{headerLabel}</h3>
          </div>
          <div className="flex items-center gap-2">
            {/* View mode toggle */}
            <div className="flex rounded-lg border overflow-hidden">
              {(["day", "week", "month", "year"] as CalendarViewMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={cn(
                    "px-3 py-1.5 text-xs font-medium transition-colors",
                    viewMode === mode
                      ? "bg-primary text-primary-foreground"
                      : "bg-background hover:bg-muted text-muted-foreground"
                  )}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
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

        {/* Views */}
        {viewMode === "month" && (
          <MonthGrid
            currentDate={currentDate}
            tasksByDate={tasksByDate}
            onTaskClick={setSelectedTaskId}
            onAddTask={setCreateDate}
          />
        )}
        {viewMode === "week" && (
          <WeekGrid
            currentDate={currentDate}
            tasksByDate={tasksByDate}
            onTaskClick={setSelectedTaskId}
            onAddTask={setCreateDate}
          />
        )}
        {viewMode === "day" && (
          <DayView
            currentDate={currentDate}
            tasksByDate={tasksByDate}
            onTaskClick={setSelectedTaskId}
            onAddTask={setCreateDate}
          />
        )}
        {viewMode === "year" && (
          <YearGrid
            currentDate={currentDate}
            tasksByDate={tasksByDate}
            onDayClick={(date) => {
              setCurrentDate(date);
              setViewMode("day");
            }}
          />
        )}
      </div>

      <TaskDetailPanel
        taskId={selectedTaskId}
        onClose={() => setSelectedTaskId(null)}
      />

      {/* Create task dialog */}
      <Dialog open={!!createDate} onOpenChange={(open) => !open && setCreateDate(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>
              Create Task — {createDate ? format(new Date(createDate + "T12:00:00"), "MMMM d, yyyy") : ""}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="cal-task-title">Title *</Label>
              <Input
                id="cal-task-title"
                placeholder="Task title..."
                value={newTaskTitle}
                onChange={(e) => setNewTaskTitle(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreateTask()}
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="cal-task-desc">Description</Label>
              <Textarea
                id="cal-task-desc"
                placeholder="Optional description..."
                value={newTaskDescription}
                onChange={(e) => setNewTaskDescription(e.target.value)}
                rows={2}
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Priority</Label>
                <Select value={newTaskPriority} onValueChange={setNewTaskPriority}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select priority" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="urgent">Urgent</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Column</Label>
                <Select value={newTaskColumnId} onValueChange={setNewTaskColumnId}>
                  <SelectTrigger>
                    <SelectValue placeholder="To Do" />
                  </SelectTrigger>
                  <SelectContent>
                    {columns.map((col) => (
                      <SelectItem key={col.id} value={col.id}>
                        {col.title}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="cal-task-labels">Labels (comma-separated)</Label>
              <Input
                id="cal-task-labels"
                placeholder="e.g. design, urgent"
                value={newTaskLabels}
                onChange={(e) => setNewTaskLabels(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDate(null)}>Cancel</Button>
            <Button onClick={handleCreateTask} disabled={!newTaskTitle.trim()}>Create Task</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

// --- Task pill component ---
function TaskPill({ task, onClick }: { task: BoardTask; onClick: (id: string) => void }) {
  return (
    <button
      onClick={() => onClick(task.id)}
      className={cn(
        "w-full text-left px-1.5 py-0.5 rounded text-[10px] leading-tight truncate",
        "hover:opacity-80 transition-opacity cursor-pointer",
        task.color ? "text-white" : "bg-primary/10 text-primary"
      )}
      style={task.color ? { backgroundColor: task.color } : undefined}
      title={task.title}
    >
      {task.title}
    </button>
  );
}

// --- Month View ---
function MonthGrid({
  currentDate,
  tasksByDate,
  onTaskClick,
  onAddTask,
}: {
  currentDate: Date;
  tasksByDate: Map<string, BoardTask[]>;
  onTaskClick: (id: string) => void;
  onAddTask: (dateKey: string) => void;
}) {
  const calendarDays = useMemo(() => {
    const monthStart = startOfMonth(currentDate);
    const monthEnd = endOfMonth(currentDate);
    return eachDayOfInterval({ start: startOfWeek(monthStart), end: endOfWeek(monthEnd) });
  }, [currentDate]);

  return (
    <>
      <div className="grid grid-cols-7 gap-px bg-border rounded-t-lg overflow-hidden">
        {DAY_NAMES.map((day) => (
          <div key={day} className="bg-muted px-2 py-2 text-center text-xs font-medium text-muted-foreground">
            {day}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-7 gap-px bg-border rounded-b-lg overflow-hidden -mt-4">
        {calendarDays.map((day) => {
          const dateKey = format(day, "yyyy-MM-dd");
          const dayTasks = tasksByDate.get(dateKey) || [];
          const isCurrentMonth = isSameMonth(day, currentDate);
          const today = isToday(day);
          const visible = dayTasks.slice(0, MAX_VISIBLE_TASKS);
          const overflow = dayTasks.length - MAX_VISIBLE_TASKS;

          return (
            <div
              key={dateKey}
              className={cn("group bg-background min-h-[100px] p-1.5 transition-colors", !isCurrentMonth && "bg-muted/30")}
            >
              <div className="flex items-center justify-between mb-1">
                <span className={cn(
                  "text-xs font-medium w-6 h-6 flex items-center justify-center rounded-full",
                  !isCurrentMonth && "text-muted-foreground/50",
                  today && "bg-primary text-primary-foreground"
                )}>
                  {format(day, "d")}
                </span>
                {isCurrentMonth && (
                  <button
                    onClick={() => onAddTask(dateKey)}
                    className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                    title="Add task"
                  >
                    <Plus className="h-3 w-3" />
                  </button>
                )}
              </div>
              <div className="space-y-0.5">
                {visible.map((task) => (
                  <TaskPill key={task.id} task={task} onClick={onTaskClick} />
                ))}
                {overflow > 0 && (
                  <p className="text-[10px] text-muted-foreground px-1.5">+{overflow} more</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

// --- Week View ---
function WeekGrid({
  currentDate,
  tasksByDate,
  onTaskClick,
  onAddTask,
}: {
  currentDate: Date;
  tasksByDate: Map<string, BoardTask[]>;
  onTaskClick: (id: string) => void;
  onAddTask: (dateKey: string) => void;
}) {
  const weekDays = useMemo(() => {
    const ws = startOfWeek(currentDate);
    const we = endOfWeek(currentDate);
    return eachDayOfInterval({ start: ws, end: we });
  }, [currentDate]);

  return (
    <>
      <div className="grid grid-cols-7 gap-px bg-border rounded-t-lg overflow-hidden">
        {weekDays.map((day) => (
          <div key={day.toISOString()} className="bg-muted px-2 py-2 text-center text-xs font-medium text-muted-foreground">
            {format(day, "EEE d")}
          </div>
        ))}
      </div>
      <div className="grid grid-cols-7 gap-px bg-border rounded-b-lg overflow-hidden -mt-4">
        {weekDays.map((day) => {
          const dateKey = format(day, "yyyy-MM-dd");
          const dayTasks = tasksByDate.get(dateKey) || [];
          const today = isToday(day);

          return (
            <div
              key={dateKey}
              className="group bg-background min-h-[200px] p-2 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <span className={cn(
                  "text-sm font-medium w-7 h-7 flex items-center justify-center rounded-full",
                  today && "bg-primary text-primary-foreground"
                )}>
                  {format(day, "d")}
                </span>
                <button
                  onClick={() => onAddTask(dateKey)}
                  className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Add task"
                >
                  <Plus className="h-3 w-3" />
                </button>
              </div>
              <div className="space-y-1">
                {dayTasks.map((task) => (
                  <TaskPill key={task.id} task={task} onClick={onTaskClick} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

// --- Day View ---
function DayView({
  currentDate,
  tasksByDate,
  onTaskClick,
  onAddTask,
}: {
  currentDate: Date;
  tasksByDate: Map<string, BoardTask[]>;
  onTaskClick: (id: string) => void;
  onAddTask: (dateKey: string) => void;
}) {
  const dateKey = format(currentDate, "yyyy-MM-dd");
  const dayTasks = tasksByDate.get(dateKey) || [];
  const today = isToday(currentDate);

  return (
    <div className="bg-background border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className={cn(
            "text-lg font-semibold w-9 h-9 flex items-center justify-center rounded-full",
            today && "bg-primary text-primary-foreground"
          )}>
            {format(currentDate, "d")}
          </span>
          <span className="text-muted-foreground">{format(currentDate, "EEEE")}</span>
        </div>
        <Button variant="outline" size="sm" onClick={() => onAddTask(dateKey)}>
          <Plus className="h-4 w-4 mr-1" />
          Add Task
        </Button>
      </div>

      {dayTasks.length === 0 ? (
        <div className="text-center py-12 text-muted-foreground">
          <p>No tasks scheduled for this day</p>
        </div>
      ) : (
        <div className="space-y-2">
          {dayTasks.map((task) => (
            <button
              key={task.id}
              onClick={() => onTaskClick(task.id)}
              className={cn(
                "w-full text-left px-4 py-3 rounded-lg border transition-colors hover:bg-muted/50 cursor-pointer",
                "flex items-center gap-3"
              )}
            >
              {task.color && (
                <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: task.color }} />
              )}
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate">{task.title}</p>
                {task.description && (
                  <p className="text-xs text-muted-foreground truncate">{task.description}</p>
                )}
              </div>
              {task.priority && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground shrink-0">
                  {task.priority}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Year View ---
function YearGrid({
  currentDate,
  tasksByDate,
  onDayClick,
}: {
  currentDate: Date;
  tasksByDate: Map<string, BoardTask[]>;
  onDayClick: (date: Date) => void;
}) {
  const months = useMemo(() => {
    return eachMonthOfInterval({
      start: startOfYear(currentDate),
      end: endOfYear(currentDate),
    });
  }, [currentDate]);

  return (
    <div className="grid grid-cols-3 md:grid-cols-4 gap-4">
      {months.map((monthDate) => {
        const monthStart = startOfMonth(monthDate);
        const monthEnd = endOfMonth(monthDate);
        const days = eachDayOfInterval({ start: monthStart, end: monthEnd });
        const firstDayOfWeek = getDay(monthStart); // 0=Sun

        return (
          <div key={monthDate.toISOString()} className="bg-background border rounded-lg p-3">
            <h4 className="text-xs font-semibold mb-2 text-center">{format(monthDate, "MMMM")}</h4>
            <div className="grid grid-cols-7 gap-0.5">
              {/* Day name headers */}
              {["S", "M", "T", "W", "T", "F", "S"].map((d, i) => (
                <div key={i} className="text-[8px] text-muted-foreground text-center">{d}</div>
              ))}
              {/* Empty cells for offset */}
              {Array.from({ length: firstDayOfWeek }).map((_, i) => (
                <div key={`empty-${i}`} />
              ))}
              {/* Day cells */}
              {days.map((day) => {
                const dateKey = format(day, "yyyy-MM-dd");
                const hasTasks = tasksByDate.has(dateKey);
                const today = isToday(day);

                return (
                  <button
                    key={dateKey}
                    onClick={() => onDayClick(day)}
                    className={cn(
                      "w-full aspect-square flex items-center justify-center text-[9px] rounded-sm relative",
                      "hover:bg-muted transition-colors cursor-pointer",
                      today && "font-bold text-primary"
                    )}
                    title={hasTasks ? `${tasksByDate.get(dateKey)!.length} task(s)` : undefined}
                  >
                    {format(day, "d")}
                    {hasTasks && (
                      <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-primary" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
