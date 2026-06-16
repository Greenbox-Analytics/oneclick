import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  icon?: LucideIcon;
  count?: number;
}

interface SegmentedProps<T extends string> {
  value: T;
  onChange: (v: T) => void;
  options: SegmentedOption<T>[];
  className?: string;
}

/** Pill-style segmented control used in the dashboard toolbar. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className,
}: SegmentedProps<T>) {
  return (
    <div
      role="tablist"
      className={cn(
        "inline-flex items-center gap-1 rounded-lg border bg-muted/40 p-1",
        className
      )}
    >
      {options.map((o) => {
        const active = value === o.value;
        const Icon = o.icon;
        return (
          <button
            key={o.value}
            role="tab"
            type="button"
            aria-selected={active}
            onClick={() => onChange(o.value)}
            className={cn(
              "inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
              active
                ? "bg-background text-primary shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {Icon && <Icon className="w-3.5 h-3.5" />}
            <span>{o.label}</span>
            {typeof o.count === "number" && (
              <span className="opacity-60 tabular-nums">{o.count}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
