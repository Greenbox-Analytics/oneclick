import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface StatCardProps {
  icon: LucideIcon;
  num: number;
  label: string;
  tint: { bg: string; fg: string };
  active?: boolean;
  onClick?: () => void;
}

/**
 * Clickable dashboard stat tile. Goes from muted to bordered-primary when
 * `active` to mirror the prototype's stat-card filter UX.
 */
export function StatCard({ icon: Icon, num, label, tint, active, onClick }: StatCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-lg border bg-card text-card-foreground shadow-sm p-4 text-left w-full",
        "cursor-pointer transition-all hover:border-primary/40 hover:bg-muted/40",
        active && "border-primary ring-1 ring-primary/30 bg-muted/40"
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <div
          className={cn(
            "w-9 h-9 rounded-lg flex items-center justify-center",
            tint.bg,
            tint.fg
          )}
        >
          <Icon className="w-[18px] h-[18px]" />
        </div>
      </div>
      <div className="text-3xl font-bold tracking-tight">{num}</div>
      <div className="text-xs text-muted-foreground mt-1">{label}</div>
    </button>
  );
}
