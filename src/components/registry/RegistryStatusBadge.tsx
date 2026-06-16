import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const STATUS_COLORS: Record<string, string> = {
  draft: "bg-gray-500/15 text-gray-400 border-0",
  pending_approval: "bg-amber-500/15 text-amber-400 border-0",
  pending: "bg-amber-500/15 text-amber-400 border-0",
  registered: "bg-emerald-500/15 text-emerald-400 border-0",
};

const STATUS_LABELS: Record<string, string> = {
  draft: "Draft",
  pending_approval: "Pending",
  pending: "Pending",
  registered: "Registered",
};

interface RegistryStatusBadgeProps {
  status: string;
  className?: string;
}

/** DB-status-aware pill. Maps `pending_approval` → "Pending" on display only. */
export function RegistryStatusBadge({ status, className }: RegistryStatusBadgeProps) {
  return (
    <Badge className={cn(STATUS_COLORS[status] || STATUS_COLORS.draft, "text-[11px]", className)}>
      {STATUS_LABELS[status] || status}
    </Badge>
  );
}
