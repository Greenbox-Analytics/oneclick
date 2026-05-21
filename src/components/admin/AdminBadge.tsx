import { Shield } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAdminStatus } from "@/hooks/useAdminStatus";

/**
 * Compact "Admin" badge for inline placement next to the plan header.
 * Renders null for non-admins. Tooltip explains the implicit-Pro behavior
 * so an admin who sees "Free plan" but no paywalls understands why.
 */
export function AdminBadge() {
  const isAdmin = useAdminStatus();
  if (!isAdmin) return null;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge
          variant="outline"
          className="gap-1 border-primary/40 text-primary"
        >
          <Shield className="w-3 h-3" />
          Admin
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        <div className="text-xs max-w-xs">
          Admins have unlimited access to all Pro features regardless of plan.
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
