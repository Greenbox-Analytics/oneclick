import { Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useTesterStatus } from "@/hooks/useTesterStatus";

function formatIso(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return "—";
  }
}

/**
 * Compact "Tester" badge for inline placement (e.g. Subscription page,
 * profile header). Renders nothing for non-testers.
 */
export function TesterBadge() {
  const { isTester, grantedAt, expiresAt } = useTesterStatus();
  if (!isTester) return null;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="secondary" className="gap-1">
          <Sparkles className="w-3 h-3" />
          Tester
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        <div className="text-xs">
          <div>Granted: {formatIso(grantedAt)}</div>
          <div>Expires: {expiresAt ? formatIso(expiresAt) : "never"}</div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}
