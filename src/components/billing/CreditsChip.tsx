// src/components/billing/CreditsChip.tsx
// Tiny inline "{n} credits" affordance rendered next to metered actions (Zoe
// composer, OneClick run, Registry contract parse) so credits are visible
// BEFORE a wall. Reads the cached entitlements — renders nothing while
// loading or when the credits system is off.
import { Coins } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEntitlements } from "@/hooks/useEntitlements";
import { creditStanding, POOL_ONLY_LABEL } from "@/lib/credits";

export function CreditsChip({ className }: { className?: string }) {
  const { data: ent } = useEntitlements();
  const credits = ent?.credits;
  if (!credits) return null;
  // null = an uncapped org member: no cap of their own, and the pool balance is
  // admin-only, so there is no number to put here.
  const standing = creditStanding(credits);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-border bg-muted/50 px-2 py-0.5 text-[11px] text-muted-foreground",
        standing ? "tabular-nums" : "",
        className,
      )}
      title={standing ? "Credits you have left this month" : POOL_ONLY_LABEL}
    >
      <Coins className="w-3 h-3" />
      {standing ? `${standing.remaining.toLocaleString()} credits` : "Org credits"}
    </span>
  );
}
