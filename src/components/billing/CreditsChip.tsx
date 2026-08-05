// src/components/billing/CreditsChip.tsx
// Tiny inline "{n} credits" affordance rendered next to metered actions (Zoe
// composer, OneClick run, Registry contract parse) so credits are visible
// BEFORE a wall. Reads the cached entitlements — renders nothing while
// loading or when the credits system is off.
import { Coins } from "lucide-react";
import { cn } from "@/lib/utils";
import { useEntitlements } from "@/hooks/useEntitlements";

export function CreditsChip({ className }: { className?: string }) {
  const { data: ent } = useEntitlements();
  const credits = ent?.credits;
  if (!credits) return null;
  // Org members are bounded by their monthly cap on the shared pool; everyone
  // else by their wallet balance. Mirrors CreditsUsageCard's remaining calc.
  const remaining =
    credits.memberCap != null
      ? Math.max(0, credits.memberCap - (credits.memberCapUsed ?? 0))
      : credits.balance;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-border bg-muted/50 px-2 py-0.5 text-[11px] text-muted-foreground tabular-nums",
        className,
      )}
      title="Credits you have left this month"
    >
      <Coins className="w-3 h-3" />
      {remaining.toLocaleString()} credits
    </span>
  );
}
