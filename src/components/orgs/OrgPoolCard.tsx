// src/components/orgs/OrgPoolCard.tsx
// Admin console: the org's shared credit pool — balance, cumulative
// purchased, and the "buy N more to activate" banner while pending (plan
// Task 12). Buying credits reuses the Phase A TopUpCreditsDialog with an org
// target instead of duplicating the pack picker.
import { useState } from "react";
import { Plus, Coins } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { TopUpCreditsDialog } from "@/components/billing/TopUpCreditsDialog";
import type { OrgDetail } from "@/hooks/useOrgs";

export function OrgPoolCard({ org }: { org: OrgDetail }) {
  const [topUpOpen, setTopUpOpen] = useState(false);
  const pending = org.status === "pending";

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-3.5">
        <div>
          <div className="flex items-center gap-2.5 text-[15px] font-semibold">
            <Coins className="w-[18px] h-[18px] text-muted-foreground" />
            Credit pool
          </div>
          <div className="text-[13.5px] text-muted-foreground mt-0.5">
            Shared credits your team draws from
          </div>
        </div>
        <Button size="sm" className="gap-1.5" onClick={() => setTopUpOpen(true)}>
          <Plus className="w-3.5 h-3.5" />
          Buy credits
        </Button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
        <div className="bg-background border border-border rounded-xl px-[18px] py-4">
          <div className="text-[12.5px] text-muted-foreground">Pool balance</div>
          <div className="text-[28px] font-bold tracking-tight mt-1 tabular-nums">
            {org.pool_balance.toLocaleString()}{" "}
            <span className="text-sm font-normal text-muted-foreground">credits</span>
          </div>
        </div>
        <div className="bg-background border border-border rounded-xl px-[18px] py-4">
          <div className="text-[12.5px] text-muted-foreground">Total purchased</div>
          <div className="text-[28px] font-bold tracking-tight mt-1 tabular-nums">
            {org.cumulative_purchased.toLocaleString()}{" "}
            <span className="text-sm font-normal text-muted-foreground">credits</span>
          </div>
        </div>
      </div>

      {pending && (
        <div className="mt-4 px-4 py-3.5 border border-amber-500/30 bg-amber-500/10 rounded-xl">
          <div className="text-[13px] font-medium text-amber-800 dark:text-amber-400">
            Buy {org.remaining_to_activate.toLocaleString()} more credit
            {org.remaining_to_activate === 1 ? "" : "s"} to activate
          </div>
          <p className="text-[12.5px] text-amber-800/80 dark:text-amber-400/80 mt-0.5 max-w-[520px]">
            Seats and enterprise features turn on automatically once your total purchases reach the
            minimum — no separate step needed.
          </p>
        </div>
      )}

      <TopUpCreditsDialog open={topUpOpen} onOpenChange={setTopUpOpen} orgId={org.id} orgName={org.name} />
    </Card>
  );
}
