// src/components/orgs/OrgBillingPanel.tsx
// The org billing console's money view (spec §6, Task 15): the pool's
// activity feed, recurring monthly top-up management, a transfer-in entry
// point, and the team storage meter with its PAYG overage projection.
// Mirrors CreditsUsageCard's card/meter styling.
//
// ADMIN ONLY. Every figure here comes off GET /orgs/{id}'s admin-only payload
// or the admin-only /ledger endpoint, so this must stay behind the
// AdminConsole's self_serve branch — same stance as OrgPoolCard beside it.
import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  CalendarClock,
  Clock,
  Gift,
  HardDrive,
  Loader2,
  Receipt,
  RefreshCcw,
  ShoppingCart,
  Undo2,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { ApiError } from "@/lib/apiFetch";
import { fmtDate, formatBytes } from "@/lib/utils";
import { useOpenBillingPortal } from "@/hooks/useBilling";
import { useCreditPacks } from "@/hooks/useCreditPacks";
import {
  useOrgLedger,
  useOrgUsage,
  useStartOrgTopup,
  useCancelOrgTopup,
  type OrgDetail,
  type OrgLedgerEntry,
  type OrgSeatUsage,
} from "@/hooks/useOrgs";
import { TransferCreditsDialog } from "@/components/orgs/TransferCreditsDialog";

// Kind -> icon + fallback label, used verbatim when the identity behind a row
// can't be resolved (departed member, missing metadata) — never a raw UUID.
const KIND_META: Record<string, { label: string; icon: LucideIcon }> = {
  transfer_in: { label: "Transfer from admin", icon: ArrowDownToLine },
  debit: { label: "Member spend", icon: ArrowUpRight },
  overage_debit: { label: "Member spend (over limit)", icon: ArrowUpRight },
  purchase: { label: "Credits purchased", icon: ShoppingCart },
  admin_grant: { label: "Credits granted", icon: Gift },
  monthly_grant: { label: "Monthly dispersal", icon: CalendarClock },
  dispersal: { label: "Monthly dispersal", icon: CalendarClock },
  refund: { label: "Refund", icon: Undo2 },
  clawback: { label: "Credits reclaimed", icon: Undo2 },
  expiry: { label: "Credits expired", icon: Clock },
};

/** fullName ?? email for a userId against `org.admins[]` (member-visible admin
 * contacts, already on the payload OrgBillingPanel renders next to) — null
 * when unresolvable (id missing, or an admin who's since lost the seat).
 * Exported for the pure-function unit test in __tests__/org-billing-panel.test.ts. */
export function adminName(org: OrgDetail, userId: unknown): string | null {
  if (typeof userId !== "string") return null;
  const admin = (org.admins ?? []).find((a) => a.userId === userId);
  return admin ? admin.fullName || admin.email : null;
}

/** email for an org_member_id against the admin usage rollup's seats —
 * null when unresolvable (departed member, missing metadata). Exported for
 * the same reason as `adminName`. */
export function memberEmail(seats: OrgSeatUsage[] | undefined, orgMemberId: unknown): string | null {
  if (typeof orgMemberId !== "string") return null;
  return seats?.find((s) => s.orgMemberId === orgMemberId)?.email ?? null;
}

export function ledgerLabel(
  entry: OrgLedgerEntry,
  org: OrgDetail,
  seats: OrgSeatUsage[] | undefined,
): { label: string; Icon: LucideIcon } {
  const meta = KIND_META[entry.kind];
  const Icon = meta?.icon ?? Receipt;
  if (entry.kind === "transfer_in") {
    const name = adminName(org, entry.metadata?.admin_user_id);
    return { label: name ? `Transfer from ${name}` : meta.label, Icon };
  }
  if (entry.kind === "debit") {
    const email = memberEmail(seats, entry.metadata?.org_member_id);
    return { label: email ? `Member spend by ${email}` : meta.label, Icon };
  }
  return meta ? { label: meta.label, Icon } : { label: entry.kind.replace(/_/g, " "), Icon };
}

function LedgerRow({ entry, org, seats }: { entry: OrgLedgerEntry; org: OrgDetail; seats: OrgSeatUsage[] | undefined }) {
  const { label, Icon } = ledgerLabel(entry, org, seats);
  const positive = entry.delta > 0;
  return (
    <div className="flex items-center justify-between gap-3 py-2.5 border-t border-border/60 first:border-t-0">
      <div className="flex items-center gap-2.5 min-w-0">
        <Icon className="w-4 h-4 text-muted-foreground flex-none" />
        <div className="min-w-0">
          <div className="text-sm font-medium truncate">{label}</div>
          <div className="text-xs text-muted-foreground">{fmtDate(entry.created_at)}</div>
        </div>
      </div>
      <div
        className={`text-sm font-semibold tabular-nums flex-none ${
          positive ? "text-emerald-600 dark:text-emerald-400" : ""
        }`}
      >
        {positive ? "+" : ""}
        {entry.delta.toLocaleString()}
      </div>
    </div>
  );
}

function PoolActivitySection({ org }: { org: OrgDetail }) {
  const { data: ledger, isLoading } = useOrgLedger(org.id);
  // Same query key AdminConsole already fetches (["orgs", id, "usage"]) —
  // react-query dedupes, so this is a cache hit in the common case, not a
  // second network round trip.
  const { data: usage } = useOrgUsage(org.id);

  return (
    <div>
      <div className="text-[15px] font-semibold flex items-center gap-2">
        <Receipt className="w-[18px] h-[18px] text-muted-foreground" />
        Pool activity
      </div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">Most recent 50 changes to the shared pool</div>

      <div className="mt-3">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : !ledger || ledger.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-8">No pool activity yet.</div>
        ) : (
          <div className="max-h-80 overflow-y-auto">
            {ledger.map((entry, i) => (
              <LedgerRow key={`${entry.created_at}-${i}`} entry={entry} org={org} seats={usage?.seats} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MonthlyTopUpSection({ org }: { org: OrgDetail }) {
  const { data: packsData, isLoading: packsLoading } = useCreditPacks();
  const startTopup = useStartOrgTopup();
  const cancelTopup = useCancelOrgTopup();
  const [confirmCancel, setConfirmCancel] = useState(false);

  const recurringPacks = (packsData?.packs ?? []).filter((p) => p.recurringPriceId);
  const active = !!org.topup_stripe_subscription_id;

  const errorMessage =
    startTopup.error instanceof ApiError
      ? startTopup.error.message
      : startTopup.error
        ? "Couldn't start the monthly top-up. Please try again."
        : null;

  return (
    <div>
      <div className="text-[15px] font-semibold flex items-center gap-2">
        <RefreshCcw className="w-[18px] h-[18px] text-muted-foreground" />
        Monthly top-up
      </div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Automatically refill the pool every month from an admin&apos;s card
      </div>

      {active ? (
        <div className="mt-3 bg-background border border-border rounded-xl px-4 py-3.5 flex items-center justify-between gap-4 flex-wrap">
          {/* ponytail: no credit amount shown here — `organizations` carries
              only the Stripe subscription id, not which pack it started
              from, so there's no amount to read without a new column or a
              live Stripe fetch. Purchaser name is free (org.admins is
              already on this payload); the amount isn't. */}
          <p className="text-[13px] text-muted-foreground max-w-[440px]">
            A monthly top-up is active — it renews via{" "}
            {adminName(org, org.topup_admin_id) ?? "the purchasing admin"}&apos;s billing until it&apos;s
            canceled.
          </p>
          <Button variant="outline" size="sm" onClick={() => setConfirmCancel(true)} disabled={cancelTopup.isPending}>
            {cancelTopup.isPending && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
            Cancel
          </Button>
        </div>
      ) : packsLoading ? (
        <div className="flex items-center justify-center py-6">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        </div>
      ) : recurringPacks.length === 0 ? (
        <p className="text-sm text-muted-foreground mt-3">No monthly top-up plans are available right now.</p>
      ) : (
        <div className="mt-3">
          {recurringPacks.map((pack) => (
            <div
              key={pack.key}
              className="flex items-center justify-between gap-3 py-3 border-t border-border/60 first:border-t-0"
            >
              <div>
                <div className="text-[14.5px] font-medium">{pack.credits.toLocaleString()} credits / month</div>
                <div className="text-[13px] text-muted-foreground mt-0.5">
                  US${(pack.price_cents / 100).toFixed(0)} / month · billed to your card
                </div>
              </div>
              <Button
                size="sm"
                onClick={() => startTopup.mutate({ orgId: org.id, key: pack.key })}
                disabled={startTopup.isPending}
              >
                {startTopup.isPending && startTopup.variables?.key === pack.key && (
                  <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                )}
                Start
              </Button>
            </div>
          ))}
        </div>
      )}

      {errorMessage && <p className="text-sm text-destructive mt-2">{errorMessage}</p>}

      <AlertDialog open={confirmCancel} onOpenChange={setConfirmCancel}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel the monthly top-up?</AlertDialogTitle>
            <AlertDialogDescription>
              The pool stops refilling automatically each month. Credits already in the pool stay —
              nothing is taken back. Any admin can start a new monthly top-up later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep it</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                cancelTopup.mutate({ orgId: org.id });
                setConfirmCancel(false);
              }}
            >
              Cancel top-up
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function StorageSection({ org }: { org: OrgDetail }) {
  const storage = org.teamStorage;

  if (!storage) {
    return (
      <div>
        <div className="text-[15px] font-semibold flex items-center gap-2">
          <HardDrive className="w-[18px] h-[18px] text-muted-foreground" />
          Storage
        </div>
        <p className="text-[13.5px] text-muted-foreground mt-0.5">
          Storage details are available once this team has an active plan holder.
        </p>
      </div>
    );
  }

  const pct = storage.poolBytes > 0 ? Math.min(100, Math.round((storage.usedBytes / storage.poolBytes) * 100)) : 0;
  const over = storage.overageGb > 0;
  const monthlyCost = storage.overageGb * storage.ratePerGb;

  return (
    <div>
      <div className="text-[15px] font-semibold flex items-center gap-2">
        <HardDrive className="w-[18px] h-[18px] text-muted-foreground" />
        Storage
      </div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        {formatBytes(storage.usedBytes)} of {formatBytes(storage.poolBytes)} used
      </div>

      <div className="h-[7px] rounded-[5px] bg-muted mt-3 overflow-hidden">
        <span
          className={`block h-full rounded-[5px] ${over ? "bg-amber-500" : "bg-primary"}`}
          style={{ width: `${Math.max(pct, storage.usedBytes > 0 ? 2 : 0)}%` }}
        />
      </div>

      {over && (
        <p className="text-[12.5px] text-amber-700 dark:text-amber-400 mt-2.5">
          {storage.overageGb.toLocaleString()} GB over — ≈ ${monthlyCost.toFixed(2)}/mo billed to the covering
          admin.
        </p>
      )}
    </div>
  );
}

export function OrgBillingPanel({ org }: { org: OrgDetail }) {
  const { openPortal, isPending: isOpeningPortal } = useOpenBillingPortal();
  const [transferOpen, setTransferOpen] = useState(false);

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-3.5 flex-wrap">
        <div>
          <div className="text-[15px] font-semibold">Billing</div>
          <div className="text-[13.5px] text-muted-foreground mt-0.5 max-w-[520px]">
            Team AI work uses the team pool — top it up or transfer credits.
          </div>
        </div>
        <Button size="sm" variant="outline" onClick={() => setTransferOpen(true)}>
          Transfer credits
        </Button>
      </div>

      <div className="mt-5 space-y-6">
        <PoolActivitySection org={org} />
        <MonthlyTopUpSection org={org} />
        <StorageSection org={org} />
      </div>

      <div className="mt-6 pt-4 border-t border-border/60 flex items-center justify-between gap-4 flex-wrap">
        <p className="text-xs text-muted-foreground/80 max-w-[440px]">
          Purchases are billed to the purchasing admin&apos;s account.
        </p>
        <Button variant="ghost" size="sm" onClick={openPortal} disabled={isOpeningPortal}>
          {isOpeningPortal && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
          Manage in Stripe billing portal
        </Button>
      </div>

      <TransferCreditsDialog orgId={org.id} orgName={org.name} open={transferOpen} onOpenChange={setTransferOpen} />
    </Card>
  );
}
