// src/components/admin/UserCreditsCard.tsx
// Credits section of the admin UserDetailSheet. Personal wallets get
// gift/remove + ledger; org-managed users get a hand-off to the org sheet —
// entitlements.credits for them IS the org pool, and a personal-wallet gift
// would be invisible (members debit the pool, never their personal wallet).
import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useAdminUserLedger, useAdminUserCreditMutations } from "@/hooks/useAdmin";
import type { Entitlements } from "@/hooks/useEntitlements";

export function UserCreditsCard({
  userId,
  entitlements,
  onOpenOrg,
}: {
  userId: string;
  entitlements: Entitlements;
  onOpenOrg: (orgId: string) => void;
}) {
  const credits = entitlements.credits;
  const orgCtx =
    entitlements.billingContext?.type === "org" ? entitlements.billingContext : null;
  const ledgerQuery = useAdminUserLedger(orgCtx ? null : userId);
  const { grantCredits, adjustCredits } = useAdminUserCreditMutations(userId);

  const [mode, setMode] = useState<"gift" | "remove" | null>(null);
  const [amount, setAmount] = useState("");
  const [reason, setReason] = useState("");
  const [refundId, setRefundId] = useState("");
  // One minted key per form-open/attempt cycle; re-minted after a successful
  // submit so a second deliberate action can't dedupe-collide with the first.
  // Remove lets support paste a Stripe refund/dispute id instead (the adjust
  // payload's documented intent).
  const [mintedKey, setMintedKey] = useState(() => crypto.randomUUID());

  if (!credits) return null;

  if (orgCtx) {
    return (
      <div className="border-t border-border pt-4 mt-4 space-y-2">
        <div className="text-sm font-semibold">Credits</div>
        <p className="text-xs text-muted-foreground">
          This user spends from <strong>{orgCtx.orgName}</strong>&apos;s shared pool (
          {credits.balance.toLocaleString()} credits available) — a personal-wallet gift
          would be invisible to them. Gift credits to the organization instead.
        </p>
        <Button size="sm" variant="outline" onClick={() => onOpenOrg(orgCtx.orgId)}>
          Manage in Organizations tab
        </Button>
      </div>
    );
  }

  const amountNum = Number(amount);
  const validAmount = Number.isInteger(amountNum) && amountNum > 0 && amountNum <= 1_000_000;

  const submit = () => {
    if (!mode || !validAmount) return;
    const idempotencyKey = mode === "remove" && refundId.trim() ? refundId.trim() : mintedKey;
    const mutation = mode === "gift" ? grantCredits : adjustCredits;
    mutation.mutate(
      { amount: amountNum, reason: reason.trim() || `admin ${mode}`, idempotencyKey },
      {
        onSuccess: (data) => {
          if (data.result?.duplicate) {
            toast.info("Already applied — this was a duplicate submission; no credits moved.");
          } else if (mode === "remove" && (data.result?.shortfall ?? 0) > 0) {
            // adjust is reserve-only and clamps — a request for more than the
            // reserve holds partially applies. Say so, don't claim the full amount moved.
            const removed = data.result?.removed ?? 0;
            toast.warning(
              `Removed ${removed.toLocaleString()} of ${amountNum.toLocaleString()} — only reserve credits can be removed.`,
            );
          } else {
            toast.success(
              mode === "gift"
                ? `Gifted ${amountNum.toLocaleString()} credits.`
                : `Removed ${amountNum.toLocaleString()} credits.`,
            );
          }
          setMintedKey(crypto.randomUUID());
          setMode(null);
          setAmount("");
          setReason("");
          setRefundId("");
        },
        onError: (e) => toast.error(e instanceof Error ? e.message : "Failed."),
      },
    );
  };

  return (
    <div className="border-t border-border pt-4 mt-4 space-y-3">
      <div className="text-sm font-semibold">Credits — personal wallet</div>
      <div className="text-xs text-muted-foreground tabular-nums">
        {credits.balance.toLocaleString()} available ({credits.bundleBalance.toLocaleString()}{" "}
        monthly + {credits.reserveBalance.toLocaleString()} reserve)
      </div>

      {mode === null ? (
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => setMode("gift")}>
            Gift credits
          </Button>
          <Button size="sm" variant="outline" onClick={() => setMode("remove")}>
            Remove credits
          </Button>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex gap-2">
            <Input
              type="number"
              min={1}
              max={1_000_000}
              placeholder="Amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="w-28 h-8"
            />
            <Input
              placeholder="Reason (shows in the ledger)"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="flex-1 h-8"
            />
          </div>
          {mode === "remove" && (
            <>
              <Input
                placeholder="Stripe refund/dispute id (optional — used as the dedupe key)"
                value={refundId}
                onChange={(e) => setRefundId(e.target.value)}
                className="h-8"
              />
              <p className="text-xs text-muted-foreground">
                Only reserve credits can be removed — monthly credits aren&apos;t clawback-able.
              </p>
            </>
          )}
          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={submit}
              disabled={!validAmount || grantCredits.isPending || adjustCredits.isPending}
            >
              {mode === "gift" ? "Gift" : "Remove"}
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setMode(null)}>
              Cancel
            </Button>
          </div>
        </div>
      )}

      {(ledgerQuery.data?.length ?? 0) > 0 && (
        <div className="space-y-1">
          <div className="text-xs font-medium text-muted-foreground">Recent activity</div>
          {ledgerQuery.data.slice(0, 10).map((entry, i) => (
            <div key={i} className="flex justify-between text-xs text-muted-foreground">
              <span>
                {entry.kind}
                {entry.created_at ? (
                  <span className="opacity-70"> · {new Date(entry.created_at).toLocaleDateString()}</span>
                ) : null}
              </span>
              <span className="tabular-nums">
                {entry.delta > 0 ? "+" : ""}
                {entry.delta.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
