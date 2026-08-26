// src/components/orgs/TransferCreditsDialog.tsx
// Move personal RESERVE credits into the org's shared pool (spec §4.1, Task
// 15). Mirrors OrgSeatsTable's CapDialog idioms: a plain Dialog (full control
// of when it closes — only on success, never on a bare click), a draft input,
// and a submit button disabled while invalid/pending.
import { useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError } from "@/lib/apiFetch";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useTransferCredits } from "@/hooks/useOrgs";

export function TransferCreditsDialog({
  orgId,
  orgName,
  open,
  onOpenChange,
}: {
  orgId: string;
  orgName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { data: ent } = useEntitlements();
  const [value, setValue] = useState("");
  const transfer = useTransferCredits();

  // null = not yet loaded; treat as "nothing to transfer" rather than
  // flashing an input the submit would reject anyway.
  const reserve = ent?.credits?.reserveBalance ?? 0;
  const hasReserve = reserve > 0;

  const parsed = Number(value);
  const valid = hasReserve && value !== "" && Number.isFinite(parsed) && parsed > 0 && parsed <= reserve;

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      setValue("");
      transfer.reset();
    }
    onOpenChange(next);
  };

  const submit = () => {
    if (!valid) return;
    transfer.mutate({ orgId, amount: parsed }, { onSuccess: () => handleOpenChange(false) });
  };

  // 409 (insufficient reserve) carries a structured {reason, reserveBalance} —
  // apiErrorFromBody already turns `reason` into `error.message`, so this is
  // rendered verbatim rather than re-worded.
  const errorMessage = transfer.error
    ? transfer.error instanceof ApiError
      ? transfer.error.message
      : "Couldn't transfer credits. Please try again."
    : null;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Transfer credits to {orgName}</DialogTitle>
          <DialogDescription>
            Only purchased or gifted credits can move to a team — monthly credits can&apos;t be
            transferred.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3">
          <div className="bg-background border border-border rounded-xl px-4 py-3.5 flex items-center justify-between gap-3">
            <span className="text-sm text-muted-foreground">Your reserve balance</span>
            <span className="text-sm font-semibold tabular-nums">{reserve.toLocaleString()} credits</span>
          </div>

          {hasReserve ? (
            <div className="space-y-1.5">
              <Label htmlFor="transfer-amount">Credits to transfer</Label>
              <Input
                id="transfer-amount"
                type="number"
                min={1}
                max={reserve}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="0"
              />
              <p className="text-xs text-muted-foreground">Up to {reserve.toLocaleString()} credits available.</p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              You don&apos;t have any reserve credits to transfer. Buy a credit pack from your{" "}
              <a href="/profile" className="underline underline-offset-2">
                personal billing
              </a>{" "}
              page first — packs never expire and can be moved to a team any time.
            </p>
          )}

          {errorMessage && <p className="text-sm text-destructive">{errorMessage}</p>}
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={() => handleOpenChange(false)}>
            Cancel
          </Button>
          {hasReserve && (
            <Button onClick={submit} disabled={!valid || transfer.isPending}>
              {transfer.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Transfer
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
