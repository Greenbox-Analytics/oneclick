// src/components/orgs/OrgRequestsPanel.tsx
// Admin console: pending credit-request queue (approve-with-amount / deny)
// + resolved history. `seats` (from the usage rollup the page already
// fetches) resolves org_member_id -> email for display — credit_requests
// rows only carry the member id.
import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import {
  useOrgCreditRequests,
  useApproveCreditRequest,
  useDenyCreditRequest,
  type OrgCreditRequest,
  type OrgSeatUsage,
} from "@/hooks/useOrgs";
import { fmtDate } from "@/lib/utils";

export function OrgRequestsPanel({ orgId, seats }: { orgId: string; seats: OrgSeatUsage[] }) {
  const { data: requests, isLoading } = useOrgCreditRequests(orgId);
  const approve = useApproveCreditRequest();
  const deny = useDenyCreditRequest();

  const [approveTarget, setApproveTarget] = useState<OrgCreditRequest | null>(null);
  const [approveAmount, setApproveAmount] = useState("");
  const [denyTarget, setDenyTarget] = useState<OrgCreditRequest | null>(null);
  const [denyNote, setDenyNote] = useState("");

  const emailByMemberId = new Map(seats.map((s) => [s.orgMemberId, s.email]));
  const requesterLabel = (r: OrgCreditRequest) => emailByMemberId.get(r.org_member_id) ?? "A member";

  const pending = (requests ?? []).filter((r) => r.status === "pending");
  const resolved = (requests ?? []).filter((r) => r.status !== "pending");

  const openApprove = (r: OrgCreditRequest) => {
    setApproveTarget(r);
    setApproveAmount(r.requested_credits != null ? String(r.requested_credits) : "");
  };
  const closeApprove = (open: boolean) => {
    if (!open) {
      setApproveTarget(null);
      setApproveAmount("");
    }
  };
  const approveAmountValue = Number(approveAmount);
  const submitApprove = () => {
    if (!approveTarget || !approveAmountValue || approveAmountValue <= 0) return;
    approve.mutate(
      { orgId, requestId: approveTarget.id, credits: approveAmountValue },
      { onSuccess: () => closeApprove(false) },
    );
  };

  const closeDeny = (open: boolean) => {
    if (!open) {
      setDenyTarget(null);
      setDenyNote("");
    }
  };
  const submitDeny = () => {
    if (!denyTarget) return;
    deny.mutate(
      { orgId, requestId: denyTarget.id, note: denyNote.trim() || undefined },
      { onSuccess: () => closeDeny(false) },
    );
  };

  return (
    <Card className="p-6">
      <div className="text-[15px] font-semibold">Credit requests</div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Members ask for more credits when their seat runs low
      </div>

      <div className="mt-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : pending.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-8">No open requests</div>
        ) : (
          <div className="space-y-2">
            {pending.map((r) => (
              <div
                key={r.id}
                className="flex items-center justify-between gap-3 bg-background border border-border rounded-xl px-4 py-3.5"
              >
                <div className="min-w-0">
                  <div className="text-sm font-medium truncate">{requesterLabel(r)}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">
                    {r.requested_credits != null
                      ? `Asked for ${r.requested_credits.toLocaleString()} credits`
                      : "Asked for more — amount up to you"}
                    {" · "}
                    {fmtDate(r.created_at)}
                  </div>
                  {r.note && <p className="text-xs text-muted-foreground/80 mt-1 italic">&quot;{r.note}&quot;</p>}
                </div>
                <div className="flex items-center gap-2 flex-none">
                  <Button size="sm" variant="outline" onClick={() => setDenyTarget(r)}>
                    Deny
                  </Button>
                  <Button size="sm" onClick={() => openApprove(r)}>
                    Approve…
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {resolved.length > 0 && (
        <div className="mt-6 pt-4 border-t border-border">
          <div className="text-[11px] font-semibold tracking-[0.11em] uppercase text-muted-foreground/70 mb-2.5">
            History
          </div>
          <div className="space-y-1.5">
            {resolved.map((r) => (
              <div key={r.id} className="flex items-center justify-between gap-3 text-sm py-1.5">
                <div className="min-w-0 truncate text-muted-foreground">{requesterLabel(r)}</div>
                <div className="flex items-center gap-2 flex-none">
                  {r.status === "approved" && (
                    <span className="text-xs text-muted-foreground tabular-nums">
                      +{(r.resolved_credits ?? 0).toLocaleString()}
                    </span>
                  )}
                  <Badge
                    variant="outline"
                    className={
                      r.status === "approved"
                        ? "border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 capitalize"
                        : "border-border text-muted-foreground bg-muted capitalize"
                    }
                  >
                    {r.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Approve — amount pre-filled from requested_credits when the member gave one. */}
      <Dialog open={!!approveTarget} onOpenChange={closeApprove}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Approve request</DialogTitle>
            <DialogDescription>
              Choose how many credits to send {approveTarget ? requesterLabel(approveTarget) : "this member"} from the
              pool.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="approve-amount">Credits</Label>
            <Input
              id="approve-amount"
              type="number"
              min={1}
              value={approveAmount}
              onChange={(e) => setApproveAmount(e.target.value)}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => closeApprove(false)}>
              Cancel
            </Button>
            <Button onClick={submitApprove} disabled={!approveAmountValue || approveAmountValue <= 0 || approve.isPending}>
              {approve.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Approve
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Deny — optional note explaining why. */}
      <Dialog open={!!denyTarget} onOpenChange={closeDeny}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Deny request</DialogTitle>
            <DialogDescription>
              Optionally let {denyTarget ? requesterLabel(denyTarget) : "this member"} know why.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="deny-note">Note (optional)</Label>
            <Textarea
              id="deny-note"
              placeholder="e.g. Let's revisit next month"
              value={denyNote}
              onChange={(e) => setDenyNote(e.target.value)}
              rows={3}
            />
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => closeDeny(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={submitDeny} disabled={deny.isPending}>
              {deny.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Deny
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
