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

/** Badge classes for a credit request's status — shared with the member view
 * in pages/Organization.tsx so the two lists colour a status identically. */
export function requestStatusClass(status: string): string {
  if (status === "approved") {
    return "border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 capitalize";
  }
  if (status === "pending") {
    return "border-amber-500/30 text-amber-700 dark:text-amber-400 bg-amber-500/10 capitalize";
  }
  return "border-border text-muted-foreground bg-muted capitalize";
}

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
    setApproveAmount(r.requested_cap != null ? String(r.requested_cap) : "");
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
      { orgId, requestId: approveTarget.id, cap: approveAmountValue },
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
        Members ask for a higher monthly limit when they hit theirs
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
                    {r.requested_cap != null
                      ? `Asked for a ${r.requested_cap.toLocaleString()} credit / month limit`
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
                      {(r.resolved_cap ?? 0).toLocaleString()} / mo
                    </span>
                  )}
                  <Badge variant="outline" className={requestStatusClass(r.status)}>
                    {r.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Approve — pre-filled from requested_cap when the member named one. */}
      <Dialog open={!!approveTarget} onOpenChange={closeApprove}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Approve request</DialogTitle>
            <DialogDescription>
              Set {approveTarget ? requesterLabel(approveTarget) : "this member"}&apos;s new monthly
              limit.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="approve-amount">New monthly limit</Label>
            <Input
              id="approve-amount"
              type="number"
              min={1}
              value={approveAmount}
              onChange={(e) => setApproveAmount(e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Nothing is set aside — this only raises their ceiling.
            </p>
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
