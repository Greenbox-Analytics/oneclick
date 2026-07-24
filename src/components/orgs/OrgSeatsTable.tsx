// src/components/orgs/OrgSeatsTable.tsx
// Admin console: per-seat balance/spend/status + storage-vs-cap (round 5 —
// the finite per-seat storage ceiling must be visible before an upload
// fails), with allocate/reclaim/role/suspend/reactivate/remove actions.
import { useState } from "react";
import { MoreHorizontal, Loader2, Coins, Undo2, ShieldOff, RotateCcw, UserX } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
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
import {
  useOrgUsage,
  useUpdateOrgMemberRole,
  useSuspendOrgMember,
  useReactivateOrgMember,
  useRemoveOrgMember,
  useAllocateCredits,
  useReclaimCredits,
  type OrgSeatUsage,
  type OrgRole,
} from "@/hooks/useOrgs";
import { formatBytes } from "@/lib/utils";

const STATUS_STYLE: Record<string, string> = {
  active: "border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10",
  suspended: "border-amber-500/30 text-amber-700 dark:text-amber-400 bg-amber-500/10",
  removed: "border-border text-muted-foreground bg-muted",
};

function AllocateDialog({
  seat,
  orgId,
  open,
  onOpenChange,
}: {
  seat: OrgSeatUsage | null;
  orgId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [amount, setAmount] = useState("");
  const allocate = useAllocateCredits();
  const amountValue = Number(amount);

  const handleClose = (next: boolean) => {
    onOpenChange(next);
    if (!next) setAmount("");
  };

  const handleSubmit = () => {
    if (!seat || !amountValue || amountValue <= 0) return;
    allocate.mutate({ orgId, memberId: seat.orgMemberId, amount: amountValue }, { onSuccess: () => handleClose(false) });
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Allocate credits</DialogTitle>
          <DialogDescription>
            Move credits from the pool into {seat?.email ?? "this member"}&apos;s seat.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2">
          <Label htmlFor="allocate-amount">Amount</Label>
          <Input
            id="allocate-amount"
            type="number"
            min={1}
            placeholder="e.g. 500"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleClose(false)}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!amountValue || amountValue <= 0 || allocate.isPending}>
            {allocate.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Allocate
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ReclaimDialog({
  seat,
  orgId,
  open,
  onOpenChange,
}: {
  seat: OrgSeatUsage | null;
  orgId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [reclaimAll, setReclaimAll] = useState(true);
  const [amount, setAmount] = useState("");
  const reclaim = useReclaimCredits();
  const amountValue = Number(amount);

  const handleClose = (next: boolean) => {
    onOpenChange(next);
    if (!next) {
      setAmount("");
      setReclaimAll(true);
    }
  };

  const handleSubmit = () => {
    if (!seat) return;
    if (!reclaimAll && (!amountValue || amountValue <= 0)) return;
    reclaim.mutate(
      { orgId, memberId: seat.orgMemberId, amount: reclaimAll ? null : amountValue },
      { onSuccess: () => handleClose(false) },
    );
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Reclaim credits</DialogTitle>
          <DialogDescription>
            Move credits from {seat?.email ?? "this member"}&apos;s seat back into the pool. Seat balance:{" "}
            {(seat?.seatBalance ?? 0).toLocaleString()} credits.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Checkbox id="reclaim-all" checked={reclaimAll} onCheckedChange={(v) => setReclaimAll(v === true)} />
            <Label htmlFor="reclaim-all" className="font-normal">
              Reclaim the entire balance
            </Label>
          </div>
          {!reclaimAll && (
            <div className="space-y-2">
              <Label htmlFor="reclaim-amount">Amount</Label>
              <Input
                id="reclaim-amount"
                type="number"
                min={1}
                placeholder="e.g. 200"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
              />
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => handleClose(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={(!reclaimAll && (!amountValue || amountValue <= 0)) || reclaim.isPending}
          >
            {reclaim.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Reclaim
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function OrgSeatsTable({ orgId, currentUserId }: { orgId: string; currentUserId?: string }) {
  const { data: usage, isLoading, isError } = useOrgUsage(orgId);
  const updateRole = useUpdateOrgMemberRole();
  const suspend = useSuspendOrgMember();
  const reactivate = useReactivateOrgMember();
  const remove = useRemoveOrgMember();

  const [allocateSeat, setAllocateSeat] = useState<OrgSeatUsage | null>(null);
  const [reclaimSeat, setReclaimSeat] = useState<OrgSeatUsage | null>(null);
  const [confirmAction, setConfirmAction] = useState<{ type: "suspend" | "remove"; seat: OrgSeatUsage } | null>(null);

  if (isLoading) {
    return (
      <Card className="p-6 flex items-center justify-center py-12">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </Card>
    );
  }
  if (isError || !usage) {
    return (
      <Card className="p-6 text-sm text-muted-foreground text-center py-10">
        Couldn&apos;t load seats. Please try refreshing.
      </Card>
    );
  }

  const seats = usage.seats;
  const confirmDescription = confirmAction
    ? `${confirmAction.seat.email ?? "This member"} will lose access and their remaining credits will be reclaimed to the pool. ${
        confirmAction.type === "suspend" ? "You can reactivate them later." : "You can re-invite them later."
      }`
    : "";

  return (
    <Card className="p-6">
      <div className="text-[15px] font-semibold">Seats</div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Everyone with access to this organization, and what they&apos;ve used
      </div>

      <div className="mt-4">
        {seats.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-10">
            No seats yet — invite someone to get started.
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Member</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Seat balance</TableHead>
                <TableHead className="text-right">Spent</TableHead>
                <TableHead>Storage</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {seats.map((seat) => {
                const isSelf = seat.userId === currentUserId;
                const storagePct = seat.storageCapBytes > 0 ? seat.storageBytes / seat.storageCapBytes : 0;
                const nearCap = storagePct >= 0.8;
                const isActive = seat.status === "active";
                return (
                  <TableRow key={seat.orgMemberId}>
                    <TableCell className="max-w-[220px]">
                      <div className="text-sm font-medium truncate">{seat.email ?? "Unknown"}</div>
                      {isSelf && <div className="text-xs text-muted-foreground">You</div>}
                    </TableCell>
                    <TableCell>
                      {isActive && !isSelf ? (
                        <Select
                          value={seat.role}
                          onValueChange={(role) =>
                            updateRole.mutate({ orgId, memberId: seat.orgMemberId, role: role as OrgRole })
                          }
                        >
                          <SelectTrigger className="h-7 w-24 text-xs" aria-label={`Change role for ${seat.email}`}>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="admin">Admin</SelectItem>
                            <SelectItem value="member">Member</SelectItem>
                          </SelectContent>
                        </Select>
                      ) : (
                        <Badge variant="outline" className="capitalize">
                          {seat.role}
                        </Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={`capitalize ${STATUS_STYLE[seat.status] ?? ""}`}>
                        {seat.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{seat.seatBalance.toLocaleString()}</TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {seat.spentAllTime.toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <div
                        className={`text-xs ${nearCap ? "text-amber-700 dark:text-amber-400 font-medium" : "text-muted-foreground"}`}
                      >
                        {formatBytes(seat.storageBytes)} of {formatBytes(seat.storageCapBytes)}
                      </div>
                      <div className="h-1.5 w-24 rounded-full bg-muted mt-1 overflow-hidden">
                        <span
                          className={`block h-full rounded-full ${nearCap ? "bg-amber-500" : "bg-primary"}`}
                          style={{ width: `${Math.min(100, Math.round(storagePct * 100))}%` }}
                        />
                      </div>
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={`Actions for ${seat.email}`}>
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => setAllocateSeat(seat)}>
                            <Coins className="w-3.5 h-3.5 mr-2" /> Allocate credits…
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => setReclaimSeat(seat)} disabled={seat.seatBalance <= 0}>
                            <Undo2 className="w-3.5 h-3.5 mr-2" /> Reclaim credits…
                          </DropdownMenuItem>
                          {!isSelf && (
                            <>
                              <DropdownMenuSeparator />
                              {seat.status === "active" && (
                                <DropdownMenuItem onClick={() => setConfirmAction({ type: "suspend", seat })}>
                                  <ShieldOff className="w-3.5 h-3.5 mr-2" /> Suspend
                                </DropdownMenuItem>
                              )}
                              {(seat.status === "suspended" || seat.status === "removed") && (
                                <DropdownMenuItem
                                  onClick={() => reactivate.mutate({ orgId, memberId: seat.orgMemberId })}
                                >
                                  <RotateCcw className="w-3.5 h-3.5 mr-2" /> Reactivate
                                </DropdownMenuItem>
                              )}
                              {seat.status !== "removed" && (
                                <DropdownMenuItem
                                  className="text-destructive focus:text-destructive"
                                  onClick={() => setConfirmAction({ type: "remove", seat })}
                                >
                                  <UserX className="w-3.5 h-3.5 mr-2" /> Remove
                                </DropdownMenuItem>
                              )}
                            </>
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </div>

      <AllocateDialog
        seat={allocateSeat}
        orgId={orgId}
        open={!!allocateSeat}
        onOpenChange={(o) => !o && setAllocateSeat(null)}
      />
      <ReclaimDialog
        seat={reclaimSeat}
        orgId={orgId}
        open={!!reclaimSeat}
        onOpenChange={(o) => !o && setReclaimSeat(null)}
      />

      <AlertDialog open={!!confirmAction} onOpenChange={(o) => !o && setConfirmAction(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {confirmAction?.type === "suspend" ? "Suspend this member?" : "Remove this member?"}
            </AlertDialogTitle>
            <AlertDialogDescription>{confirmDescription}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (!confirmAction) return;
                if (confirmAction.type === "suspend") {
                  suspend.mutate({ orgId, memberId: confirmAction.seat.orgMemberId });
                } else {
                  remove.mutate({ orgId, memberId: confirmAction.seat.orgMemberId });
                }
                setConfirmAction(null);
              }}
            >
              {confirmAction?.type === "suspend" ? "Suspend" : "Remove"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}
