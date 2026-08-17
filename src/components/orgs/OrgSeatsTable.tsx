// src/components/orgs/OrgSeatsTable.tsx
// Admin console: per-member monthly cap, spend against it, and status, with
// cap/role/suspend/reactivate/remove actions.
//
// Members hold no credit balance — they spend from the org pool up to their cap
// — so the money column is "used of cap", and setting a cap moves nothing.
import { useState } from "react";
import { MoreHorizontal, Loader2, Gauge, ShieldOff, RotateCcw, UserX } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
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
  useSetMemberCap,
  type OrgSeatUsage,
  type OrgRole,
} from "@/hooks/useOrgs";
import { orgNoun, orgNounCap } from "@/lib/tiers";

/** Stored `monthly_cap` meaning "no ceiling", as opposed to null = inherit the
 * org default. Needed because every org now carries a default (2,000), so
 * clearing a member's own cap inherits it instead of removing the limit.
 * Mirrors orgs.service.UNLIMITED_CAP; normalized away server-side, so it never
 * appears in `effectiveCap`. */
const UNLIMITED_CAP = -1;

type CapMode = "default" | "custom" | "none";

const STATUS_STYLE: Record<string, string> = {
  active: "border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10",
  suspended: "border-amber-500/30 text-amber-700 dark:text-amber-400 bg-amber-500/10",
  removed: "border-border text-muted-foreground bg-muted",
};

function CapDialog({
  seat,
  orgId,
  orgKind,
  defaultCap,
  open,
  onOpenChange,
}: {
  seat: OrgSeatUsage | null;
  orgId: string;
  orgKind?: string | null;
  defaultCap: number | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [value, setValue] = useState("");
  // Three genuinely different settings, which is why this is a radio and not a
  // checkbox: "inherit" (null) stopped meaning "no limit" once every org
  // gained a default, so "none" needs its own choice — the -1 sentinel.
  const [mode, setMode] = useState<CapMode>("custom");
  const setCap = useSetMemberCap();

  // "Use the organization default" is only offered when there IS one. Every org
  // has one out of the box (2,000), so this is normally shown; it disappears
  // only if an admin cleared it, where inheriting would mean nothing at all.
  const canUseDefault = defaultCap != null;

  // Re-seed whenever a different member's dialog opens.
  const seatKey = seat?.orgMemberId ?? "";
  const [seededFor, setSeededFor] = useState("");
  if (open && seatKey && seededFor !== seatKey) {
    setSeededFor(seatKey);
    const stored = seat?.monthlyCap;
    setMode(stored === UNLIMITED_CAP ? "none" : stored == null && canUseDefault ? "default" : "custom");
    // Never seed the sentinel into the number box — it is a marker, not an amount.
    setValue(stored != null && stored !== UNLIMITED_CAP ? String(stored) : "");
  }

  const parsed = Number(value);
  const valid = mode !== "custom" || (value !== "" && Number.isFinite(parsed) && parsed >= 0);

  const submit = () => {
    if (!seat || !valid) return;
    const cap = mode === "default" ? null : mode === "none" ? UNLIMITED_CAP : parsed;
    setCap.mutate({ orgId, memberId: seat.orgMemberId, cap }, { onSuccess: () => onOpenChange(false) });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Monthly credit limit</DialogTitle>
          <DialogDescription>
            How much of the shared pool {seat?.email ?? "this member"} can use each month. Nothing is set aside — the
            limit just stops them spending more than this.
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-3">
          <RadioGroup value={mode} onValueChange={(v) => setMode(v as CapMode)} className="gap-2.5">
            {canUseDefault && (
              <div className="flex items-center gap-2">
                <RadioGroupItem value="default" id="cap-default" />
                <Label htmlFor="cap-default" className="font-normal">
                  {orgNounCap(orgKind)} default ({defaultCap.toLocaleString()} credits / month)
                </Label>
              </div>
            )}
            <div className="flex items-center gap-2">
              <RadioGroupItem value="custom" id="cap-custom" />
              <Label htmlFor="cap-custom" className="font-normal">
                A set amount each month
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="none" id="cap-none" />
              <Label htmlFor="cap-none" className="font-normal">
                No limit
              </Label>
            </div>
          </RadioGroup>

          {mode === "custom" && (
            <div className="space-y-1.5 pl-6">
              <Label htmlFor="cap-amount">Credits per month</Label>
              <Input
                id="cap-amount"
                type="number"
                min={0}
                value={value}
                onChange={(e) => setValue(e.target.value)}
                placeholder="2000"
              />
            </div>
          )}
          {mode === "none" && (
            <p className="text-xs text-muted-foreground pl-6">
              They can spend the shared pool down on their own. The pool balance is still the real ceiling.
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Used {(seat?.capUsed ?? 0).toLocaleString()} credits so far this month.
          </p>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={!valid || setCap.isPending}>
            {setCap.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
            Save limit
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function OrgSeatsTable({
  orgId,
  currentUserId,
  orgKind,
}: {
  orgId: string;
  currentUserId?: string;
  orgKind?: string | null;
}) {
  const { data: usage, isLoading, isError } = useOrgUsage(orgId);
  const updateRole = useUpdateOrgMemberRole();
  const suspend = useSuspendOrgMember();
  const reactivate = useReactivateOrgMember();
  const remove = useRemoveOrgMember();

  const [capSeat, setCapSeat] = useState<OrgSeatUsage | null>(null);
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
    ? `${confirmAction.seat.email ?? "This member"} will lose access to this ${orgNoun(orgKind)}. Nothing is deducted — they never held credits, only a limit. ${
        confirmAction.type === "suspend" ? "You can reactivate them later." : "You can re-invite them later."
      }`
    : "";

  return (
    <Card className="p-6">
      <div className="text-[15px] font-semibold">Members</div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Everyone with access, what they&apos;ve used from the pool this month, and their limit
      </div>

      <div className="mt-4">
        {seats.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-10">
            No members yet — invite someone to get started.
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Member</TableHead>
                <TableHead>Role</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Used this month</TableHead>
                <TableHead className="text-right">Limit</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {seats.map((seat) => {
                const isSelf = seat.userId === currentUserId;
                // cap_exceeded: a concurrent over-cap debit is recorded, never
                // rejected — surface it instead of showing "5,200 / 5,000" flat.
                const overCap = seat.effectiveCap != null && seat.spentThisPeriod > seat.effectiveCap;
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
                    <TableCell
                      className={`text-right tabular-nums ${overCap ? "text-amber-700 dark:text-amber-400 font-medium" : ""}`}
                      title={overCap ? "Over their monthly limit — overage still came from the pool" : undefined}
                    >
                      {seat.spentThisPeriod.toLocaleString()}
                      {seat.effectiveCap != null && (
                        <span className={overCap ? "" : "text-muted-foreground"}>
                          {" "}/ {seat.effectiveCap.toLocaleString()}
                        </span>
                      )}
                      {overCap && <div className="text-[11px] font-normal">over limit</div>}
                    </TableCell>
                    <TableCell className="text-right tabular-nums text-muted-foreground">
                      {seat.effectiveCap == null
                        ? "No limit"
                        : seat.monthlyCap == null
                          ? `${seat.effectiveCap.toLocaleString()} (default)`
                          : seat.effectiveCap.toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-7 w-7" aria-label={`Actions for ${seat.email}`}>
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => setCapSeat(seat)}>
                            <Gauge className="w-3.5 h-3.5 mr-2" /> Set monthly limit…
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

      <CapDialog
        seat={capSeat}
        orgId={orgId}
        orgKind={orgKind}
        defaultCap={usage.defaultMemberCap}
        open={!!capSeat}
        onOpenChange={(o) => !o && setCapSeat(null)}
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
