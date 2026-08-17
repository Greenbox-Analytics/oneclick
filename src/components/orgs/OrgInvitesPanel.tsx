// src/components/orgs/OrgInvitesPanel.tsx
// Admin console: pending invites list + invite-by-email form + revoke.
// Pending invites live here now — the old workspace Teams panel is gone.
import { useState } from "react";
import { Link } from "react-router-dom";
import { Loader2, Mail, UserPlus, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ApiError } from "@/lib/apiFetch";
import { useOrgInvites, useInviteOrgMember, useCancelOrgInvite, type OrgRole } from "@/hooks/useOrgs";
import { orgNoun } from "@/lib/tiers";

/** Seat-wall 402 detail shape from POST /orgs/{id}/invites (orgs/router.py). */
interface SeatWallDetail {
  reason: string;
  nextStep: "upgrade" | "contact";
}

function seatWallDetail(e: unknown): SeatWallDetail | null {
  if (!(e instanceof ApiError) || !e.detail || typeof e.detail !== "object") return null;
  const d = e.detail as { reason?: unknown; nextStep?: unknown };
  if (d.nextStep !== "upgrade" && d.nextStep !== "contact") return null;
  return { reason: typeof d.reason === "string" ? d.reason : e.message, nextStep: d.nextStep };
}

export function OrgInvitesPanel({ orgId, orgKind }: { orgId: string; orgKind?: string | null }) {
  const { data: invites, isLoading } = useOrgInvites(orgId);
  const inviteMember = useInviteOrgMember();
  const cancelInvite = useCancelOrgInvite();

  const [dialogOpen, setDialogOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<OrgRole>("member");
  const [seatWall, setSeatWall] = useState<SeatWallDetail | null>(null);

  const handleOpenChange = (open: boolean) => {
    setDialogOpen(open);
    if (!open) {
      setEmail("");
      setRole("member");
      setSeatWall(null);
    }
  };

  const handleInvite = () => {
    if (!email.trim()) return;
    setSeatWall(null);
    inviteMember.mutate(
      { orgId, email: email.trim(), role },
      {
        onSuccess: () => handleOpenChange(false),
        onError: (e) => setSeatWall(seatWallDetail(e)),
      },
    );
  };

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-3.5">
        <div>
          <div className="text-[15px] font-semibold">Invites</div>
          <div className="text-[13.5px] text-muted-foreground mt-0.5">Pending invitations to join this {orgNoun(orgKind)}</div>
        </div>
        <Button size="sm" variant="outline" className="gap-1.5" onClick={() => setDialogOpen(true)}>
          <UserPlus className="w-3.5 h-3.5" />
          Invite
        </Button>
      </div>

      <div className="mt-4">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : !invites || invites.length === 0 ? (
          <div className="text-sm text-muted-foreground text-center py-8">No pending invites</div>
        ) : (
          <div className="space-y-2">
            {invites.map((inv) => (
              <div
                key={inv.id}
                className="flex items-center justify-between gap-3 bg-background border border-border rounded-xl px-4 py-3"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <Mail className="w-3.5 h-3.5 text-muted-foreground flex-none" />
                  <div className="min-w-0">
                    <div className="text-sm truncate">{inv.email}</div>
                    <div className="text-xs text-muted-foreground capitalize">
                      Invited as {inv.role}
                      {inv.expires_at && ` — expires ${new Date(inv.expires_at).toLocaleDateString()}`}
                    </div>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive flex-none"
                  aria-label={`Cancel invite to ${inv.email}`}
                  title={`Cancel invite to ${inv.email}`}
                  onClick={() => cancelInvite.mutate({ orgId, inviteId: inv.id })}
                >
                  <X className="w-3.5 h-3.5" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>

      <Dialog open={dialogOpen} onOpenChange={handleOpenChange}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Invite to {orgNoun(orgKind)}</DialogTitle>
            <DialogDescription>They&apos;ll get an email invite to join with their own seat.</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="org-invite-email">Email</Label>
              <Input
                id="org-invite-email"
                type="email"
                placeholder="name@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={role} onValueChange={(v) => setRole(v as OrgRole)}>
                <SelectTrigger aria-label="Invite role">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="admin">Admin</SelectItem>
                  <SelectItem value="member">Member</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {seatWall && (
              <div className="text-sm text-muted-foreground bg-muted rounded-lg px-3 py-2">
                {seatWall.reason}{" "}
                {seatWall.nextStep === "upgrade" ? (
                  <Link to="/pricing" className="text-primary underline underline-offset-2">
                    Upgrade to Pro
                  </Link>
                ) : (
                  <Link to="/contact" className="text-primary underline underline-offset-2">
                    Talk to us about Enterprise
                  </Link>
                )}
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => handleOpenChange(false)}>
              Cancel
            </Button>
            <Button onClick={handleInvite} disabled={!email.trim() || inviteMember.isPending}>
              {inviteMember.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
              Send invite
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
