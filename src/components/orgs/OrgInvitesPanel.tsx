// src/components/orgs/OrgInvitesPanel.tsx
// Admin console: pending invites list + invite-by-email form + revoke.
// Mirrors the pending-invites idiom in TeamsPanel.tsx.
import { useState } from "react";
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
import { useOrgInvites, useInviteOrgMember, useCancelOrgInvite, type OrgRole } from "@/hooks/useOrgs";

export function OrgInvitesPanel({ orgId }: { orgId: string }) {
  const { data: invites, isLoading } = useOrgInvites(orgId);
  const inviteMember = useInviteOrgMember();
  const cancelInvite = useCancelOrgInvite();

  const [dialogOpen, setDialogOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<OrgRole>("member");

  const handleOpenChange = (open: boolean) => {
    setDialogOpen(open);
    if (!open) {
      setEmail("");
      setRole("member");
    }
  };

  const handleInvite = () => {
    if (!email.trim()) return;
    inviteMember.mutate(
      { orgId, email: email.trim(), role },
      { onSuccess: () => handleOpenChange(false) },
    );
  };

  return (
    <Card className="p-6">
      <div className="flex items-start justify-between gap-3.5">
        <div>
          <div className="text-[15px] font-semibold">Invites</div>
          <div className="text-[13.5px] text-muted-foreground mt-0.5">Pending invitations to join this organization</div>
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
            <DialogTitle>Invite to organization</DialogTitle>
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
