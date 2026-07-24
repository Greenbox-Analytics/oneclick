// src/pages/Organization.tsx
// Licensing Phase B (spec §7, plan Tasks 12-13) — the /organization console:
// the admin view (Task 12) and the member view (Task 13, MemberPanel below —
// seat balance/usage, "Request more credits" form, own request history).
// Reachable regardless of LICENSING_ENABLED: it self-handles the flag-off
// state (GET /orgs 404s for everyone while the flag is off) by showing the
// same "create an organization" empty state a flag-on user with zero orgs
// sees, rather than a route-level hard-hide/redirect — submitting that form
// while the flag is off simply surfaces the backend's 404 as a toast, same
// as any other disabled-feature attempt.
import { useState } from "react";
import { Building2, Loader2, Send } from "lucide-react";
import { toast } from "sonner";
import { PageHeader } from "@/components/layout/PageHeader";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useAuth } from "@/contexts/AuthContext";
import { ApiError } from "@/lib/apiFetch";
import { fmtDate } from "@/lib/utils";
import { useEntitlements } from "@/hooks/useEntitlements";
import { useCreditUsage, type CreditAction } from "@/hooks/useCreditUsage";
import {
  useMyOrgs,
  useOrg,
  useOrgUsage,
  useCreateOrg,
  useOrgCreditRequests,
  useSubmitCreditRequest,
  type OrgSummary,
} from "@/hooks/useOrgs";
import { OrgPoolCard } from "@/components/orgs/OrgPoolCard";
import { OrgSeatsTable } from "@/components/orgs/OrgSeatsTable";
import { OrgInvitesPanel } from "@/components/orgs/OrgInvitesPanel";
import { OrgRequestsPanel } from "@/components/orgs/OrgRequestsPanel";
import { OrgSettingsPanel } from "@/components/orgs/OrgSettingsPanel";
import { OrgLinkedProjectsPanel } from "@/components/orgs/OrgLinkedProjectsPanel";

const TOOL_LABELS: Record<CreditAction, string> = {
  oneclick_run: "OneClick run",
  registry_parse: "Registry parse",
  zoe_message: "Zoe message",
};

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background">
      <PageHeader backTo="/profile" />
      <main className="container mx-auto px-4 py-8 max-w-5xl">{children}</main>
    </div>
  );
}

function CreateOrgPanel({ onCreated }: { onCreated?: (orgId: string) => void }) {
  const [name, setName] = useState("");
  const createOrg = useCreateOrg();

  const handleCreate = () => {
    if (!name.trim()) return;
    createOrg.mutate(
      { name: name.trim() },
      {
        onSuccess: (org) => {
          setName("");
          if (org?.id) onCreated?.(org.id);
        },
      },
    );
  };

  return (
    <Card className="p-8 max-w-lg mx-auto text-center">
      <div className="w-12 h-12 rounded-full bg-primary/10 text-primary flex items-center justify-center mx-auto">
        <Building2 className="w-6 h-6" />
      </div>
      <h1 className="text-xl font-semibold tracking-tight mt-4">Create an organization</h1>
      <p className="text-[13.5px] text-muted-foreground mt-1.5 max-w-sm mx-auto">
        Organizations share one credit pool across your whole team, with seats, invites, and admin
        controls. A new organization starts <strong>pending</strong> and turns on automatically once
        its first credit purchases reach the minimum — no separate activation step.
      </p>

      <div className="mt-6 text-left space-y-2 max-w-xs mx-auto">
        <Label htmlFor="new-org-name">Organization name</Label>
        <Input
          id="new-org-name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Greenbox Analytics"
          onKeyDown={(e) => e.key === "Enter" && handleCreate()}
        />
      </div>

      <Button className="mt-5" onClick={handleCreate} disabled={!name.trim() || createOrg.isPending}>
        {createOrg.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
        Create organization
      </Button>
    </Card>
  );
}

function OrgHeader({ org }: { org: OrgSummary }) {
  return (
    <div className="mb-6 flex items-center gap-3 flex-wrap">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">{org.name}</h1>
        <p className="text-muted-foreground mt-1">Organization credits, seats, and members</p>
      </div>
      {org.status === "pending" && (
        <Badge variant="outline" className="border-amber-500/30 text-amber-700 dark:text-amber-400 bg-amber-500/10">
          Pending activation
        </Badge>
      )}
    </div>
  );
}

function AdminConsole({ orgId }: { orgId: string }) {
  const { user } = useAuth();
  const { data: org, isLoading } = useOrg(orgId);
  const { data: usage } = useOrgUsage(orgId);

  if (isLoading || !org) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-[22px]">
      <OrgHeader org={org} />
      <OrgPoolCard org={org} />
      <OrgSeatsTable orgId={orgId} currentUserId={user?.id} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-[22px] items-start">
        <OrgInvitesPanel orgId={orgId} />
        <OrgRequestsPanel orgId={orgId} seats={usage?.seats ?? []} />
      </div>
      <OrgLinkedProjectsPanel orgId={orgId} seats={usage?.seats ?? []} />
      <OrgSettingsPanel org={org} />
    </div>
  );
}

/** Member (non-admin) view (plan Task 13): seat balance + usage (when this
 * org is the caller's active billing context), a "Request more credits" form,
 * and the member's own request history. `GET /orgs/{id}/credit-requests`
 * scopes itself to the caller's own rows for non-admins (backend), so
 * `useOrgCreditRequests` is reused as-is here — no separate "my requests"
 * endpoint needed. */
function MemberRequestForm({ orgId, hasPending }: { orgId: string; hasPending: boolean }) {
  const [amount, setAmount] = useState("");
  const [note, setNote] = useState("");
  const submitRequest = useSubmitCreditRequest();

  if (hasPending) {
    return (
      <div className="mt-4 bg-amber-500/10 border border-amber-500/30 rounded-xl px-4 py-3.5 text-[13px] text-amber-700 dark:text-amber-400">
        You already have a request waiting for your admin.
      </div>
    );
  }

  const handleSubmit = () => {
    const trimmed = amount.trim();
    const parsed = trimmed ? Number(trimmed) : undefined;
    if (trimmed && (!Number.isFinite(parsed) || (parsed as number) <= 0)) {
      toast.error("Enter a positive number of credits, or leave it blank.");
      return;
    }
    submitRequest.mutate(
      { orgId, requestedCredits: parsed, note: note.trim() || undefined },
      { onSuccess: () => { setAmount(""); setNote(""); } },
    );
  };

  return (
    <div className="mt-4 space-y-3">
      <div className="grid grid-cols-1 sm:grid-cols-[160px_1fr] gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="request-amount">Amount (optional)</Label>
          <Input
            id="request-amount"
            type="number"
            min={1}
            placeholder="Let admin decide"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
          />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="request-note">Note (optional)</Label>
          <Textarea
            id="request-note"
            rows={1}
            placeholder="e.g. Wrapping up a big OneClick run"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>
      </div>
      <Button size="sm" onClick={handleSubmit} disabled={submitRequest.isPending}>
        {submitRequest.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
        <Send className="w-3.5 h-3.5 mr-1.5" />
        Send request
      </Button>
    </div>
  );
}

function statusBadgeClass(status: string): string {
  if (status === "approved") {
    return "border-emerald-500/30 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 capitalize";
  }
  if (status === "pending") {
    return "border-amber-500/30 text-amber-700 dark:text-amber-400 bg-amber-500/10 capitalize";
  }
  return "border-border text-muted-foreground bg-muted capitalize";
}

function MemberPanel({ org }: { org: OrgSummary }) {
  const { data: ent } = useEntitlements();
  const { data: usage } = useCreditUsage();
  const { data: myRequests, isLoading: requestsLoading } = useOrgCreditRequests(org.id);

  // billingContext is the canonical org-context signal (present even when
  // CREDITS_ENABLED is off); credits.managedByOrg is the back-compat fallback.
  const managedByOrg =
    ent?.billingContext?.type === "org" ? ent.billingContext : ent?.credits?.managedByOrg;
  const isActiveContext = managedByOrg?.orgId === org.id;
  const hasPending = (myRequests ?? []).some((r) => r.status === "pending");
  const usedTools = (usage?.tools ?? []).filter((t) => t.count > 0);

  return (
    <div className="flex flex-col gap-[22px]">
      <OrgHeader org={org} />
      <Card className="p-6">
        <div className="text-[15px] font-semibold">Your seat</div>
        <div className="text-[13.5px] text-muted-foreground mt-0.5">
          You&apos;re a member of {org.name} — an admin manages invites and credits
        </div>

        <div className="mt-4 bg-background border border-border rounded-xl px-[18px] py-4">
          {isActiveContext ? (
            <>
              <div className="text-[12.5px] text-muted-foreground">Seat balance</div>
              <div className="text-[28px] font-bold tracking-tight mt-1 tabular-nums">
                {(ent?.credits?.balance ?? 0).toLocaleString()}{" "}
                <span className="text-sm font-normal text-muted-foreground">credits</span>
              </div>

              {usedTools.length > 0 && (
                <div className="mt-4 pt-3.5 border-t border-border/60">
                  <div className="text-[11px] font-semibold tracking-[0.11em] uppercase text-muted-foreground/70 mb-2">
                    Your usage
                  </div>
                  <div className="space-y-1.5">
                    {usedTools.map((t) => (
                      <div key={t.action} className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">{TOOL_LABELS[t.action] ?? t.action}</span>
                        <span className="tabular-nums">
                          {t.spent.toLocaleString()} cr · {t.count}x
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <p className="text-[13px] text-muted-foreground">
              Switch your billing context to {org.name} from your{" "}
              <a href="/profile" className="underline underline-offset-2">
                Profile
              </a>{" "}
              page to see your live seat balance here.
            </p>
          )}
        </div>
      </Card>

      <Card className="p-6">
        <div className="text-[15px] font-semibold">Request more credits</div>
        <div className="text-[13.5px] text-muted-foreground mt-0.5">
          Running low? Ask your admin to send more from the organization&apos;s pool.
        </div>

        <MemberRequestForm orgId={org.id} hasPending={hasPending} />

        {requestsLoading ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : (
          (myRequests ?? []).length > 0 && (
            <div className="mt-6 pt-4 border-t border-border">
              <div className="text-[11px] font-semibold tracking-[0.11em] uppercase text-muted-foreground/70 mb-2.5">
                Your requests
              </div>
              <div className="space-y-1.5">
                {(myRequests ?? []).map((r) => (
                  <div key={r.id} className="flex items-center justify-between gap-3 text-sm py-1.5">
                    <div className="min-w-0 text-muted-foreground">
                      {r.requested_credits != null
                        ? `Asked for ${r.requested_credits.toLocaleString()}`
                        : "Asked for more — amount up to admin"}
                      {" · "}
                      {fmtDate(r.created_at)}
                    </div>
                    <div className="flex items-center gap-2 flex-none">
                      {r.status === "approved" && (
                        <span className="text-xs text-muted-foreground tabular-nums">
                          +{(r.resolved_credits ?? 0).toLocaleString()}
                        </span>
                      )}
                      <Badge variant="outline" className={statusBadgeClass(r.status)}>
                        {r.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        )}
      </Card>
    </div>
  );
}

const Organization = () => {
  const { data: orgs, isLoading, error } = useMyOrgs();
  const [selectedOrgId, setSelectedOrgId] = useState<string | null>(null);

  if (isLoading) {
    return (
      <PageShell>
        <div className="flex items-center justify-center py-24">
          <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
        </div>
      </PageShell>
    );
  }

  // GET /orgs 404s for EVERY caller while LICENSING_ENABLED is off (router-
  // level gate) — that's indistinguishable from "flag on, zero orgs" from
  // here, so both land on the same create-an-org empty state.
  const licensingOff = error instanceof ApiError && error.status === 404;
  const orgList = orgs ?? [];

  if (licensingOff || orgList.length === 0) {
    return (
      <PageShell>
        <CreateOrgPanel onCreated={setSelectedOrgId} />
      </PageShell>
    );
  }

  const orgId = selectedOrgId && orgList.some((o) => o.id === selectedOrgId) ? selectedOrgId : orgList[0].id;
  const selected = orgList.find((o) => o.id === orgId) ?? orgList[0];
  const isAdmin = selected.my_role === "admin";

  return (
    <PageShell>
      {orgList.length > 1 && (
        <div className="mb-6">
          <Select value={orgId} onValueChange={setSelectedOrgId}>
            <SelectTrigger className="w-64" aria-label="Select organization">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {orgList.map((o) => (
                <SelectItem key={o.id} value={o.id}>
                  {o.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      {isAdmin ? <AdminConsole orgId={orgId} /> : <MemberPanel org={selected} />}
    </PageShell>
  );
};

export default Organization;
