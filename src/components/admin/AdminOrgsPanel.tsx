// src/components/admin/AdminOrgsPanel.tsx
// Admin console → Organizations: list every org (flag-independent — suspended
// and archived stay visible, labeled) with a detail drawer for the license
// (status, activation progress, suspend/reinstate) and the shared credit pool
// (gift, monthly dispersal, ledger).
import { useState } from "react";
import { toast } from "sonner";
import { ChevronRight, KeyRound, Loader2, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  useAdminOrgs,
  useAdminOrgPool,
  useAdminOrgMutations,
  useAdminOrgUsage,
  useAdminKeyLookup,
  useAdminPartnerKeys,
  useAdminRevokePartnerKey,
  type KeyTraceHit,
  type AdminOrgRow,
} from "@/hooks/useAdminOrgs";
import { OrgUsageAnalysis } from "@/components/orgs/OrgUsageAnalysis";
import { keyStatus } from "@/lib/partnerKeys";
import { KeyValue, ORG_STATUS_TONE, SectionLabel, shortDate, Tag } from "@/components/admin/ui";

export function AdminOrgsPanel({
  selectedOrgId,
  onSelectOrg,
}: {
  selectedOrgId: string | null;
  onSelectOrg: (id: string | null) => void;
}) {
  const orgsQuery = useAdminOrgs();
  const [query, setQuery] = useState("");
  const orgs = orgsQuery.data ?? [];
  const list = orgs.filter((o) =>
    (o.name ?? "").toLowerCase().includes(query.trim().toLowerCase()),
  );
  const selected = orgs.find((o) => o.id === selectedOrgId) ?? null;

  return (
    <>
      <KeyTrace onSelectOrg={onSelectOrg} />

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="relative w-full max-w-xs">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search organizations"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <span className="ml-auto text-[11.5px] text-muted-foreground">
          {list.length} organization{list.length === 1 ? "" : "s"}
        </span>
      </div>

      <Card className="overflow-hidden p-0">
        {orgsQuery.isLoading && (
          <div className="p-6 text-sm text-muted-foreground">Loading organizations…</div>
        )}
        {orgsQuery.error && (
          <div className="p-6 text-sm text-destructive">
            Couldn&apos;t load organizations. Refresh to try again.
          </div>
        )}
        {!orgsQuery.isLoading && !orgsQuery.error && list.length === 0 && (
          <div className="p-6 text-sm text-muted-foreground">
            {orgs.length === 0 ? "No organizations yet." : "No organizations match."}
          </div>
        )}
        {list.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b border-border text-left text-[10.5px] uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-2.5 font-semibold">Organization</th>
                  <th className="px-4 py-2.5 font-semibold">Status</th>
                  <th className="px-4 py-2.5 text-right font-semibold">Members</th>
                  <th className="px-4 py-2.5 text-right font-semibold">Pool</th>
                  <th className="px-4 py-2.5 text-right font-semibold">Monthly</th>
                  <th className="px-4 py-2.5 text-right font-semibold">Activation</th>
                  <th className="w-8 px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {list.map((o) => (
                  <tr
                    key={o.id}
                    className="cursor-pointer border-b border-border/60 last:border-b-0 hover:bg-muted/40"
                    onClick={() => onSelectOrg(o.id)}
                  >
                    <td className="px-4 py-2.5 font-medium">{o.name ?? "—"}</td>
                    <td className="px-4 py-2.5">
                      <Tag tone={ORG_STATUS_TONE[o.status] ?? "neutral"}>{o.status}</Tag>
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums">{o.memberCount}</td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                      {(o.bundleBalance + o.reserveBalance).toLocaleString()}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                      {o.monthlyDispersalCredits.toLocaleString()}
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono tabular-nums text-muted-foreground">
                      {o.status === "pending"
                        ? `${o.cumulativePaidIn.toLocaleString()} / ${o.activationFloor.toLocaleString()}`
                        : "—"}
                    </td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground">
                      <ChevronRight className="h-4 w-4" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <OrgDetailSheet
        key={selectedOrgId ?? "closed"}
        org={selected}
        onClose={() => onSelectOrg(null)}
      />
    </>
  );
}

function OrgDetailSheet({ org, onClose }: { org: AdminOrgRow | null; onClose: () => void }) {
  const poolQuery = useAdminOrgPool(org?.id ?? null);
  const { grantCredits, setDispersal, setStatus, setPartnerApi } = useAdminOrgMutations();

  const [giftAmount, setGiftAmount] = useState("");
  const [giftReason, setGiftReason] = useState("");
  // One key per grant ATTEMPT: a double-click or retry of the same attempt
  // dedupes at the RPC, while a second deliberate gift re-mints in onSuccess
  // (a useMemo keyed on org.id would silently no-op the second gift and
  // still toast success). Failed attempts keep the key — that's the point.
  const [giftKey, setGiftKey] = useState(() => crypto.randomUUID());
  const [dispersalDraft, setDispersalDraft] = useState<string | null>(null);
  const [tab, setTab] = useState("license");

  if (!org) return null;

  // Same fallback chain as the displayed "Cumulative paid-in" row — the pool
  // query settles after the row's own org.cumulativePaidIn is already visible,
  // so both the display and the activation check must read the same value.
  const paidIn = poolQuery.data?.cumulativePaidIn ?? org.cumulativePaidIn;
  const pool = poolQuery.data?.poolBalance ?? org.bundleBalance + org.reserveBalance;

  const amountNum = Number(giftAmount);
  const validAmount = Number.isFinite(amountNum) && Number.isInteger(amountNum) && amountNum > 0;
  const willActivate =
    org.status === "pending" && validAmount && paidIn + amountNum >= org.activationFloor;

  const submitGift = () => {
    if (!validAmount) return;
    grantCredits.mutate(
      {
        orgId: org.id,
        amount: amountNum,
        reason: giftReason.trim() || "admin grant",
        idempotencyKey: giftKey,
      },
      {
        onSuccess: (data) => {
          if (data.result?.duplicate) {
            toast.info("Already applied — this was a duplicate submission; no credits moved.");
          } else {
            toast.success(`Granted ${amountNum.toLocaleString()} credits to ${org.name ?? "org"}.`);
          }
          setGiftKey(crypto.randomUUID());
          setGiftAmount("");
          setGiftReason("");
        },
        onError: (e) => toast.error(e instanceof Error ? e.message : "Grant failed."),
      },
    );
  };

  const dispersalValue = dispersalDraft ?? String(org.monthlyDispersalCredits);
  const submitDispersal = () => {
    if (dispersalValue.trim() === "") {
      toast.error("Enter a number of credits — the field can't be empty.");
      return;
    }
    const parsed = Number(dispersalValue);
    if (!Number.isFinite(parsed) || parsed < 0) {
      toast.error("Enter a non-negative number of credits.");
      return;
    }
    if (!Number.isInteger(parsed)) {
      toast.error("Enter a whole number of credits.");
      return;
    }
    setDispersal.mutate(
      { orgId: org.id, monthlyDispersalCredits: parsed },
      {
        onSuccess: () => {
          toast.success("Monthly dispersal updated.");
          setDispersalDraft(null);
        },
        onError: (e) => toast.error(e instanceof Error ? e.message : "Update failed."),
      },
    );
  };

  const submitStatus = (action: "suspend" | "reactivate") => {
    setStatus.mutate(
      { orgId: org.id, action },
      {
        onSuccess: () =>
          toast.success(action === "suspend" ? "License suspended." : "License reinstated."),
        onError: (e) => toast.error(e instanceof Error ? e.message : "Status change failed."),
      },
    );
  };

  const activationPct = Math.min(100, org.activationFloor ? (paidIn / org.activationFloor) * 100 : 0);
  // Activation is derived, not a toggle: an org goes active the moment its
  // cumulative paid-in crosses the floor (orgs/wallets.maybe_activate_org).
  // So "activate this org" IS "gift the shortfall" — pre-fill it rather than
  // offering a status flip that would break `active ⟺ paid-in ≥ floor`.
  const shortfall = Math.max(0, org.activationFloor - paidIn);

  return (
    <Sheet open={!!org} onOpenChange={(open) => !open && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-lg">
        <SheetHeader>
          <SheetTitle className="truncate">{org.name ?? "Organization"}</SheetTitle>
        </SheetHeader>

        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <Tag tone={ORG_STATUS_TONE[org.status] ?? "neutral"}>{org.status}</Tag>
          <Tag>
            {org.memberCount} member{org.memberCount === 1 ? "" : "s"}
          </Tag>
          {org.archivedAt && <Tag tone="bad">Archived {shortDate(org.archivedAt)}</Tag>}
          {org.partnerApiEnabled && <Tag tone="ok">Partner API</Tag>}
        </div>

        <Tabs value={tab} onValueChange={setTab} className="mt-4">
          <TabsList className="w-full justify-start">
            <TabsTrigger value="license">License</TabsTrigger>
            <TabsTrigger value="credits">Credits</TabsTrigger>
            <TabsTrigger value="usage">Usage</TabsTrigger>
          </TabsList>

          <TabsContent value="license" className="space-y-6 pt-4">
            {org.status === "pending" && (
              <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 dark:border-amber-900 dark:bg-amber-950/40">
                <div className="text-[13px] font-semibold">Awaiting activation</div>
                <div className="mb-2 mt-1 text-xs">
                  {paidIn.toLocaleString()} of {org.activationFloor.toLocaleString()} credits paid
                  in. There is no approve step — the license activates itself the moment paid-in
                  credits reach the floor, whether they come from the org buying a pack, the
                  monthly dispersal, or an admin gift. Activation is permanent; a clawback
                  won&apos;t reverse it.
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-background">
                  <div className="h-full bg-amber-500" style={{ width: `${activationPct}%` }} />
                </div>
                <Button
                  size="sm"
                  className="mt-3"
                  onClick={() => {
                    setGiftAmount(String(shortfall));
                    setGiftReason("activation grant");
                    setTab("credits");
                  }}
                >
                  Activate now — gift {shortfall.toLocaleString()} credits
                </Button>
              </div>
            )}

            <section>
              <SectionLabel>License</SectionLabel>
              <KeyValue k="Status" v={org.status} />
              <KeyValue k="Active members" v={String(org.memberCount)} />
              <KeyValue k="Monthly dispersal" v={org.monthlyDispersalCredits.toLocaleString()} />
              <KeyValue k="Cumulative paid-in" v={paidIn.toLocaleString()} />
            </section>

            <section>
              <SectionLabel>Danger zone</SectionLabel>
              {org.status === "suspended" ? (
                <Button size="sm" disabled={setStatus.isPending} onClick={() => submitStatus("reactivate")}>
                  Reinstate license
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  className="text-destructive hover:text-destructive"
                  disabled={org.status !== "active" || setStatus.isPending}
                  onClick={() => submitStatus("suspend")}
                >
                  Suspend license
                </Button>
              )}
              <p className="mt-2 text-xs text-muted-foreground">
                {org.status === "pending"
                  ? "A pending org has never been activated — there is nothing to suspend yet."
                  : "Suspending blocks the org's members from spending the pool. Credits are left untouched."}
              </p>
            </section>
            <section>
              <SectionLabel>Partner API</SectionLabel>
              <div className="flex items-start justify-between gap-4 rounded-lg border border-border p-3">
                <p className="text-xs text-muted-foreground">
                  Lets this org&apos;s admins issue API keys and run calculations against its pool.
                  On by default for enterprise orgs; a self-serve team needs it flipped here.
                  Turning it off makes every one of their keys stop working immediately.
                </p>
                <Switch
                  aria-label="Partner API enabled"
                  checked={org.partnerApiEnabled}
                  disabled={setPartnerApi.isPending}
                  onCheckedChange={(enabled) =>
                    setPartnerApi.mutate(
                      { orgId: org.id, enabled },
                      {
                        onSuccess: () => toast.success(enabled ? "Partner API enabled." : "Partner API disabled."),
                        onError: (e) => toast.error(e instanceof Error ? e.message : "Change failed."),
                      },
                    )
                  }
                />
              </div>
            </section>
          </TabsContent>

          <TabsContent value="credits" className="space-y-6 pt-4">
            <section>
              <SectionLabel>Shared pool</SectionLabel>
              <div className="rounded-lg border border-border p-3">
                <div className="font-mono text-2xl font-semibold tabular-nums">
                  {pool.toLocaleString()}
                </div>
                <div className="text-xs text-muted-foreground">
                  {org.bundleBalance.toLocaleString()} bundle + {org.reserveBalance.toLocaleString()}{" "}
                  reserve · {org.monthlyDispersalCredits.toLocaleString()}/mo dispersal
                </div>
              </div>
            </section>

            <section>
              <SectionLabel>Gift credits</SectionLabel>
              <div className="space-y-2 rounded-lg border border-border p-3">
                <div className="flex gap-2">
                  <Input
                    type="number"
                    min={1}
                    placeholder="Amount"
                    value={giftAmount}
                    onChange={(e) => setGiftAmount(e.target.value)}
                    className="w-32"
                  />
                  <Input
                    placeholder="Reason (shows in the ledger)"
                    value={giftReason}
                    onChange={(e) => setGiftReason(e.target.value)}
                    className="flex-1"
                  />
                </div>
                {willActivate && (
                  <p className="text-xs text-amber-600">
                    This grant will activate {org.name ?? "this organization"} — activation is
                    permanent and a clawback won&apos;t reverse it.
                  </p>
                )}
                <Button size="sm" onClick={submitGift} disabled={!validAmount || grantCredits.isPending}>
                  Gift credits
                </Button>
              </div>
            </section>

            <section>
              <SectionLabel>Monthly dispersal</SectionLabel>
              <div className="space-y-2 rounded-lg border border-border p-3">
                <p className="text-xs text-muted-foreground">
                  Credits added to the pool each period by the daily sweep. Contract volume — set by
                  Msanii admins only.
                </p>
                <div className="flex gap-2">
                  <Input
                    type="number"
                    min={0}
                    value={dispersalValue}
                    onChange={(e) => setDispersalDraft(e.target.value)}
                    className="w-40"
                  />
                  <Button
                    size="sm"
                    onClick={submitDispersal}
                    disabled={setDispersal.isPending || dispersalDraft == null}
                  >
                    Save
                  </Button>
                </div>
              </div>
            </section>

            {(poolQuery.data?.ledger?.length ?? 0) > 0 && (
              <section>
                <SectionLabel>Recent activity</SectionLabel>
                {poolQuery.data?.ledger.slice(0, 10).map((entry, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between gap-3 border-b border-border/60 py-2 text-[13px] last:border-b-0"
                  >
                    <div className="min-w-0">
                      <div className="truncate">{entry.kind}</div>
                      {entry.created_at && (
                        <div className="text-xs text-muted-foreground">
                          {new Date(entry.created_at).toLocaleDateString()}
                        </div>
                      )}
                    </div>
                    <span
                      className={
                        entry.delta > 0
                          ? "font-mono font-semibold tabular-nums text-emerald-600"
                          : "font-mono font-semibold tabular-nums text-muted-foreground"
                      }
                    >
                      {entry.delta > 0 ? "+" : ""}
                      {entry.delta.toLocaleString()}
                    </span>
                  </div>
                ))}
              </section>
            )}
          </TabsContent>

          <TabsContent value="usage" className="space-y-6 pt-4">
            {/* The org's own Usage card, read-only, against the /admin routes. */}
            <OrgUsageAnalysis
              orgId={org.id}
              partnerApiEnabled={org.partnerApiEnabled}
              useUsage={useAdminOrgUsage}
              useKeys={useAdminPartnerKeys}
              reportPath={`/admin/orgs/${org.id}/usage/report.pdf`}
            />
            {org.partnerApiEnabled && <AdminOrgKeys orgId={org.id} />}
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}

// Live keys are the ALARMING state here — you are on this screen because you
// think one leaked. Already-dead keys are the reassuring answer.
const TRACE_TONE = { active: "warn", revoked: "ok", expired: "ok" } as const;

/** Trace a key found in the wild (a public repo, a log, a support ticket) back
 * to the org that owns it. What is pasted goes straight into a POST body and
 * only its first 12 characters are used, so the secret never reaches a URL, an
 * access log or the query cache. Revoking still happens in the org's console —
 * this answers WHERE. */
function KeyTrace({ onSelectOrg }: { onSelectOrg: (id: string) => void }) {
  const [pasted, setPasted] = useState("");
  const [pending, setPending] = useState<KeyTraceHit | null>(null);
  const lookup = useAdminKeyLookup();
  const revoke = useAdminRevokePartnerKey();
  const hits = lookup.data?.keys;

  const confirmRevoke = () => {
    if (!pending) return;
    revoke.mutate(
      { orgId: pending.org_id, keyId: pending.id },
      {
        onSuccess: () => {
          toast.success(`Revoked "${pending.label}".`);
          // Re-trace off the key's OWN prefix, not the box (which may have been
          // edited since), so the row redraws as revoked.
          lookup.mutate(pending.key_prefix);
        },
        onError: (e) => toast.error(e instanceof Error ? e.message : "Revoke failed."),
        onSettled: () => setPending(null),
      },
    );
  };

  return (
    <Card className="mb-3 p-4">
      <SectionLabel>Trace an API key</SectionLabel>
      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (pasted.trim()) lookup.mutate(pasted.trim());
        }}
      >
        <div className="relative w-full max-w-sm">
          <KeyRound className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="mk_live_…"
            value={pasted}
            onChange={(e) => setPasted(e.target.value)}
            className="pl-9 font-mono"
            autoComplete="off"
            spellCheck={false}
          />
        </div>
        <Button type="submit" disabled={!pasted.trim() || lookup.isPending}>
          {lookup.isPending ? "Looking up…" : "Look up"}
        </Button>
        <p className="text-[11.5px] text-muted-foreground">
          Paste the prefix or the whole key — only the first 12 characters are sent.
        </p>
      </form>

      {lookup.isError && (
        <p className="mt-3 text-xs text-destructive">That isn&apos;t a Msanii API key.</p>
      )}
      {hits?.length === 0 && (
        <p className="mt-3 text-xs text-muted-foreground">No key matches that prefix.</p>
      )}
      {hits?.map((k) => (
        <div key={k.id} className="mt-3 border-t border-border pt-3 first:border-t-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[13px] font-medium">{k.label}</span>
            <Tag tone={TRACE_TONE[k.status]}>{k.status}</Tag>
            <Button
              variant="link"
              className="h-auto p-0 text-[13px]"
              onClick={() => onSelectOrg(k.org_id)}
            >
              {k.org_name ?? "Open team"}
            </Button>
            <span className="text-[11.5px] text-muted-foreground">
              last used {shortDate(k.last_used_at)}
            </span>
            {/* Only a live key can be killed — revoking again would just reset
                the 30-day clock that hides it from the console. */}
            {k.status === "active" && (
              <Button
                variant="ghost"
                size="sm"
                className="ml-auto text-destructive hover:text-destructive"
                aria-label={`Revoke ${k.label}`}
                onClick={() => setPending(k)}
              >
                Revoke
              </Button>
            )}
          </div>
          <p className="mt-1 text-[11.5px] text-muted-foreground">
            {k.recent_ips.length === 0
              ? "No requests in the last 7 days."
              : `Source IPs (7d): ${k.recent_ips.map((s) => `${s.ip} ×${s.requests}`).join(", ")}`}
          </p>
        </div>
      ))}

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              Revoke &quot;{pending?.label}&quot; for {pending?.org_name ?? "this team"}?
            </DialogTitle>
            <DialogDescription>
              Anything using this key stops working on its next request. This can&apos;t be undone —
              the team has to mint a replacement, and nobody there is told automatically.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRevoke} disabled={revoke.isPending}>
              {revoke.isPending && <Loader2 className="mr-1.5 h-4 w-4 animate-spin" />}
              Revoke key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}

/** Read-only: minting and revoking stay in the org's own console. */
function AdminOrgKeys({ orgId }: { orgId: string }) {
  const { data, isLoading, isError } = useAdminPartnerKeys(orgId);
  const folderName = new Map((data?.folders ?? []).map((f) => [f.id, f.name]));
  const keys = data?.keys ?? [];

  return (
    <section>
      <SectionLabel>API keys</SectionLabel>
      {isLoading && <p className="text-xs text-muted-foreground">Loading keys…</p>}
      {isError && <p className="text-xs text-destructive">Couldn&apos;t load API keys.</p>}
      {!isLoading && !isError && keys.length === 0 && (
        <p className="text-xs text-muted-foreground">No API keys yet.</p>
      )}
      {keys.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-border text-left text-[10.5px] uppercase tracking-wider text-muted-foreground">
                <th className="py-2 pr-3 font-semibold">Name</th>
                <th className="py-2 pr-3 font-semibold">Folder</th>
                <th className="py-2 pr-3 font-semibold">Status</th>
                <th className="py-2 pr-3 font-semibold">Created by</th>
                <th className="py-2 font-semibold">Last used</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k.id} className="border-b border-border/60 last:border-b-0">
                  <td className="py-2 pr-3 font-medium">{k.label}</td>
                  <td className="py-2 pr-3 text-muted-foreground">
                    {(k.folder_id && folderName.get(k.folder_id)) || "—"}
                  </td>
                  <td className="py-2 pr-3 capitalize text-muted-foreground">{keyStatus(k)}</td>
                  <td className="py-2 pr-3 text-muted-foreground">{k.created_by_label ?? "—"}</td>
                  <td className="py-2 text-muted-foreground">{shortDate(k.last_used_at) || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
