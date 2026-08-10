// src/components/admin/AdminUsersPanel.tsx
// Admin console → Users: search + filter chips + table, and the per-user
// drawer (Access / Credits / Usage).
//
// Search is server-side and paginated; the filter chips narrow the CURRENT
// page only (the listing endpoint has no tier/role predicates, and faking a
// global count from one page would be a lie). The header says so.
import { useState } from "react";
import { toast } from "sonner";
import { ChevronRight, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { useAuth } from "@/contexts/AuthContext";
import { useAnalytics } from "@/hooks/useAnalytics";
import {
  useAdminUsers,
  useAdminMutations,
  type AdminUserRow,
  type OverridePayloadInput,
} from "@/hooks/useAdmin";
import { useEntitlementsForUser, type RawOverride } from "@/hooks/useEntitlements";
import {
  useTesterGrants,
  useCreateTesterGrant,
  useRevokeTesterGrant,
} from "@/hooks/useTesterGrants";
import { isPaidTier, tierLabel } from "@/lib/tiers";
import { UserCreditsCard } from "@/components/admin/UserCreditsCard";
import { shortDate, Tag } from "@/components/admin/ui";

export type UserFilter = "all" | "paid" | "free" | "testers" | "admins" | "override";

const FILTER_LABELS: { id: UserFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "paid", label: "Paid" },
  { id: "free", label: "Free" },
  { id: "testers", label: "Testers" },
  { id: "admins", label: "Admins" },
  { id: "override", label: "Overrides" },
];

export const formatBytes = (bytes: number): string => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
};

interface UsersPanelProps {
  search: string;
  onSearchChange: (v: string) => void;
  page: number;
  onPageChange: (p: number) => void;
  filter: UserFilter;
  onFilterChange: (f: UserFilter) => void;
  selectedUserId: string | null;
  onSelectUser: (id: string | null) => void;
  onOpenOrg: (orgId: string) => void;
  /** Jumps to the Beta testers view, where the grant form lives. */
  onGrantTester: () => void;
}

export function AdminUsersPanel({
  search,
  onSearchChange,
  page,
  onPageChange,
  filter,
  onFilterChange,
  selectedUserId,
  onSelectUser,
  onOpenOrg,
  onGrantTester,
}: UsersPanelProps) {
  const usersQuery = useAdminUsers(search, page);
  const grantsQuery = useTesterGrants();
  const testerIds = new Set(
    (grantsQuery.data ?? []).filter((g) => g.user_id).map((g) => g.user_id as string),
  );

  const rows = usersQuery.data?.users ?? [];
  const filtered = rows.filter((u) => matchesFilter(u, filter, testerIds));

  return (
    <>
      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="relative w-full max-w-sm">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search by email…"
            value={search}
            onChange={(e) => {
              onSearchChange(e.target.value);
              onPageChange(1);
            }}
            className="pl-9"
          />
        </div>
        <span className="ml-auto text-[11.5px] text-muted-foreground">
          {filtered.length}
          {filter !== "all" ? ` of ${rows.length}` : ""} on this page
        </span>
        <Button size="sm" onClick={onGrantTester}>
          Grant tester access
        </Button>
      </div>

      <div className="mb-3 flex flex-wrap gap-1.5">
        {FILTER_LABELS.map((f) => (
          <button
            key={f.id}
            onClick={() => onFilterChange(f.id)}
            aria-pressed={filter === f.id}
            className={cn(
              "rounded-full border px-3 py-1 text-[12.5px] font-medium transition-colors",
              filter === f.id
                ? "border-primary bg-primary text-primary-foreground"
                : "border-border bg-card text-muted-foreground hover:border-primary/40",
            )}
          >
            {f.label}
            <span className="ml-1.5 font-mono text-[11px] opacity-70">
              {rows.filter((u) => matchesFilter(u, f.id, testerIds)).length}
            </span>
          </button>
        ))}
      </div>

      <Card className="overflow-hidden p-0">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-[10.5px] uppercase tracking-wider text-muted-foreground">
                <th className="px-4 py-2.5 font-semibold">Name</th>
                <th className="px-4 py-2.5 font-semibold">Email</th>
                <th className="px-4 py-2.5 font-semibold">Plan</th>
                <th className="px-4 py-2.5 font-semibold">Access</th>
                <th className="px-4 py-2.5 font-semibold">Joined</th>
                <th className="w-8 px-4 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {usersQuery.isLoading && (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">
                    Loading…
                  </td>
                </tr>
              )}
              {usersQuery.error && (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-destructive">
                    Couldn&apos;t load users — please refresh.
                  </td>
                </tr>
              )}
              {!usersQuery.isLoading && !usersQuery.error && filtered.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">
                    No users on this page match.
                  </td>
                </tr>
              )}
              {filtered.map((u) => (
                <tr
                  key={u.id}
                  onClick={() => onSelectUser(u.id)}
                  className="cursor-pointer border-b border-border/60 last:border-b-0 hover:bg-muted/40"
                >
                  <td className="px-4 py-2.5 font-medium">
                    {u.name ?? <span className="text-muted-foreground">—</span>}
                  </td>
                  <td className="px-4 py-2.5 text-muted-foreground">
                    {u.email ?? "—"}
                  </td>
                  <td className="px-4 py-2.5">
                    <Tag tone={isPaidTier(u.tier) ? "paid" : "neutral"}>{tierLabel(u.tier)}</Tag>
                  </td>
                  <td className="px-4 py-2.5">
                    <div className="flex flex-wrap gap-1">
                      {testerIds.has(u.id) && <Tag tone="warn">Tester</Tag>}
                      {(u.is_admin || u.is_env_admin) && (
                        <Tag tone="info">{u.is_env_admin ? "Admin (env)" : "Admin"}</Tag>
                      )}
                      {u.has_override && <Tag>Override</Tag>}
                      {!testerIds.has(u.id) && !u.is_admin && !u.is_env_admin && !u.has_override && (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-xs text-muted-foreground">
                    {shortDate(u.created_at)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-muted-foreground">
                    <ChevronRight className="h-4 w-4" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="mt-3 flex items-center justify-end gap-2">
        <Button
          size="sm"
          variant="outline"
          disabled={page <= 1}
          onClick={() => onPageChange(Math.max(1, page - 1))}
        >
          ← Prev
        </Button>
        <span className="text-sm text-muted-foreground">Page {page}</span>
        <Button
          size="sm"
          variant="outline"
          disabled={!usersQuery.data?.has_more}
          onClick={() => onPageChange(page + 1)}
        >
          Next →
        </Button>
      </div>

      <UserDetailSheet
        key={selectedUserId ?? "closed"}
        userId={selectedUserId}
        onClose={() => onSelectUser(null)}
        onOpenOrg={onOpenOrg}
      />
    </>
  );
}

function matchesFilter(u: AdminUserRow, filter: UserFilter, testerIds: Set<string>): boolean {
  switch (filter) {
    case "paid":
      return isPaidTier(u.tier);
    case "free":
      return !isPaidTier(u.tier);
    case "testers":
      return testerIds.has(u.id);
    case "admins":
      return u.is_admin || u.is_env_admin;
    case "override":
      return u.has_override;
    default:
      return true;
  }
}

// ---------------------------------------------------------------------------
// User-detail drawer
// ---------------------------------------------------------------------------

interface SheetProps {
  userId: string | null;
  onClose: () => void;
  onOpenOrg: (orgId: string) => void;
}

function UserDetailSheet({ userId, onClose, onOpenOrg }: SheetProps) {
  const detailQuery = useEntitlementsForUser(userId);
  const { grantPro, revokePro, clearOverride, promoteAdmin, demoteAdmin, recalcStorage } =
    useAdminMutations();
  const grantsQuery = useTesterGrants();
  const createGrant = useCreateTesterGrant();
  const revokeGrant = useRevokeTesterGrant();
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [testerDays, setTesterDays] = useState("");
  const [testerCredits, setTesterCredits] = useState("");
  const { user: currentUser } = useAuth();
  const { captureAdminUserPromoted, captureAdminUserDemoted } = useAnalytics();

  const data = detailQuery.data;
  const grant = (grantsQuery.data ?? []).find((g) => g.user_id && g.user_id === userId) ?? null;

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    try {
      await fn();
      toast.success(ok);
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  return (
    <Sheet open={!!userId} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-lg">
        <SheetHeader>
          <SheetTitle className="truncate">
            {data ? (data.user.name ?? data.user.email ?? data.user.id) : "Loading…"}
          </SheetTitle>
          {data?.user.name && data.user.email && (
            <p className="truncate text-sm text-muted-foreground">{data.user.email}</p>
          )}
        </SheetHeader>

        {detailQuery.isLoading && (
          <div className="flex items-center justify-center py-10">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        )}

        {data && (
          <>
            <div className="mt-3 flex flex-wrap gap-1.5">
              <Tag tone={isPaidTier(data.entitlements.tier) ? "paid" : "neutral"}>
                {tierLabel(data.entitlements.tier)}
              </Tag>
              {(data.user.is_admin || data.user.is_env_admin) && (
                <Tag tone="info">{data.user.is_env_admin ? "Admin (env)" : "Platform admin"}</Tag>
              )}
              {grant && <Tag tone="warn">Beta tester</Tag>}
              {data.entitlements.hasOverrides && <Tag>Override</Tag>}
              <span className="text-[11.5px] text-muted-foreground">
                Joined {shortDate(data.user.created_at)}
              </span>
            </div>

            <Tabs defaultValue="access" className="mt-4">
              <TabsList className="w-full justify-start">
                <TabsTrigger value="access">Access</TabsTrigger>
                <TabsTrigger value="credits">Credits</TabsTrigger>
                <TabsTrigger value="usage">Usage</TabsTrigger>
              </TabsList>

              <TabsContent value="access" className="space-y-6 pt-4">
                <section>
                  <SectionLabel>Plan</SectionLabel>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={isPaidTier(data.entitlements.tier) ? "default" : "outline"}>
                      {tierLabel(data.entitlements.tier)}
                    </Badge>
                    {!isPaidTier(data.entitlements.tier) ? (
                      <Button
                        size="sm"
                        disabled={grantPro.isPending}
                        onClick={() =>
                          run(() => grantPro.mutateAsync(userId!), `Granted ${tierLabel("basic")}`)
                        }
                      >
                        Grant {tierLabel("basic")}
                      </Button>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        disabled={revokePro.isPending}
                        onClick={() =>
                          run(
                            () => revokePro.mutateAsync(userId!),
                            `Revoked ${tierLabel(data.entitlements.tier)}`,
                          )
                        }
                      >
                        Revoke {tierLabel(data.entitlements.tier)}
                      </Button>
                    )}
                  </div>
                  {/* The /grant endpoint always sets tier="basic" — there is no
                      pro-grant path to expose here without a backend change. */}
                </section>

                <section>
                  <SectionLabel>Beta tester access</SectionLabel>
                  <div className="rounded-lg border border-border p-3">
                    {grant ? (
                      <>
                        <KeyValue k="Status" v="Active" />
                        <KeyValue
                          k="Expires"
                          v={grant.expires_at ? shortDate(grant.expires_at) : "Never"}
                        />
                        <KeyValue k="Reason" v={grant.reason || "tester"} />
                        <Button
                          size="sm"
                          variant="outline"
                          className="mt-3 text-destructive hover:text-destructive"
                          disabled={revokeGrant.isPending}
                          onClick={() =>
                            run(
                              () => revokeGrant.mutateAsync(userId!),
                              "Tester access revoked.",
                            )
                          }
                        >
                          Revoke tester access
                        </Button>
                      </>
                    ) : (
                      <>
                        <div className="grid grid-cols-2 gap-2">
                          <label className="text-xs font-medium text-muted-foreground">
                            Expires after (days)
                            <Input
                              type="number"
                              min={1}
                              max={3650}
                              placeholder="No expiry"
                              value={testerDays}
                              onChange={(e) => setTesterDays(e.target.value)}
                              className="mt-1"
                            />
                          </label>
                          <label className="text-xs font-medium text-muted-foreground">
                            Starting credits
                            <Input
                              type="number"
                              min={1}
                              max={1_000_000}
                              placeholder="Default"
                              value={testerCredits}
                              onChange={(e) => setTesterCredits(e.target.value)}
                              className="mt-1"
                            />
                          </label>
                        </div>
                        <Button
                          size="sm"
                          className="mt-3"
                          disabled={!data.user.email || createGrant.isPending}
                          onClick={() =>
                            run(
                              () =>
                                createGrant.mutateAsync({
                                  email: data.user.email!,
                                  grant_duration_days: testerDays ? Number(testerDays) : null,
                                  credits: testerCredits ? Number(testerCredits) : null,
                                  reason: "tester",
                                }),
                              "Tester access granted.",
                            )
                          }
                        >
                          Grant tester access
                        </Button>
                      </>
                    )}
                  </div>
                </section>

                <section>
                  <SectionLabel>Platform role</SectionLabel>
                  <div className="flex flex-wrap items-center gap-2">
                    {data.user.is_env_admin ? (
                      <span className="text-xs text-muted-foreground">
                        Managed via ADMIN_EMAILS — edit env to revoke.
                      </span>
                    ) : data.user.is_admin ? (
                      currentUser?.id === userId ? (
                        <span className="text-xs text-muted-foreground">
                          You can&apos;t demote yourself.
                        </span>
                      ) : (
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={demoteAdmin.isPending}
                          onClick={() =>
                            run(async () => {
                              await demoteAdmin.mutateAsync(userId!);
                              captureAdminUserDemoted(userId!);
                            }, "Demoted from admin")
                          }
                        >
                          Demote from admin
                        </Button>
                      )
                    ) : (
                      <Button
                        size="sm"
                        disabled={promoteAdmin.isPending}
                        onClick={() =>
                          run(async () => {
                            await promoteAdmin.mutateAsync(userId!);
                            captureAdminUserPromoted(userId!);
                          }, "Promoted to admin")
                        }
                      >
                        Promote to admin
                      </Button>
                    )}
                  </div>
                </section>

                <section>
                  <SectionLabel>Limit override</SectionLabel>
                  <div className="rounded-lg border border-border p-3">
                    <KeyValue
                      k="Status"
                      v={data.entitlements.hasOverrides ? "Active" : "Tier defaults"}
                    />
                    <div className="mt-3 flex gap-2">
                      <Button size="sm" variant="outline" onClick={() => setOverrideOpen(true)}>
                        {data.entitlements.hasOverrides ? "Edit override" : "Apply override"}
                      </Button>
                      {data.entitlements.hasOverrides && (
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={clearOverride.isPending}
                          onClick={() =>
                            run(() => clearOverride.mutateAsync(userId!), "Override cleared")
                          }
                        >
                          Clear
                        </Button>
                      )}
                    </div>
                  </div>
                  {overrideOpen && userId && (
                    <OverrideEditor
                      userId={userId}
                      currentOverride={data.override ?? null}
                      onDone={() => setOverrideOpen(false)}
                    />
                  )}
                </section>
              </TabsContent>

              <TabsContent value="credits" className="pt-1">
                {data.entitlements.credits ? (
                  <UserCreditsCard
                    userId={userId!}
                    entitlements={data.entitlements}
                    onOpenOrg={onOpenOrg}
                  />
                ) : (
                  <p className="pt-4 text-sm text-muted-foreground">
                    Credits are off for this deployment (CREDITS_ENABLED), so this user has no
                    wallet to administer.
                  </p>
                )}
              </TabsContent>

              <TabsContent value="usage" className="space-y-6 pt-4">
                <section>
                  <SectionLabel>This period</SectionLabel>
                  {[
                    { tool: "Zoe queries", n: data.entitlements.usage.zoeQueriesThisPeriod },
                    { tool: "OneClick runs", n: data.entitlements.usage.oneclickRunsThisPeriod },
                    { tool: "Split sheets", n: data.entitlements.usage.splitSheetsThisPeriod },
                  ].map((row) => (
                    <div key={row.tool} className="mb-3">
                      <div className="mb-1 flex items-center justify-between text-[13px]">
                        <span>{row.tool}</span>
                        <span className="font-mono tabular-nums text-muted-foreground">{row.n}</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full bg-accent"
                          style={{ width: `${Math.min(100, row.n * 2)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </section>

                <section>
                  <SectionLabel>Account</SectionLabel>
                  <div className="flex items-center justify-between gap-2 border-b border-dashed border-border py-1.5 text-[13px]">
                    <span>Storage used</span>
                    <span className="flex items-center gap-1.5">
                      <b className="font-mono tabular-nums">
                        {formatBytes(data.entitlements.usage.totalStorageBytes)}
                      </b>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-6 px-2 text-xs"
                        disabled={recalcStorage.isPending}
                        onClick={async () => {
                          try {
                            const res = await recalcStorage.mutateAsync(userId!);
                            toast.success(
                              `Storage recalculated: ${formatBytes(res.total_storage_bytes)}`,
                            );
                          } catch (e) {
                            toast.error(`Failed: ${(e as Error).message}`);
                          }
                        }}
                      >
                        {recalcStorage.isPending ? "…" : "Recalculate"}
                      </Button>
                    </span>
                  </div>
                  <KeyValue k="Joined" v={shortDate(data.user.created_at)} />
                </section>
              </TabsContent>
            </Tabs>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 text-[10.5px] font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </div>
  );
}

function KeyValue({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-dashed border-border py-1.5 text-[13px] last:border-b-0">
      <span>{k}</span>
      <b className="font-mono tabular-nums">{v}</b>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Override editor (inline form within the drawer)
// ---------------------------------------------------------------------------

interface OverrideEditorProps {
  userId: string;
  currentOverride: RawOverride | null;
  onDone: () => void;
}

// Normalize a number-or-null override field to its string form for an <input>,
// treating -1 (the "unlimited" sentinel) as empty since unlimited is a checkbox.
const numToInputOrEmpty = (v: number | null | undefined): string =>
  v === null || v === undefined || v === -1 ? "" : String(v);

// Normalize a tri-state bool override field to the select's value.
const boolToSelect = (v: boolean | null | undefined): "" | "true" | "false" =>
  v === null || v === undefined ? "" : v ? "true" : "false";

const ALL_INTEGRATIONS = ["google_drive"] as const;

function OverrideEditor({ userId, currentOverride, onDone }: OverrideEditorProps) {
  const { applyOverride } = useAdminMutations();

  // Inputs pre-fill from currentOverride and are re-submitted with their current
  // values (the editor doesn't omit unchanged fields). Why this is safe: Supabase
  // upsert uses PostgREST merge-duplicates → INSERT ... ON CONFLICT DO UPDATE SET
  // col = EXCLUDED.col for fields in the payload only. Re-submitting the same
  // value is a no-op write; columns not in the payload are not touched.
  //
  // Known UX limitation: clearing an input (deleting the pre-filled value) is
  // silently ignored — handleSubmit skips empty fields, so the existing override
  // value stays in the DB. To remove individual override fields, admins must use
  // "Clear override" (full row delete) and re-apply with only the fields they
  // want to keep. Helper text below the form makes this explicit.
  const [maxArtists, setMaxArtists] = useState(numToInputOrEmpty(currentOverride?.max_artists));
  const [maxProjects, setMaxProjects] = useState(numToInputOrEmpty(currentOverride?.max_projects));
  const [maxTasks, setMaxTasks] = useState(numToInputOrEmpty(currentOverride?.max_tasks));
  // Storage stored as bytes in DB; admin enters GB for ergonomics
  const [maxStorageGb, setMaxStorageGb] = useState(
    currentOverride?.max_storage_bytes != null && currentOverride.max_storage_bytes !== -1
      ? String(currentOverride.max_storage_bytes / (1024 * 1024 * 1024))
      : "",
  );
  const [maxSplitSheets, setMaxSplitSheets] = useState(
    numToInputOrEmpty(currentOverride?.max_split_sheets_per_month),
  );

  // "Unlimited" flags — when true, the corresponding input is disabled and -1
  // is sent on submit. Pre-fill from currentOverride: -1 → unlimited checked.
  const [maxArtistsUnlimited, setMaxArtistsUnlimited] = useState(
    currentOverride?.max_artists === -1,
  );
  const [maxProjectsUnlimited, setMaxProjectsUnlimited] = useState(
    currentOverride?.max_projects === -1,
  );
  const [maxTasksUnlimited, setMaxTasksUnlimited] = useState(currentOverride?.max_tasks === -1);
  const [maxStorageUnlimited, setMaxStorageUnlimited] = useState(
    currentOverride?.max_storage_bytes === -1,
  );
  const [maxSplitSheetsUnlimited, setMaxSplitSheetsUnlimited] = useState(
    currentOverride?.max_split_sheets_per_month === -1,
  );

  // Feature flags (tri-state)
  const [zoe, setZoe] = useState(boolToSelect(currentOverride?.zoe_enabled));
  const [oneclick, setOneclick] = useState(boolToSelect(currentOverride?.oneclick_enabled));
  const [registry, setRegistry] = useState(boolToSelect(currentOverride?.registry_enabled));

  // Integrations — checkboxes; null override = "use tier default"
  const [integrationsOverridden, setIntegrationsOverridden] = useState(
    currentOverride?.integrations_allowed != null,
  );
  const [integrations, setIntegrations] = useState<string[]>(
    currentOverride?.integrations_allowed ?? [],
  );

  const [reason, setReason] = useState(currentOverride?.reason ?? "");
  const [expiresDays, setExpiresDays] = useState<string>(""); // expiry is set anew each save

  const toggleIntegration = (name: string) => {
    setIntegrations((prev) =>
      prev.includes(name) ? prev.filter((i) => i !== name) : [...prev, name],
    );
  };

  const handleSubmit = async () => {
    const payload: OverridePayloadInput = {};

    // Caps — Unlimited flag wins; otherwise parse the number input if non-empty.
    if (maxArtistsUnlimited) payload.max_artists = -1;
    else if (maxArtists.trim() !== "") payload.max_artists = parseInt(maxArtists, 10);

    if (maxProjectsUnlimited) payload.max_projects = -1;
    else if (maxProjects.trim() !== "") payload.max_projects = parseInt(maxProjects, 10);

    if (maxTasksUnlimited) payload.max_tasks = -1;
    else if (maxTasks.trim() !== "") payload.max_tasks = parseInt(maxTasks, 10);

    if (maxStorageUnlimited) payload.max_storage_bytes = -1;
    else if (maxStorageGb.trim() !== "")
      payload.max_storage_bytes = Math.round(parseFloat(maxStorageGb) * 1024 * 1024 * 1024);

    if (maxSplitSheetsUnlimited) payload.max_split_sheets_per_month = -1;
    else if (maxSplitSheets.trim() !== "")
      payload.max_split_sheets_per_month = parseInt(maxSplitSheets, 10);

    // Feature toggles
    if (zoe) payload.zoe_enabled = zoe === "true";
    if (oneclick) payload.oneclick_enabled = oneclick === "true";
    if (registry) payload.registry_enabled = registry === "true";

    // Integrations — only include if "override integrations" is checked
    if (integrationsOverridden) payload.integrations_allowed = integrations;

    if (reason.trim()) payload.reason = reason.trim();
    if (expiresDays.trim()) payload.expires_days = parseInt(expiresDays, 10);

    if (Object.keys(payload).length === 0) {
      toast.error("Set at least one field");
      return;
    }

    try {
      await applyOverride.mutateAsync({ userId, payload });
      toast.success("Override applied");
      onDone();
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  const capField = (
    label: string,
    value: string,
    setValue: (v: string) => void,
    unlimited: boolean,
    setUnlimited: (v: boolean) => void,
    step?: string,
  ) => (
    <div className="space-y-1 text-xs">
      <label className="block">
        {label}
        <Input
          type="number"
          step={step}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="—"
          disabled={unlimited}
          className={unlimited ? "opacity-50" : ""}
        />
      </label>
      <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <input
          type="checkbox"
          checked={unlimited}
          onChange={(e) => setUnlimited(e.target.checked)}
        />
        Unlimited
      </label>
    </div>
  );

  return (
    <Card className="mt-3 space-y-3 p-4">
      <div className="text-sm font-medium">
        Override fields{" "}
        {currentOverride ? "(pre-filled with current values)" : "(only set what you want to override)"}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {capField("Max artists", maxArtists, setMaxArtists, maxArtistsUnlimited, setMaxArtistsUnlimited)}
        {capField("Max projects", maxProjects, setMaxProjects, maxProjectsUnlimited, setMaxProjectsUnlimited)}
        {capField("Max tasks", maxTasks, setMaxTasks, maxTasksUnlimited, setMaxTasksUnlimited)}
        {capField("Max storage (GB)", maxStorageGb, setMaxStorageGb, maxStorageUnlimited, setMaxStorageUnlimited, "0.1")}
        {capField("Max split sheets / month", maxSplitSheets, setMaxSplitSheets, maxSplitSheetsUnlimited, setMaxSplitSheetsUnlimited)}
      </div>

      <div className="grid grid-cols-3 gap-3">
        {(
          [
            ["Zoe enabled", zoe, setZoe],
            ["OneClick enabled", oneclick, setOneclick],
            ["Registry enabled", registry, setRegistry],
          ] as const
        ).map(([label, value, setValue]) => (
          <label key={label} className="text-xs">
            {label}
            <select
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
              value={value}
              onChange={(e) => setValue(e.target.value as "" | "true" | "false")}
            >
              <option value="">— (use tier default)</option>
              <option value="true">Yes</option>
              <option value="false">No</option>
            </select>
          </label>
        ))}
      </div>

      <div className="space-y-2">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={integrationsOverridden}
            onChange={(e) => setIntegrationsOverridden(e.target.checked)}
          />
          Override integrations (uncheck to use tier default)
        </label>
        {integrationsOverridden && (
          <div className="flex flex-wrap gap-3 pl-6">
            {ALL_INTEGRATIONS.map((name) => (
              <label key={name} className="flex items-center gap-2 text-xs">
                <input
                  type="checkbox"
                  checked={integrations.includes(name)}
                  onChange={() => toggleIntegration(name)}
                />
                {name.replace("_", " ")}
              </label>
            ))}
          </div>
        )}
      </div>

      <label className="block text-xs">
        Reason (optional)
        <Input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Beta tester comp" />
      </label>
      <label className="block text-xs">
        Expires in N days (optional, leave empty for no expiry)
        <Input
          type="number"
          value={expiresDays}
          onChange={(e) => setExpiresDays(e.target.value)}
          placeholder="—"
        />
      </label>

      <p className="border-t border-border pt-3 text-xs text-muted-foreground">
        Heads up: deleting a pre-filled value or switching a feature back to &ldquo;—&rdquo; is
        silently ignored — Save preserves the existing override field. To remove individual
        overrides, use <strong>Clear override</strong> and re-apply with only the fields you want to
        keep.
      </p>

      <div className="flex justify-end gap-2 pt-1">
        <Button variant="ghost" size="sm" onClick={onDone}>
          Cancel
        </Button>
        <Button size="sm" onClick={handleSubmit} disabled={applyOverride.isPending}>
          Save
        </Button>
      </div>
    </Card>
  );
}
