import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Music, Search } from "lucide-react";
import { useNavigate } from "react-router-dom";
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
  ApiError,
} from "@/hooks/useTesterGrants";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import { useAnalytics } from "@/hooks/useAnalytics";

const formatBytes = (bytes: number): string => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
};

// ---------------------------------------------------------------------------
// TesterGrantsPanel — grant / revoke beta tester access
// ---------------------------------------------------------------------------

const TesterGrantsPanel = () => {
  const grantsQuery = useTesterGrants();
  const createGrant = useCreateTesterGrant();
  const revokeGrant = useRevokeTesterGrant();

  const [email, setEmail] = useState("");
  const [expiresAt, setExpiresAt] = useState("");
  const [reason, setReason] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;

    try {
      await createGrant.mutateAsync({
        email: email.trim(),
        expires_at: expiresAt ? expiresAt : null,
        reason: reason.trim() || "tester",
      });
      toast.success(`Granted tester access to ${email.trim()}`);
      setEmail("");
      setExpiresAt("");
      setReason("");
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        toast.error(`User ${email.trim()} hasn't signed up yet. Ask them to register first.`);
      } else {
        toast.error("Failed to grant tester access. Try again.");
      }
    }
  };

  const handleRevoke = async (userId: string) => {
    try {
      await revokeGrant.mutateAsync(userId);
      toast.success("Tester access revoked.");
    } catch {
      toast.error("Failed to revoke tester access. Try again.");
    }
  };

  return (
    <Card className="p-4 mb-6">
      <div className="text-sm font-semibold mb-4">Beta Tester Access</div>

      {/* Grant form */}
      <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3 mb-6">
        <label className="text-xs flex-1 min-w-[200px]">
          Email (required)
          <Input
            type="email"
            required
            placeholder="user@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-1"
          />
        </label>
        <label className="text-xs w-40">
          Expires (optional)
          <Input
            type="date"
            value={expiresAt}
            onChange={(e) => setExpiresAt(e.target.value)}
            className="mt-1"
          />
        </label>
        <label className="text-xs flex-1 min-w-[160px]">
          Reason (optional)
          <Input
            type="text"
            placeholder="tester"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="mt-1"
          />
        </label>
        <Button type="submit" size="sm" disabled={createGrant.isPending} className="mb-0.5">
          Grant Tester Access
        </Button>
      </form>

      {/* Active grants table */}
      <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Active grants</div>
      <table className="w-full text-sm">
        <thead className="bg-muted/50 text-left text-xs uppercase tracking-wide text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-medium">User ID</th>
            <th className="px-3 py-2 font-medium">Expires</th>
            <th className="px-3 py-2 font-medium text-right">Actions</th>
          </tr>
        </thead>
        <tbody>
          {grantsQuery.isLoading && (
            <tr>
              <td colSpan={3} className="px-3 py-4 text-center text-muted-foreground">
                Loading...
              </td>
            </tr>
          )}
          {!grantsQuery.isLoading && grantsQuery.data?.length === 0 && (
            <tr>
              <td colSpan={3} className="px-3 py-4 text-center text-muted-foreground">
                No active tester grants.
              </td>
            </tr>
          )}
          {grantsQuery.data?.map((grant) => (
            <tr key={grant.user_id} className="border-t border-border hover:bg-muted/30">
              <td className="px-3 py-2 font-mono text-xs">{grant.user_id}</td>
              <td className="px-3 py-2 text-muted-foreground">
                {grant.expires_at ? grant.expires_at.split("T")[0] : "Never"}
              </td>
              <td className="px-3 py-2 text-right">
                <Button
                  size="sm"
                  variant="ghost"
                  className="text-destructive hover:text-destructive"
                  onClick={() => handleRevoke(grant.user_id)}
                  disabled={revokeGrant.isPending}
                >
                  Revoke
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
};

const AdminUsers = () => {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null);

  const usersQuery = useAdminUsers(search, page);
  const posthogUrl = import.meta.env.VITE_POSTHOG_DASHBOARD_URL || "https://us.posthog.com";

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={() => navigate("/")}
          >
            <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
              <Music className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-lg font-semibold tracking-tight">Msanii Admin</span>
          </div>
          <Button variant="ghost" onClick={() => navigate("/dashboard")}>
            Back to dashboard
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <h1 className="text-2xl font-semibold tracking-tight mb-6">Users</h1>

        <div className="flex justify-end mb-4">
          <Button variant="outline" asChild>
            <a href={posthogUrl} target="_blank" rel="noopener noreferrer">
              View Analytics ↗
            </a>
          </Button>
        </div>

        <TesterGrantsPanel />

        <div className="flex items-center gap-3 mb-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by email…"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              className="pl-9"
            />
          </div>
        </div>

        <Card className="overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/50 text-left text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-4 py-3 font-medium">Email</th>
                <th className="px-4 py-3 font-medium">Tier</th>
                <th className="px-4 py-3 font-medium">Admin</th>
                <th className="px-4 py-3 font-medium">Override</th>
                <th className="px-4 py-3 font-medium">Created</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {usersQuery.isLoading && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">Loading…</td></tr>
              )}
              {usersQuery.error && (
                <tr><td colSpan={6} className="px-4 py-8 text-center text-destructive">
                  Couldn't load users — please refresh.
                </td></tr>
              )}
              {usersQuery.data?.users.map((u: AdminUserRow) => (
                <tr key={u.id} className="border-t border-border hover:bg-muted/30">
                  <td className="px-4 py-3">{u.email ?? <span className="text-muted-foreground">—</span>}</td>
                  <td className="px-4 py-3">
                    <Badge variant={u.tier === "pro" ? "default" : "outline"}>
                      {u.tier === "pro" ? "Pro" : "Free"}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    {u.is_env_admin ? (
                      <Badge variant="secondary">Admin (env)</Badge>
                    ) : u.is_admin ? (
                      <Badge variant="secondary">Admin</Badge>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {u.has_override ? "Yes" : "—"}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground text-xs">
                    {u.created_at?.split("T")[0] ?? "—"}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Button size="sm" variant="ghost" onClick={() => setSelectedUserId(u.id)}>
                      Manage →
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <div className="flex items-center justify-end gap-2 mt-4">
          <Button
            size="sm" variant="outline"
            disabled={page <= 1}
            onClick={() => setPage(p => Math.max(1, p - 1))}
          >
            ← Prev
          </Button>
          <span className="text-sm text-muted-foreground">Page {page}</span>
          <Button
            size="sm" variant="outline"
            disabled={!usersQuery.data?.has_more}
            onClick={() => setPage(p => p + 1)}
          >
            Next →
          </Button>
        </div>
      </main>

      <UserDetailSheet
        userId={selectedUserId}
        onClose={() => setSelectedUserId(null)}
      />
    </div>
  );
};

// ---------------------------------------------------------------------------
// User-detail side sheet
// ---------------------------------------------------------------------------

interface SheetProps {
  userId: string | null;
  onClose: () => void;
}

const UserDetailSheet = ({ userId, onClose }: SheetProps) => {
  const detailQuery = useEntitlementsForUser(userId);
  const { grantPro, revokePro, clearOverride, promoteAdmin, demoteAdmin } = useAdminMutations();
  const [overrideOpen, setOverrideOpen] = useState(false);
  const { user: currentUser } = useAuth();
  const { captureAdminUserPromoted, captureAdminUserDemoted } = useAnalytics();

  const handleGrant = async () => {
    if (!userId) return;
    try {
      await grantPro.mutateAsync(userId);
      toast.success("Granted Pro");
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  const handleRevoke = async () => {
    if (!userId) return;
    try {
      await revokePro.mutateAsync(userId);
      toast.success("Revoked Pro");
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  const handleClearOverride = async () => {
    if (!userId) return;
    try {
      await clearOverride.mutateAsync(userId);
      toast.success("Override cleared");
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  const handlePromote = async () => {
    if (!userId) return;
    try {
      await promoteAdmin.mutateAsync(userId);
      captureAdminUserPromoted(userId);
      toast.success("Promoted to admin");
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  const handleDemote = async () => {
    if (!userId) return;
    try {
      await demoteAdmin.mutateAsync(userId);
      captureAdminUserDemoted(userId);
      toast.success("Demoted from admin");
    } catch (e) {
      toast.error(`Failed: ${(e as Error).message}`);
    }
  };

  const data = detailQuery.data;

  return (
    <Sheet open={!!userId} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full sm:max-w-lg overflow-y-auto">
        <SheetHeader>
          <SheetTitle>{data?.user.email ?? "Loading…"}</SheetTitle>
        </SheetHeader>

        {detailQuery.isLoading && (
          <div className="py-8 flex items-center justify-center">
            <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {data && (
          <div className="space-y-6 mt-6">
            <div className="text-xs text-muted-foreground">
              User since {data.user.created_at?.split("T")[0] ?? "—"}
            </div>

            {/* Tier */}
            <section>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Tier</div>
              <div className="flex items-center gap-2">
                <Badge variant={data.entitlements.tier === "pro" ? "default" : "outline"}>
                  {data.entitlements.tier === "pro" ? "Pro" : "Free"}
                </Badge>
                {data.entitlements.tier === "free" ? (
                  <Button size="sm" onClick={handleGrant} disabled={grantPro.isPending}>
                    Grant Pro
                  </Button>
                ) : (
                  <Button size="sm" variant="outline" onClick={handleRevoke} disabled={revokePro.isPending}>
                    Revoke Pro
                  </Button>
                )}
              </div>
            </section>

            {/* Role */}
            <section>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Role</div>
              <div className="flex items-center gap-2 flex-wrap">
                {data.user.is_env_admin ? (
                  <>
                    <Badge variant="secondary">Admin (env)</Badge>
                    <span className="text-xs text-muted-foreground">
                      Managed via ADMIN_EMAILS — edit env to revoke.
                    </span>
                  </>
                ) : data.user.is_admin ? (
                  <>
                    <Badge variant="secondary">Admin</Badge>
                    {currentUser?.id === userId ? (
                      <span className="text-xs text-muted-foreground">
                        (cannot demote yourself)
                      </span>
                    ) : (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleDemote}
                        disabled={demoteAdmin.isPending}
                      >
                        Demote
                      </Button>
                    )}
                  </>
                ) : (
                  <Button
                    size="sm"
                    onClick={handlePromote}
                    disabled={promoteAdmin.isPending}
                  >
                    Promote to admin
                  </Button>
                )}
              </div>
            </section>

            {/* Override */}
            <section>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Override</div>
              <div className="text-sm mb-2">
                {data.entitlements.hasOverrides ? "Active override applied" : "No override"}
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setOverrideOpen(true)}>
                  {data.entitlements.hasOverrides ? "Edit override" : "Apply override"}
                </Button>
                {data.entitlements.hasOverrides && (
                  <Button size="sm" variant="ghost" onClick={handleClearOverride} disabled={clearOverride.isPending}>
                    Clear override
                  </Button>
                )}
              </div>
            </section>

            {/* Usage */}
            <section>
              <div className="text-xs uppercase tracking-wide text-muted-foreground mb-2">Current usage</div>
              <div className="space-y-1 text-sm">
                <div>Storage: <span className="font-medium">{formatBytes(data.entitlements.usage.totalStorageBytes)}</span></div>
                <div>Split sheets this period: <span className="font-medium">{data.entitlements.usage.splitSheetsThisPeriod}</span></div>
                <div>Zoe queries this period: <span className="font-medium">{data.entitlements.usage.zoeQueriesThisPeriod}</span></div>
                <div>OneClick runs this period: <span className="font-medium">{data.entitlements.usage.oneclickRunsThisPeriod}</span></div>
              </div>
            </section>
          </div>
        )}

        {overrideOpen && userId && (
          <OverrideEditor
            userId={userId}
            currentOverride={data?.override ?? null}
            onDone={() => setOverrideOpen(false)}
          />
        )}
      </SheetContent>
    </Sheet>
  );
};

// ---------------------------------------------------------------------------
// Override editor (inline form within the sheet)
// ---------------------------------------------------------------------------

interface OverrideEditorProps {
  userId: string;
  currentOverride: RawOverride | null;
  onDone: () => void;
}

// Normalize a number-or-null override field to its string form for an <input>.
const numToInput = (v: number | null | undefined): string =>
  v === null || v === undefined ? "" : String(v);

// Normalize a tri-state bool override field to the select's value.
const boolToSelect = (v: boolean | null | undefined): "" | "true" | "false" =>
  v === null || v === undefined ? "" : v ? "true" : "false";

const ALL_INTEGRATIONS = ["google_drive", "slack", "notion"] as const;

const OverrideEditor = ({ userId, currentOverride, onDone }: OverrideEditorProps) => {
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

  // Caps (numbers — null/undefined → empty string)
  const [maxArtists, setMaxArtists] = useState(numToInput(currentOverride?.max_artists));
  const [maxProjects, setMaxProjects] = useState(numToInput(currentOverride?.max_projects));
  const [maxTasks, setMaxTasks] = useState(numToInput(currentOverride?.max_tasks));
  // Storage stored as bytes in DB; admin enters GB for ergonomics
  const [maxStorageGb, setMaxStorageGb] = useState(
    currentOverride?.max_storage_bytes != null
      ? String(currentOverride.max_storage_bytes / (1024 * 1024 * 1024))
      : "",
  );
  const [maxSplitSheets, setMaxSplitSheets] = useState(numToInput(currentOverride?.max_split_sheets_per_month));

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
  const [expiresDays, setExpiresDays] = useState<string>("");  // expiry is set anew each save

  const toggleIntegration = (name: string) => {
    setIntegrations((prev) =>
      prev.includes(name) ? prev.filter((i) => i !== name) : [...prev, name],
    );
  };

  const handleSubmit = async () => {
    const payload: OverridePayloadInput = {};

    // Caps — only include if input is non-empty (means admin wants to set/update)
    if (maxArtists.trim() !== "") payload.max_artists = parseInt(maxArtists, 10);
    if (maxProjects.trim() !== "") payload.max_projects = parseInt(maxProjects, 10);
    if (maxTasks.trim() !== "") payload.max_tasks = parseInt(maxTasks, 10);
    if (maxStorageGb.trim() !== "")
      payload.max_storage_bytes = Math.round(parseFloat(maxStorageGb) * 1024 * 1024 * 1024);
    if (maxSplitSheets.trim() !== "") payload.max_split_sheets_per_month = parseInt(maxSplitSheets, 10);

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

  return (
    <Card className="p-4 mt-6 space-y-3">
      <div className="text-sm font-medium">
        Override fields {currentOverride ? "(pre-filled with current values)" : "(only set what you want to override)"}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="text-xs">
          Max artists
          <Input type="number" value={maxArtists} onChange={(e) => setMaxArtists(e.target.value)} placeholder="—" />
        </label>
        <label className="text-xs">
          Max projects
          <Input type="number" value={maxProjects} onChange={(e) => setMaxProjects(e.target.value)} placeholder="—" />
        </label>
        <label className="text-xs">
          Max tasks
          <Input type="number" value={maxTasks} onChange={(e) => setMaxTasks(e.target.value)} placeholder="—" />
        </label>
        <label className="text-xs">
          Max storage (GB)
          <Input type="number" step="0.1" value={maxStorageGb}
                 onChange={(e) => setMaxStorageGb(e.target.value)} placeholder="—" />
        </label>
        <label className="text-xs">
          Max split sheets / month
          <Input type="number" value={maxSplitSheets}
                 onChange={(e) => setMaxSplitSheets(e.target.value)} placeholder="—" />
        </label>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <label className="text-xs">
          Zoe enabled
          <select className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
                  value={zoe} onChange={(e) => setZoe(e.target.value as typeof zoe)}>
            <option value="">— (use tier default)</option><option value="true">Yes</option><option value="false">No</option>
          </select>
        </label>
        <label className="text-xs">
          OneClick enabled
          <select className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
                  value={oneclick} onChange={(e) => setOneclick(e.target.value as typeof oneclick)}>
            <option value="">— (use tier default)</option><option value="true">Yes</option><option value="false">No</option>
          </select>
        </label>
        <label className="text-xs">
          Registry enabled
          <select className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
                  value={registry} onChange={(e) => setRegistry(e.target.value as typeof registry)}>
            <option value="">— (use tier default)</option><option value="true">Yes</option><option value="false">No</option>
          </select>
        </label>
      </div>

      <div className="space-y-2">
        <label className="flex items-center gap-2 text-xs">
          <input type="checkbox" checked={integrationsOverridden}
                 onChange={(e) => setIntegrationsOverridden(e.target.checked)} />
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

      <label className="text-xs block">
        Reason (optional)
        <Input value={reason} onChange={(e) => setReason(e.target.value)} placeholder="Beta tester comp" />
      </label>
      <label className="text-xs block">
        Expires in N days (optional, leave empty for no expiry)
        <Input type="number" value={expiresDays} onChange={(e) => setExpiresDays(e.target.value)} placeholder="—" />
      </label>

      <p className="text-xs text-muted-foreground border-t border-border pt-3">
        Heads up: deleting a pre-filled value or switching a feature back to "—"
        is silently ignored — Save preserves the existing override field. To
        remove individual overrides, use <strong>Clear override</strong> and
        re-apply with only the fields you want to keep.
      </p>

      <div className="flex justify-end gap-2 pt-2">
        <Button variant="ghost" size="sm" onClick={onDone}>Cancel</Button>
        <Button size="sm" onClick={handleSubmit} disabled={applyOverride.isPending}>Save</Button>
      </div>
    </Card>
  );
};

export default AdminUsers;
