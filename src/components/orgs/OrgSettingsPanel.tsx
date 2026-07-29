// src/components/orgs/OrgSettingsPanel.tsx
// Admin console: org name, default seat allowance (the sweep's monthly
// auto-top-up), and archive (reclaim-all-first guard).
import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useUpdateOrg, useArchiveOrg, useSetOrgDispersal, type OrgDetail } from "@/hooks/useOrgs";

// Client-side-only suggested default (plan Task 12/14, round 5: cut the
// backend env fallback — the sweep reads the STORED column and NULL/0 means
// manual-only, so a runtime env default would never be read. This is purely
// a UI pre-fill when the admin flips the toggle on).
const SUGGESTED_DEFAULT_CAP = 2000;
const SUGGESTED_DISPERSAL = 10000;

export function OrgSettingsPanel({ org }: { org: OrgDetail }) {
  const updateOrg = useUpdateOrg();
  const archiveOrg = useArchiveOrg();
  const setDispersal = useSetOrgDispersal();

  const [name, setName] = useState(org.name);
  const [dispersal, setDispersalValue] = useState(String(org.monthly_dispersal_credits ?? 0));
  const [capEnabled, setCapEnabled] = useState((org.default_member_cap ?? 0) > 0);
  const [cap, setCap] = useState(
    org.default_member_cap && org.default_member_cap > 0 ? String(org.default_member_cap) : String(SUGGESTED_DEFAULT_CAP),
  );

  // Re-sync local drafts when the selected org changes (or a save/refetch lands
  // new server values) — same pattern as TeamCardSettings' startEdit.
  useEffect(() => {
    setName(org.name);
    setDispersalValue(String(org.monthly_dispersal_credits ?? 0));
    setCapEnabled((org.default_member_cap ?? 0) > 0);
    setCap(
      org.default_member_cap && org.default_member_cap > 0
        ? String(org.default_member_cap)
        : String(SUGGESTED_DEFAULT_CAP),
    );
  }, [org.id, org.name, org.default_member_cap, org.monthly_dispersal_credits]);

  const nameDirty = name.trim() !== org.name && !!name.trim();
  const capValue = Number(cap);
  const dispersalValue = Number(dispersal);
  const storedCap = org.default_member_cap ?? 0;
  const storedDispersal = org.monthly_dispersal_credits ?? 0;
  const capDirty = capEnabled ? capValue !== storedCap && capValue > 0 : storedCap > 0;
  const dispersalDirty = Number.isFinite(dispersalValue) && dispersalValue >= 0 && dispersalValue !== storedDispersal;

  const handleToggleCap = (enabled: boolean) => {
    setCapEnabled(enabled);
    if (enabled && (!cap || Number(cap) <= 0)) setCap(String(SUGGESTED_DEFAULT_CAP));
  };

  const handleSave = () => {
    if (nameDirty) updateOrg.mutate({ orgId: org.id, name: name.trim() });
    // The contract dials live on their own endpoint: they're the commercial
    // terms, not display preferences, and the dispersal only takes effect at the
    // next period boundary.
    if (dispersalDirty || capDirty) {
      setDispersal.mutate({
        orgId: org.id,
        monthlyDispersalCredits: dispersalDirty ? dispersalValue : storedDispersal,
        defaultMemberCap: capEnabled ? capValue : null,
      });
    }
  };

  const canSave = nameDirty || capDirty || dispersalDirty;
  const saving = updateOrg.isPending || setDispersal.isPending;
  const totalCommitted = capEnabled && capValue > 0 ? capValue * Math.max(1, org.member_count) : 0;

  return (
    <Card className="p-6">
      <div className="text-[15px] font-semibold">Settings</div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">Organization name and contract terms</div>

      <div className="mt-4 space-y-5">
        <div className="space-y-2">
          <Label htmlFor="org-name">Organization name</Label>
          <Input id="org-name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>

        <div className="bg-background border border-border rounded-xl px-4 py-3.5 space-y-1.5">
          <Label htmlFor="org-dispersal" className="text-sm font-medium">
            Monthly credits
          </Label>
          <p className="text-[12.5px] text-muted-foreground max-w-[460px]">
            Added to the shared pool at the start of each month under your contract. Unused monthly
            credits don&apos;t carry over — credits you buy separately never expire.
          </p>
          <Input
            id="org-dispersal"
            type="number"
            min={0}
            className="max-w-[180px]"
            value={dispersal}
            onChange={(e) => setDispersalValue(e.target.value)}
            placeholder={String(SUGGESTED_DISPERSAL)}
          />
        </div>

        <div className="bg-background border border-border rounded-xl px-4 py-3.5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-medium">Default member limit</div>
              <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[440px]">
                How much of the pool each member can use per month, unless you set theirs
                individually. Limits are ceilings, not reservations — they can add up to more than
                the pool holds, since most members won&apos;t reach theirs.
              </p>
            </div>
            <Switch checked={capEnabled} onCheckedChange={handleToggleCap} />
          </div>
          {capEnabled && (
            <div className="mt-3 max-w-[180px] space-y-1.5">
              <Label htmlFor="org-cap" className="text-xs">
                Credits per member / month
              </Label>
              <Input id="org-cap" type="number" min={1} value={cap} onChange={(e) => setCap(e.target.value)} />
            </div>
          )}
          {capEnabled && capValue > 0 && dispersalValue > 0 && (
            <p className="text-[12px] text-muted-foreground mt-2">
              {org.member_count} member{org.member_count === 1 ? "" : "s"} × {capValue.toLocaleString()} ={" "}
              {totalCommitted.toLocaleString()} credits of limit against {dispersalValue.toLocaleString()} dispersed
              {totalCommitted > dispersalValue
                ? " — fine unless everyone maxes out at once."
                : "."}
            </p>
          )}
        </div>

        <Button onClick={handleSave} disabled={!canSave || saving}>
          {saving && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
          Save changes
        </Button>
      </div>

      <div className="mt-6 pt-5 border-t border-destructive/30">
        <div className="text-sm font-semibold text-destructive">Archive organization</div>
        <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[520px]">
          Archiving hides the organization for everyone and can&apos;t be undone from the app. Credits
          left in the pool stay there — contact support if you need them refunded or moved.
        </p>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              variant="outline"
              className="mt-3 text-destructive hover:text-destructive"
              disabled={!!org.archived_at}
            >
              {org.archived_at ? "Archived" : "Archive organization…"}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Archive {org.name}?</AlertDialogTitle>
              <AlertDialogDescription>
                This only succeeds once every seat is at a zero balance — reclaim credits from the
                Seats table first if this is rejected. Members lose access immediately once archived.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction onClick={() => archiveOrg.mutate({ orgId: org.id })}>Archive</AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </Card>
  );
}
