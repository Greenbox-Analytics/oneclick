// src/components/orgs/OrgSettingsPanel.tsx
// Admin console: org name, default member limit, and archive.
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
import { useUpdateOrg, useArchiveOrg, type OrgDetail } from "@/hooks/useOrgs";

// Client-side-only suggested default (plan Task 12/14, round 5: cut the
// backend env fallback — the sweep reads the STORED column and NULL/0 means
// manual-only, so a runtime env default would never be read. This is purely
// a UI pre-fill when the admin flips the toggle on).
const SUGGESTED_DEFAULT_CAP = 2000;

export function OrgSettingsPanel({ org }: { org: OrgDetail }) {
  const updateOrg = useUpdateOrg();
  const archiveOrg = useArchiveOrg();
  const [name, setName] = useState(org.name);
  const [capEnabled, setCapEnabled] = useState((org.default_member_cap ?? 0) > 0);
  const [cap, setCap] = useState(
    org.default_member_cap && org.default_member_cap > 0 ? String(org.default_member_cap) : String(SUGGESTED_DEFAULT_CAP),
  );

  // Re-sync local drafts when the selected org changes (or a save/refetch lands
  // new server values) — same pattern as TeamCardSettings' startEdit.
  useEffect(() => {
    setName(org.name);
    setCapEnabled((org.default_member_cap ?? 0) > 0);
    setCap(
      org.default_member_cap && org.default_member_cap > 0
        ? String(org.default_member_cap)
        : String(SUGGESTED_DEFAULT_CAP),
    );
  }, [org.id, org.name, org.default_member_cap, org.monthly_dispersal_credits]);

  const nameDirty = name.trim() !== org.name && !!name.trim();
  const capValue = Number(cap);
  const storedCap = org.default_member_cap ?? 0;
  const dispersalValue = org.monthly_dispersal_credits ?? 0;
  const capDirty = capEnabled ? capValue !== storedCap && capValue > 0 : storedCap > 0;

  const handleToggleCap = (enabled: boolean) => {
    setCapEnabled(enabled);
    if (enabled && (!cap || Number(cap) <= 0)) setCap(String(SUGGESTED_DEFAULT_CAP));
  };

  const handleSave = () => {
    const fields: { orgId: string; name?: string; default_member_cap?: number | null } = { orgId: org.id };
    if (nameDirty) fields.name = name.trim();
    if (capDirty) fields.default_member_cap = capEnabled ? capValue : null;
    updateOrg.mutate(fields);
  };

  const canSave = nameDirty || capDirty;
  const saving = updateOrg.isPending;
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

        {/* Read-only: the dispersal is the contract, and Msanii sets it. Editable
            here it would be free credits — anyone can create an org and is auto-made
            its admin, and dispersed credits count toward activation. */}
        <div className="bg-background border border-border rounded-xl px-4 py-3.5">
          <div className="text-sm font-medium">Monthly credits</div>
          <div className="text-[26px] font-bold tracking-tight mt-1 tabular-nums">
            {dispersalValue.toLocaleString()}{" "}
            <span className="text-sm font-normal text-muted-foreground">credits / month</span>
          </div>
          <p className="text-[12.5px] text-muted-foreground mt-1 max-w-[460px]">
            {dispersalValue > 0 ? (
              <>
                Added to your shared pool at the start of each month under your contract. Unused
                monthly credits don&apos;t carry over — credits you buy separately never expire. To
                change this, talk to us.
              </>
            ) : (
              <>
                No monthly contract on this organization — you&apos;re running on credits you buy,
                which never expire. Talk to us to set up a monthly allocation.
              </>
            )}
          </p>
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
                Members lose access immediately. Credits left in the pool stay held — contact
                support if you need them refunded or moved. Artists owned by the team stay with it
                but become unreachable while it&apos;s archived.
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
