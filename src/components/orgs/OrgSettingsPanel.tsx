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
import { useUpdateOrg, useArchiveOrg, type OrgDetail } from "@/hooks/useOrgs";

// Client-side-only suggested default (plan Task 12/14, round 5: cut the
// backend env fallback — the sweep reads the STORED column and NULL/0 means
// manual-only, so a runtime env default would never be read. This is purely
// a UI pre-fill when the admin flips the toggle on).
const SUGGESTED_DEFAULT_ALLOWANCE = 500;

export function OrgSettingsPanel({ org }: { org: OrgDetail }) {
  const updateOrg = useUpdateOrg();
  const archiveOrg = useArchiveOrg();

  const [name, setName] = useState(org.name);
  const [allowanceEnabled, setAllowanceEnabled] = useState((org.default_seat_allowance ?? 0) > 0);
  const [allowance, setAllowance] = useState(
    org.default_seat_allowance && org.default_seat_allowance > 0
      ? String(org.default_seat_allowance)
      : String(SUGGESTED_DEFAULT_ALLOWANCE),
  );

  // Re-sync local drafts when the selected org changes (or a save/refetch
  // lands new server values) — same pattern as TeamCardSettings' startEdit.
  useEffect(() => {
    setName(org.name);
    setAllowanceEnabled((org.default_seat_allowance ?? 0) > 0);
    setAllowance(
      org.default_seat_allowance && org.default_seat_allowance > 0
        ? String(org.default_seat_allowance)
        : String(SUGGESTED_DEFAULT_ALLOWANCE),
    );
  }, [org.id, org.name, org.default_seat_allowance]);

  const nameDirty = name.trim() !== org.name && !!name.trim();
  const allowanceValue = Number(allowance);
  const storedAllowance = org.default_seat_allowance ?? 0;
  const allowanceDirty = allowanceEnabled ? allowanceValue !== storedAllowance && allowanceValue > 0 : storedAllowance > 0;

  const handleToggleAllowance = (enabled: boolean) => {
    setAllowanceEnabled(enabled);
    if (enabled && (!allowance || Number(allowance) <= 0)) {
      setAllowance(String(SUGGESTED_DEFAULT_ALLOWANCE));
    }
  };

  const handleSave = () => {
    const fields: { orgId: string; name?: string; default_seat_allowance?: number | null } = { orgId: org.id };
    if (nameDirty) fields.name = name.trim();
    if (allowanceDirty) fields.default_seat_allowance = allowanceEnabled ? allowanceValue : null;
    updateOrg.mutate(fields);
  };

  const canSave = nameDirty || allowanceDirty;

  return (
    <Card className="p-6">
      <div className="text-[15px] font-semibold">Settings</div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">Organization name and seat defaults</div>

      <div className="mt-4 space-y-5">
        <div className="space-y-2">
          <Label htmlFor="org-name">Organization name</Label>
          <Input id="org-name" value={name} onChange={(e) => setName(e.target.value)} />
        </div>

        <div className="bg-background border border-border rounded-xl px-4 py-3.5">
          <div className="flex items-center justify-between gap-4">
            <div>
              <div className="text-sm font-medium">Automatic monthly top-up</div>
              <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[420px]">
                Every active seat is topped back up to this amount each month, pool permitting. Leave
                off to allocate credits manually.
              </p>
            </div>
            <Switch checked={allowanceEnabled} onCheckedChange={handleToggleAllowance} />
          </div>
          {allowanceEnabled && (
            <div className="mt-3 max-w-[160px] space-y-1.5">
              <Label htmlFor="org-allowance" className="text-xs">
                Credits per seat / month
              </Label>
              <Input
                id="org-allowance"
                type="number"
                min={1}
                value={allowance}
                onChange={(e) => setAllowance(e.target.value)}
              />
            </div>
          )}
        </div>

        <Button onClick={handleSave} disabled={!canSave || updateOrg.isPending}>
          {updateOrg.isPending && <Loader2 className="w-4 h-4 mr-2 animate-spin" />}
          Save changes
        </Button>
      </div>

      <div className="mt-6 pt-5 border-t border-destructive/30">
        <div className="text-sm font-semibold text-destructive">Archive organization</div>
        <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[520px]">
          Reclaim every seat&apos;s credits back to the pool first (Seats table below) — archiving is
          blocked while any seat still holds a balance. Archiving hides the organization for everyone
          and can&apos;t be undone from the app.
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
