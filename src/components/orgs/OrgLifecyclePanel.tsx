// src/components/orgs/OrgLifecyclePanel.tsx
// Self-serve admin console only (2026-08-15 pricing/teams spec §3/§8):
// covering admin + claim/release, archive/unarchive, and the terminal
// dissolve flow. Enterprise orgs never render this — kind-gated at the call
// site in Organization.tsx's AdminConsole, which is what keeps the enterprise
// console byte-identical to today. Mirrors OrgSettingsPanel's card/button
// idioms (archive's AlertDialog copy in particular).
import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
import { fmtDate } from "@/lib/utils";
import { useClaimCoverage, useReleaseCoverage, useArchiveOrg, useUnarchiveOrg, type OrgDetail } from "@/hooks/useOrgs";
import { DissolveOrgDialog } from "./DissolveOrgDialog";

function coverageCopy(org: OrgDetail, currentUserId?: string): string {
  if (!org.covered_by) return "Nobody is covering this team yet — claim it to keep it active.";
  if (!org.covered_at) return "Coverage was released — any admin with a free slot can claim it.";
  return org.covered_by === currentUserId
    ? "You're currently covering this team — its slot and storage count against your plan."
    : "Another admin is currently covering this team.";
}

export function OrgLifecyclePanel({
  orgId,
  org,
  currentUserId,
}: {
  orgId: string;
  org: OrgDetail;
  currentUserId?: string;
}) {
  const claim = useClaimCoverage();
  const release = useReleaseCoverage();
  const archive = useArchiveOrg();
  const unarchive = useUnarchiveOrg();
  const [dissolveOpen, setDissolveOpen] = useState(false);

  if (org.dissolved_at) {
    return (
      <Card className="p-6">
        <div className="text-[15px] font-semibold">Team coverage &amp; lifecycle</div>
        <p className="text-[13.5px] text-muted-foreground mt-2">
          This team was dissolved on {fmtDate(org.dissolved_at)}. Its artists and members have been
          returned; the record is kept for reference.
        </p>
      </Card>
    );
  }

  const isActiveCoverer = org.covered_by === currentUserId && !!org.covered_at;

  return (
    <Card className="p-6">
      <div className="text-[15px] font-semibold">Team coverage &amp; lifecycle</div>
      <div className="text-[13.5px] text-muted-foreground mt-0.5">
        Who&apos;s paying for this team&apos;s slot and storage, and how to pause, restore, or close it
      </div>

      {!org.archived_at && (
        <div className="mt-4 bg-background border border-border rounded-xl px-4 py-3.5">
          <div className="text-sm font-medium">Covering admin</div>
          <p className="text-[12.5px] text-muted-foreground mt-1 max-w-[460px]">{coverageCopy(org, currentUserId)}</p>
          <div className="mt-3 flex gap-2 flex-wrap">
            {isActiveCoverer ? (
              <Button size="sm" variant="outline" onClick={() => release.mutate({ orgId })} disabled={release.isPending}>
                {release.isPending && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
                Release coverage
              </Button>
            ) : org.status === "lapsed" ? (
              // The page-level StandingBanner already renders the Reactivate
              // CTA for a lapsed org (same useClaimCoverage mutation) — a
              // second claim button here would be a duplicate action with no
              // added context, so this points at the banner instead.
              <p className="text-[12.5px] text-muted-foreground">See the banner above to reactivate this team.</p>
            ) : (
              <Button size="sm" variant="outline" onClick={() => claim.mutate({ orgId })} disabled={claim.isPending}>
                {claim.isPending && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
                Claim coverage
              </Button>
            )}
          </div>
        </div>
      )}

      <div className="mt-5 pt-5 border-t border-border space-y-4">
        {org.archived_at ? (
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div>
              <div className="text-sm font-medium">Archived</div>
              <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[440px]">
                This team is hidden and inactive. Unarchiving needs a free team slot, and enough room
                left in your storage pool for what this team is already holding.
              </p>
            </div>
            <Button size="sm" variant="outline" onClick={() => unarchive.mutate({ orgId })} disabled={unarchive.isPending}>
              {unarchive.isPending && <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />}
              Unarchive
            </Button>
          </div>
        ) : (
          <AlertDialog>
            <div className="flex items-center justify-between gap-4 flex-wrap">
              <div>
                <div className="text-sm font-medium">Archive</div>
                <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[440px]">
                  Frees your team slot and hides the team for everyone. Reversible — unarchive any
                  time, slot and storage allowing.
                </p>
              </div>
              <AlertDialogTrigger asChild>
                <Button size="sm" variant="outline">
                  Archive…
                </Button>
              </AlertDialogTrigger>
            </div>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Archive {org.name}?</AlertDialogTitle>
                <AlertDialogDescription>
                  Members lose access immediately. Credits left in the pool stay held — contact
                  support if you need them refunded or moved. This frees your team slot; unarchive any
                  time to bring it back.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={() => archive.mutate({ orgId })}>Archive</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}

        <div className="flex items-center justify-between gap-4 flex-wrap pt-4 border-t border-destructive/30">
          <div>
            <div className="text-sm font-semibold text-destructive">Dissolve team</div>
            <p className="text-[12.5px] text-muted-foreground mt-0.5 max-w-[440px]">
              Permanently closes the team. Artists return to their creators, purchased credits in the
              pool are forfeited, and this can&apos;t be undone.
            </p>
          </div>
          <Button
            variant="outline"
            className="text-destructive hover:text-destructive flex-none"
            onClick={() => setDissolveOpen(true)}
          >
            Dissolve…
          </Button>
        </div>
      </div>

      <DissolveOrgDialog orgId={orgId} orgName={org.name} open={dissolveOpen} onOpenChange={setDissolveOpen} />
    </Card>
  );
}
