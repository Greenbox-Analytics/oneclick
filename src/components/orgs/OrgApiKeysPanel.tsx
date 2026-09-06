// src/components/orgs/OrgApiKeysPanel.tsx
// Admin console: the partner API keys of a partner-enabled org, with per-key
// spend this period (useOrgUsage().byKey). Rendered by AdminConsole only when
// org.partner_api_enabled — never for members, never for a non-enabled org.
import { useState } from "react";
import { Link } from "react-router-dom";
import { BookOpen, KeyRound, Loader2, Plus } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useOrgUsage } from "@/hooks/useOrgs";
import { usePartnerKeys, useRevokePartnerKey } from "@/hooks/usePartnerKeys";
import { keyRows, type KeyRow as KeyRowData, type KeyStatus } from "@/lib/partnerKeys";
import { fmtDate } from "@/lib/utils";
import { CreateApiKeyDialog } from "./CreateApiKeyDialog";

const STATUS_LABEL: Record<KeyStatus, string> = { active: "Active", revoked: "Revoked", expired: "Expired" };
const STATUS_VARIANT: Record<KeyStatus, "default" | "secondary" | "outline"> = {
  active: "default",
  revoked: "secondary",
  expired: "outline",
};

function KeyRow({
  row,
  usageLoaded,
  onRevoke,
}: {
  row: KeyRowData;
  /** False until the separate usage query resolves — a spend of 0 would read as "this key spent nothing". */
  usageLoaded: boolean;
  onRevoke: (r: KeyRowData) => void;
}) {
  const { key, status, usage } = row;
  return (
    <TableRow>
      <TableCell className="font-medium">{key.label}</TableCell>
      <TableCell className="font-mono text-[12px] text-muted-foreground">{key.key_prefix}…</TableCell>
      <TableCell className="text-muted-foreground">{key.created_by_label ?? "—"}</TableCell>
      <TableCell>
        <Badge variant={STATUS_VARIANT[status]}>{STATUS_LABEL[status]}</Badge>
      </TableCell>
      <TableCell className="text-right tabular-nums">{usageLoaded ? (usage?.credits ?? 0) : "—"}</TableCell>
      <TableCell className="text-right tabular-nums">{usageLoaded ? (usage?.runs ?? 0) : "—"}</TableCell>
      {/* last_used_at is stamped on every resolve; usage.lastUsedAt only on a debited run this period. */}
      <TableCell className="text-muted-foreground">{fmtDate(key.last_used_at ?? usage?.lastUsedAt)}</TableCell>
      <TableCell className="text-right">
        {status === "active" && (
          <Button variant="ghost" size="sm" aria-label={`Revoke key for ${key.label}`} onClick={() => onRevoke(row)}>
            Revoke
          </Button>
        )}
      </TableCell>
    </TableRow>
  );
}

export function OrgApiKeysPanel({ orgId }: { orgId: string }) {
  const { data, isLoading, isError } = usePartnerKeys(orgId);
  const { data: usage, isSuccess: usageLoaded } = useOrgUsage(orgId);
  const revoke = useRevokePartnerKey();
  const [createOpen, setCreateOpen] = useState(false);
  const [showInactive, setShowInactive] = useState(false);
  const [pending, setPending] = useState<KeyRowData | null>(null);

  // Never fall through to the empty state on a failed load: it invites minting
  // a duplicate of a key they already have.
  if (isError) {
    return (
      <Card className="p-6 text-sm text-muted-foreground text-center py-10">
        Couldn&apos;t load API keys. Please try refreshing.
      </Card>
    );
  }

  const rows = keyRows(data?.keys ?? [], usage?.byKey);
  const visible = showInactive ? rows : rows.filter((r) => r.status === "active");
  const inactiveCount = rows.filter((r) => r.status !== "active").length;

  const confirmRevoke = () => {
    if (!pending) return;
    revoke.mutate({ orgId, keyId: pending.key.id }, { onSettled: () => setPending(null) });
  };

  return (
    <Card className="p-5 flex flex-col gap-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h3 className="text-[15px] font-semibold flex items-center gap-2">
            <KeyRound className="w-4 h-4" /> API keys
          </h3>
          <p className="text-[13px] text-muted-foreground">
            Keys your systems use to run royalty calculations through Msanii. Each one is this team&apos;s credential and
            spends from its credits — keep them secret.
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button asChild size="sm" variant="ghost">
            <Link to="/docs?section=api">
              <BookOpen className="w-4 h-4 mr-1.5" /> API docs
            </Link>
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="w-4 h-4 mr-1.5" /> New API key
          </Button>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
        </div>
      ) : rows.length === 0 ? (
        <p className="text-[13px] text-muted-foreground py-6 text-center">No API keys yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Key</TableHead>
                <TableHead>Created by</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Credits this period</TableHead>
                <TableHead className="text-right">Runs</TableHead>
                <TableHead>Last used</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {visible.map((r) => (
                <KeyRow key={r.key.id} row={r} usageLoaded={usageLoaded} onRevoke={setPending} />
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {inactiveCount > 0 && (
        <div>
          <Button variant="ghost" size="sm" onClick={() => setShowInactive((v) => !v)}>
            {showInactive ? "Hide inactive" : `Show inactive (${inactiveCount})`}
          </Button>
        </div>
      )}

      <CreateApiKeyDialog orgId={orgId} open={createOpen} onOpenChange={setCreateOpen} />

      <Dialog open={!!pending} onOpenChange={(o) => !o && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Revoke "{pending?.key.label}"?</DialogTitle>
            <DialogDescription>Anything using this key stops working immediately.</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPending(null)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={confirmRevoke} disabled={revoke.isPending}>
              {revoke.isPending && <Loader2 className="w-4 h-4 mr-1.5 animate-spin" />}
              Revoke key
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
